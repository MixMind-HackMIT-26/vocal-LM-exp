import unittest
import base64
import copy
import json
from unittest.mock import Mock, patch

from harness import Session, run_turn, sample_followup


BASE = {"name": "Baseline", "pours": [{"channel": 1, "ml": 60},
                                      {"channel": 5, "ml": 40}, {"channel": 6, "ml": 50}]}


class OpenRouterTests(unittest.TestCase):
    def test_audio_and_tool_round_trip(self):
        session = Session()
        session.begin_turn(BASE)
        history, log, requests = [], [], []
        def response(name, arguments, call_id):
            return {"choices": [{"message": {"role": "assistant", "content": None,
                    "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
                    "tool_calls": [{"id": call_id, "type": "function", "function": {
                        "name": name, "arguments": json.dumps(arguments)}}]}}]}
        responses = iter([
            response("revise_drink", {"expected_version": 0, "name": "Test",
                     "explanation": "Tart", "set_amounts": [{"channel": 5, "ml": 20}]}, "edit-1"),
            response("pour_sample", {"recipe_version": 1}, "sample-1"),
            {"choices": [{"message": {"role": "assistant", "content":
                "This version has less apple juice. Would you prefer it more tart?"}}]}])
        def fake_complete(key, payload):
            self.assertEqual(key, "test-key")
            requests.append(copy.deepcopy(payload))
            return next(responses)
        with patch("harness.complete", side_effect=fake_complete), patch("builtins.print"):
            run_turn("test-key", "google/gemini-2.5-flash", history, session,
                     Mock(read_bytes=Mock(return_value=b"test audio")),
                     {"decipher_proposal": BASE, "features": {}}, "combined", log)
        audio = requests[0]["messages"][1]["content"][1]["input_audio"]
        self.assertEqual(audio["format"], "m4a")
        self.assertEqual(base64.b64decode(audio["data"]), b"test audio")
        self.assertEqual(requests[0]["tools"][0]["function"]["parameters"]["type"], "object")
        self.assertEqual(requests[1]["messages"][-1]["tool_call_id"], "edit-1")
        self.assertEqual(requests[1]["messages"][-2]["reasoning_details"][0]["data"], "opaque")
        self.assertEqual(history[-2]["tool_call_id"], "sample-1")
        self.assertIn("Would you prefer", history[-1]["content"])
        self.assertEqual(requests[2]["tool_choice"], "none")
        self.assertEqual(requests[2]["messages"][-1]["tool_call_id"], "sample-1")
        self.assertEqual(log[-1]["phase"], "sample_followup")
        self.assertEqual(len(session.ledger), 1)

    def test_followup_failure_keeps_conversation_usable(self):
        history, log = [], []
        with patch("harness.complete", side_effect=RuntimeError("timeout")), patch("builtins.print"):
            sample_followup("key", "model", history, log)
        self.assertEqual(log[-1]["kind"], "followup_fallback")
        self.assertIn("simulated sample", history[-1]["content"])

    def test_followup_never_executes_unexpected_tool(self):
        history, log = [], []
        response = {"choices": [{"message": {"role": "assistant", "tool_calls": [
            {"function": {"name": "pour_sample"}}]}}]}
        with patch("harness.complete", return_value=response), patch("harness.Session.call") as call, patch("builtins.print"):
            sample_followup("key", "model", history, log)
        call.assert_not_called()
        self.assertNotIn("tool_calls", history[-1])

    def test_bad_tool_arguments_return_error(self):
        session = Session()
        session.begin_turn(BASE)
        history = []
        responses = [{"choices": [{"message": {"role": "assistant", "tool_calls": [
            {"id": "bad", "type": "function", "function": {
                "name": "revise_drink", "arguments": "not json"}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "Please clarify."}}]}]
        with patch("harness.complete", side_effect=responses), patch("builtins.print"):
            run_turn("key", "model", history, session, None,
                     {"decipher_proposal": BASE}, "features-only", [])
        self.assertFalse(json.loads(history[2]["content"])["ok"])
        self.assertEqual(session.version, 0)


class SessionTests(unittest.TestCase):
    def proposal(self, session, changes=None):
        return session.call("revise_drink", {"expected_version": session.version,
                            "set_amounts": changes or [], "name": "Test drink", "explanation": "Test"})

    def test_server_diff_add_remove_update_and_no_change(self):
        s = Session()
        s.begin_turn(BASE)
        result = self.proposal(s, [{"channel": 1, "ml": 45}, {"channel": 5, "ml": 0},
                                   {"channel": 2, "ml": 30}])
        self.assertTrue(result["ok"])
        self.assertEqual(result["diff_base"], "internal_baseline")
        self.assertEqual([(d["channel"], d["before_ml"], d["after_ml"], d["delta_ml"])
                          for d in result["amount_diff"]],
                         [(1, 60, 45, -15), (2, 0, 30, 30), (5, 40, 0, -40)])
        self.assertEqual(result["state"]["recipe"]["pours"][-1], {"channel": 6, "ml": 50})
        s.call("pour_sample", {"recipe_version": 1})
        s.begin_turn(BASE)
        result = self.proposal(s)
        self.assertEqual(result["amount_diff"], [])
        self.assertEqual(result["diff_base"], "previous_sampled_recipe")
        self.assertEqual(result["previous_version"], 1)

    def test_revision_limits_and_continuity(self):
        s = Session()
        for version in range(1, 4):
            s.begin_turn(BASE)
            self.assertTrue(self.proposal(s, [{"channel": 5, "ml": 20}])["ok"])
            self.assertFalse(self.proposal(s)["ok"])
            self.assertTrue(s.call("pour_sample", {"recipe_version": version})["ok"])
        s.begin_turn(BASE)
        self.assertEqual(s.recipe["pours"][1]["ml"], 20)
        self.assertFalse(self.proposal(s)["ok"])

    def test_invalid_diff_is_atomic_and_stale_version_rejected(self):
        s = Session()
        s.begin_turn(BASE)
        self.assertFalse(self.proposal(s, [{"channel": 5, "ml": 900}])["ok"])
        self.assertEqual(s.recipe, BASE)
        self.assertEqual(s.version, 0)
        self.assertTrue(self.proposal(s)["ok"])
        self.assertFalse(s.call("pour_sample", {"recipe_version": 0})["ok"])

    def test_sample_and_final_do_not_repeat(self):
        s = Session()
        s.begin_turn(BASE)
        self.proposal(s)
        args = {"recipe_version": 1}
        s.call("pour_sample", args)
        self.assertTrue(s.call("pour_sample", args)["duplicate"])
        self.assertEqual(len(s.ledger), 1)
        self.assertAlmostEqual(sum(p["ml"] for p in s.ledger[0]["pours"]), 15, places=2)
        self.assertFalse(s.call("finish_and_pour", args)["ok"])
        s.begin_turn(BASE)
        self.assertTrue(s.call("finish_and_pour", args)["ok"])
        self.assertEqual(len(s.ledger), 1)
        s.confirm_final()
        s.confirm_final()
        self.assertTrue(s.call("finish_and_pour", args)["duplicate"])
        self.assertEqual(len(s.ledger), 2)

    def test_cancel_prevents_final(self):
        s = Session()
        s.begin_turn(BASE)
        self.proposal(s)
        s.call("cancel_session", {})
        self.assertFalse(s.confirm_final()["ok"])
        self.assertFalse(s.call("pour_sample", {"recipe_version": 1})["ok"])
        self.assertEqual(s.ledger, [])


if __name__ == "__main__":
    unittest.main()
