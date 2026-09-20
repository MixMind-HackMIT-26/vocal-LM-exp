import unittest
import json

from report import recording_scripts, render, source_lines


class ReportTests(unittest.TestCase):
    def test_cascade_reasoning_and_tool_result(self):
        data = {"events": [
            {"kind": "input", "file": "flat.m4a", "evidence": {}},
            {"kind": "model_response", "response": {"choices": [{"message": {
                "content": "Try this.", "reasoning": "Visible reasoning",
                "reasoning_details": [{"type": "reasoning.text", "text": "Visible reasoning"},
                                      {"type": "reasoning.encrypted", "data": "SECRET_CIPHERTEXT"}],
                "tool_calls": [{"id": "c1", "function": {"name": "revise_drink", "arguments": "{}"}}]
            }}]}},
            {"kind": "tool", "name": "revise_drink", "args": {}, "result": {"ok": False, "error": "Stale version"}},
            {"kind": "operator", "command": "/cancel", "result": {"ok": True}}
        ]}
        text = render(data, "example.json", {"flat.m4a": ["I'm fine.", "Quiet"]})
        self.assertIn("Step 1: flat.m4a", text)
        self.assertIn("Step 2: /cancel", text)
        self.assertIn("I'm fine.", text)
        self.assertIn("encrypted", text)
        self.assertNotIn("SECRET_CIPHERTEXT", text)
        self.assertEqual(text.count("Visible reasoning"), 1)
        self.assertLess(text.index("Tool request:"), text.index("Tool result ("))
        self.assertIn("Stale version", text)

    def test_unknown_recording_and_empty_run(self):
        text = render({"events": [{"kind": "input", "file": "custom.m4a"}]}, "run.json")
        self.assertIn("no transcript stored", text)
        self.assertIn("No events recorded", render({"events": []}, "empty.json"))

    def test_recording_scripts_match_readme(self):
        scripts = recording_scripts()
        self.assertEqual(len(scripts), 10)
        self.assertIn("I'm fine, really.", scripts["flat.m4a"][0])

    def test_exact_source_lines_and_compact_summaries(self):
        data = {"events": [{"kind": "model_response", "response": {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 2607, "completion_tokens": 28, "total_tokens": 2635,
                      "cost": 0.00206025}}},
            {"kind": "tool", "name": "pour_sample", "result": {"ok": True, "event": {
                "kind": "sample", "version": 1, "pours": [{"channel": 1, "ml": 45}]}}}], "state": {}}
        text = json.dumps(data, indent=4)
        positions = source_lines(text)
        usage_line = positions[('events', 0, 'response', 'usage')]
        self.assertIn('"usage": {', text.splitlines()[usage_line-1])
        result_line = positions[('events', 1, 'result')]
        self.assertIn('"result": {', text.splitlines()[result_line-1])
        report = render(data, 'run.json', source_text=text, source_path='run.json')
        self.assertIn('run.json:%d>' % usage_line, report)
        self.assertIn('2607 input + 28 output = 2635 tokens; $0.002060', report)
        self.assertIn('CH 1 (Orange juice): 45 mL', report)
        self.assertNotIn('"prompt_tokens"', report)
        compact = json.dumps(data)
        self.assertTrue(all(line == 1 for line in source_lines(compact).values()))


if __name__ == "__main__":
    unittest.main()
