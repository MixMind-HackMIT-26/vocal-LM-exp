"""Recorded audio -> OpenRouter -> versioned recipes and simulated pours only."""
import argparse
import base64
import copy
import importlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import textwrap
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent / "voice_decipher_2"
sys.path.insert(0, str(BACKEND))
from catalog import CATALOG, SAMPLE_RATIO, validate
from negotiation import Session
PROMPT = """You are MixMind, a concise conversational drink bartender.
You speak as the complete MixMind product: voice analysis, recipe selection,
and conversation are all parts of YOU, not separate parties. The guest has
not seen your internal baseline or tool operations. Never narrate how you
corrected an algorithm, overrode a decipher, or changed an unseen proposal.
Separate internal planning from the guest's experience:
- For the FIRST sampled proposal, introduce the finished drink in present tense:
  "Your drink pairs citrus with ginger ale for a bright, fizzy finish. Does that
  sound like your kind of drink?" Do NOT say "I added ginger ale", "I made it
  brighter", or "I adjusted the recipe" relative to the internal baseline.
- For LATER proposals, describe changes only relative to the previous proposal
  the guest actually received, and connect them to their stated feedback:
  "This version is less sweet, with more citrus bite. Is that closer to what
  you had in mind?" Internal baseline changes are never the comparison point.
- If no guest-visible recipe changed, do not invent a change. Introduce or
  describe the current drink naturally. Ask about it without assuming a tasting
  or preference the guest has not reported.
Apply this product voice to all user-facing text, including drink explanations,
recipe rationale, messages before tools, and the closing feedback question.
Use plain sensory language such as "fizzy" instead of "effervescence". Be warm,
specific, and brief, not a technical changelog. Ingredient statements must match
the validated recipe. Do not claim more or less of something unless the actual
guest-visible recipes support that comparison.
Listen to the attached user audio. Acoustic measurements are observations, not
proof of emotions. Respect corrections and explicit ingredient exclusions.
The catalog is the deployed backend bottle mapping. Final servings have 2-6
ingredients, 10-60 ml each, lime cordial at most 20 ml, total at most 130 ml.
Use set_exclusions to persist explicit excluded channels before proposing edits.
On the first turn, use the decipher proposal as the starting point. On later
turns edit the CURRENT recipe, not the new decipher proposal. Short utterances
are weak acoustic evidence. Preserve preferences from all preceding turns.
For a drink proposal call revise_drink, then pour_sample for its returned
version. Supply set_amounts as absolute target mL, never arithmetic deltas.
Use the server's returned amount_diff for exact comparisons; its baseline is
internal on the first proposal and the previous sampled recipe thereafter.
The complete recipe in the returned state is authoritative.
You may make at most one revision per audio turn, and three sampled
proposals total (initial plus two revisions). Ask a brief question if ambiguous;
do not revise just because new audio arrived. A user can accept the current
drink without another revision. Call finish_and_pour only for explicit user
acceptance; the harness separately requires operator confirmation. Call
cancel_session for cancellation. Never call finish_and_pour in the same turn
as a sample: wait for actual feedback. After a sample, give a brief conversational
reply about the proposal and ask for feedback before waiting for the next turn.
Tools simulate dispensing; never claim real liquid was dispensed.
Keep replies to one or two short sentences, under 50 words total.
Use the returned validated recipe as truth.
If a tool fails, correct the request or explain; do not invent success.
"""


def declarations():
    def tool(name, description, properties, required):
        return dict(name=name, description=description,
                    parameters=dict(type="OBJECT", properties=properties, required=required))
    integer = {"type": "INTEGER"}
    text = {"type": "STRING"}
    return [
        tool("revise_drink", "Commit a proposal. set_amounts contains absolute target final-serving mL, "
             "NOT deltas; 0 removes an ingredient. Omitted channels retain their amounts. "
             "Empty set_amounts keeps current amounts. Returns a validated full recipe in state and "
             "server-computed amount_diff. Every successful call consumes one of three updates.",
             {"expected_version": integer, "name": text, "explanation": text,
              "set_amounts": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                  "channel": integer, "ml": integer}, "required": ["channel", "ml"]}}},
             ["expected_version", "name", "explanation", "set_amounts"]),
        tool("set_exclusions", "Persist explicitly excluded ingredients for the session.",
             {"channels": {"type": "ARRAY", "items": integer}}, ["channels"]),
        tool("pour_sample", "Simulate an 8 percent sample in a SEPARATE cup, once per version.",
             {"recipe_version": integer}, ["recipe_version"]),
        tool("finish_and_pour", "Request final dispensing after explicit acceptance; operator confirms.",
             {"recipe_version": integer}, ["recipe_version"]),
        tool("cancel_session", "Cancel without dispensing a final drink.", {}, []),
    ]


def ffmpeg_path():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("FFmpeg is unavailable. Run: python -m pip install -r requirements.txt "
                           "using the same Python environment as this harness.") from exc


def analyze_audio(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Audio file not found: %s" % path)
    sys.path.insert(0, str(BACKEND)) if str(BACKEND) not in sys.path else None
    features = importlib.import_module("features")
    bartender = importlib.import_module("local_bartender")
    with tempfile.TemporaryDirectory(prefix="mixmind-exp-") as tmp:
        wav = Path(tmp) / "utterance.wav"
        subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-i", str(path), "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
        import wave
        with wave.open(str(wav)) as audio:
            duration = audio.getnframes() / audio.getframerate()
        if not 0 < duration <= 25:
            raise ValueError("Record a clip longer than 0 and no longer than 25 seconds")
        measurements = features.extract(str(wav))
    if measurements["duration_s"] < 0.25:
        raise ValueError("No usable speech detected")
    return {"features": measurements, "axes": bartender.axes(measurements),
            "weak_acoustic_evidence": measurements["duration_s"] < 3,
            "decipher_proposal": bartender.recipe(measurements)}


def complete(api_key, payload):
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError("OpenRouter HTTP %s: %s" %
                           (exc.code, exc.read().decode("utf-8", errors="replace"))) from exc
    if result.get("error"):
        raise RuntimeError("OpenRouter error: " + json.dumps(result["error"]))
    return result


def openrouter_tools():
    def schema(value):
        if isinstance(value, dict):
            return {k: v.lower() if k == "type" and isinstance(v, str) else schema(v)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [schema(v) for v in value]
        return value
    return [{"type": "function", "function": schema(d)} for d in declarations()]


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show_json(value):
    print(json.dumps(value, indent=2, ensure_ascii=True))


def show_recipe(recipe, previous=None):
    if not recipe:
        print("  No recipe yet.")
        return
    print("  " + recipe["name"])
    old = {p["channel"]: p["ml"] for p in (previous or {}).get("pours", [])}
    current = {p["channel"]: p["ml"] for p in recipe["pours"]}
    print("\n  CH  INGREDIENT                 ML     CHANGE")
    print("  --  ---------------------  ------  ---------")
    for ch in sorted(current.keys() | old.keys()):
        ml = current.get(ch, 0)
        delta = "%+g ml" % (ml - old.get(ch, 0)) if previous else "-"
        print("  %2d  %-21s  %6g  %9s" % (ch, CATALOG[ch]["name"], ml, delta))
    print("  %-25s  %6g ml" % ("TOTAL", sum(current.values())))
    if recipe.get("rationale"):
        print("\n" + textwrap.fill(recipe["rationale"], width=72,
                                  initial_indent="  ", subsequent_indent="  "))


def show_session(session):
    section("SESSION RESULT | turn %d | recipe v%d" % (session.turn, session.version))
    show_recipe(session.recipe)
    samples = sum(e["kind"] == "sample" for e in session.ledger)
    finals = sum(e["kind"] == "final" for e in session.ledger)
    print("\n  Status: %s | Updates left: %d | Samples: %d | Final pours: %d" %
          (session.status, 3 - session.version, samples, finals))
    if session.pending_final:
        print("  NEXT: /pour to confirm the final serving, or /cancel.")
    elif session.status == "active":
        print("  NEXT: enter another .m4a file, /pour, or /cancel.")


def sample_followup(api_key, model, history, log):
    instruction = ("The sample tool has completed. Respond directly to the guest now, in one or two "
                   "short sentences and fewer than 50 words. Briefly describe the proposed drink or "
                   "a guest-visible change, then ask one specific taste-preference question. "
                   "Check the successful sample events: if this is the first sampled proposal, "
                   "introduce the WHOLE drink in present tense, with no added/changed/adjusted "
                   "language referring to the hidden decipher baseline. For later samples, compare "
                   "only with the previous sampled recipe, not this turn's decipher proposal, "
                   "and ground the change in the guest's feedback. You are MixMind as one unified "
                   "bartender, not a commentator on another algorithm's work. Use the tool "
                   "results as truth. This is a simulation: do not claim liquid was actually poured "
                   "or tasted. Do not call tools, show JSON, discuss internal rules, or infer feedback "
                   "the user has not given. If no updates remain, ask whether to finalize or cancel "
                   "instead of offering another revision.")
    start = time.monotonic()
    try:
        response = complete(api_key, {
            "model": model, "messages": [{"role": "system", "content": PROMPT + "\n" + instruction}] + history,
            "temperature": 0.2, "tools": openrouter_tools(), "tool_choice": "none",
            "provider": {"require_parameters": True}, "stream": False})
        log.append({"kind": "model_response", "phase": "sample_followup",
                    "latency_s": round(time.monotonic() - start, 3), "response": response})
        section("RAW RESPONSE | POST-SAMPLE MESSAGE")
        show_json(response)
        choices = response.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        if message.get("tool_calls") or not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Expected a text-only post-sample response")
        history.append(message)
        text = message["content"]
    except (RuntimeError, OSError, ValueError) as exc:
        # The sample already executed; a messaging failure must not retry its tools.
        text = "Your simulated sample is ready to review. Would you like to keep this version or cancel?"
        log.append({"kind": "followup_fallback", "error": str(exc), "message": text})
        history.append({"role": "assistant", "content": text})
    section("BARTENDER MESSAGE | YOUR TURN")
    print(text)


def run_turn(api_key, model, history, session, path, evidence, mode, log):
    context = {"session": session.snapshot(), "catalog": CATALOG,
               "decipher_proposal": evidence["decipher_proposal"]}
    if mode != "audio-only":
        context.update(evidence)
    parts = [{"type": "text", "text": json.dumps(context)}]
    if mode != "features-only":
        parts.append({"type": "input_audio", "input_audio": {
            "data": base64.b64encode(path.read_bytes()).decode("ascii"), "format": "m4a"}})
    else:
        parts.append({"type": "text", "text": "No audio or transcript in this ablation. Propose from features."})
    section("TURN %d | INPUTS" % session.turn)
    print("  Model: %s | Mode: %s" % (model, mode))
    print("  Audio: %s" % (str(path) if mode != "features-only" else "not sent (features-only)"))
    print("  Prior conversation messages: %d" % len(history))
    print("  Audio payload: base64 M4A (binary omitted from terminal)") if mode != "features-only" else None
    print("\n  Structured context sent to the model:")
    show_json(context)
    history.append({"role": "user", "content": parts})
    for round_index in range(1, 7):
        print("\n  Waiting for model response (round %d)..." % round_index, flush=True)
        start = time.monotonic()
        response = complete(api_key, {
            "model": model, "messages": [{"role": "system", "content": PROMPT}] + history,
            "temperature": 0.2, "tools": openrouter_tools(), "tool_choice": "auto",
            "provider": {"require_parameters": True}, "stream": False})
        latency = round(time.monotonic() - start, 3)
        log.append({"kind": "model_response", "latency_s": latency,
                    "response": response})
        section("RAW RESPONSE | round %d | %.2f s" % (round_index, latency))
        show_json(response)
        if not response.get("choices") or not response["choices"][0].get("message"):
            raise RuntimeError("OpenRouter returned no message; inspect the run log")
        content = response["choices"][0]["message"]
        history.append(content)  # Preserve tool IDs and provider reasoning details.
        replies, stop, sampled = [], False, False
        if content.get("content"):
            section("BARTENDER MESSAGE")
            print(content["content"])
        for call in content.get("tool_calls") or []:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            before = copy.deepcopy(session.recipe)
            section("TOOL CALL | %s | id=%s" % (name, call["id"]))
            try:
                arguments = json.loads(arguments)
                print("Arguments:")
                show_json(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object")
                result = ({"ok": False, "error": "Wait for the next user turn"} if stop
                          else session.call(name, arguments))
            except (ValueError, TypeError) as exc:
                result = {"ok": False, "error": "Invalid tool arguments: " + str(exc)}
                print("Invalid arguments:")
                show_json(arguments)
            print("\nResult: %s" % ("OK" if result.get("ok") else "ERROR"))
            if result.get("state"):
                state = result["state"]
                show_json({k: v for k, v in state.items() if k not in ("recipe", "ledger")})
                show_recipe(state["recipe"], before)
                if "amount_diff" in result:
                    print("\n  Server-verified changes (%s):" % result["diff_base"])
                    for change in result["amount_diff"]:
                        print("  CH %d (%s): %g -> %g mL (%+g mL)" %
                              (change["channel"], change["ingredient"], change["before_ml"],
                               change["after_ml"], change["delta_ml"]))
                    if not result["amount_diff"]:
                        print("  No ingredient amounts changed.")
            else:
                show_json(result)
            log.append({"kind": "tool", "name": name, "args": arguments, "result": result})
            replies.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
            if result.get("ok") and name in ("pour_sample", "finish_and_pour", "cancel_session"):
                stop = True
                sampled = name == "pour_sample"
        if not replies:
            return
        history.extend(replies)
        if stop:
            if sampled:
                sample_followup(api_key, model, history, log)
            return
    raise RuntimeError("Model exceeded six tool rounds in one turn")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "google/gemini-3.8-flash"))
    parser.add_argument("--mode", choices=["combined", "audio-only", "features-only"], default="combined")
    parser.add_argument("--analyze", type=Path, help="Analyze one m4a locally without OpenRouter")
    args = parser.parse_args()
    if args.analyze:
        print(json.dumps(analyze_audio(args.analyze.resolve()), indent=2))
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY first")
    session, history, events = Session(), [], []
    output = ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-%d.json" % time.time_ns())
    output.parent.mkdir(exist_ok=True)
    section("MIXMIND | VOCAL LM EXPERIMENT | SIMULATION ONLY")
    print("Model: %s | Mode: %s" % (args.model, args.mode))
    print("Audio folder:", ROOT)
    print("Full JSON log:", output)
    print("Enter an .m4a filename, /pour, or /cancel.")
    try:
        while session.status == "active":
            command = input("Audio or action> ").strip().strip('"')
            if not command:
                continue
            if command in ("/pour", "/cancel"):
                result = session.confirm_final() if command == "/pour" else session.call("cancel_session", {})
                events.append({"kind": "operator", "command": command, "result": result})
                section("OPERATOR INPUT | " + command)
                show_json({k: v for k, v in result.items() if k != "state"})
                show_session(session)
                continue
            path = Path(command)
            path = path if path.is_absolute() else ROOT / path
            if path.suffix.lower() != ".m4a" or not path.is_file():
                print("Provide an existing .m4a file")
                continue
            if path.stat().st_size > 15 * 1024 * 1024:
                print("Audio must be under 15 MiB for this inline-input experiment")
                continue
            print("\nAnalyzing %s..." % path.name, flush=True)
            evidence = analyze_audio(path)
            session.begin_turn(evidence["decipher_proposal"])
            events.append({"kind": "input", "file": path.name, "evidence": evidence})
            run_turn(os.environ["OPENROUTER_API_KEY"], args.model, history, session, path, evidence, args.mode, events)
            show_session(session)
            output.write_text(json.dumps({"model": args.model, "mode": args.mode,
                                          "events": events, "state": session.snapshot()}, indent=2), encoding="utf-8")
    except (KeyboardInterrupt, EOFError):
        session.call("cancel_session", {})
        print("\nSession cancelled")
    finally:
        output.write_text(json.dumps({"model": args.model, "mode": args.mode,
                                      "events": events, "state": session.snapshot()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
