# Vocal LM Experiment

Turn-based OpenRouter audio experiment for MixMind (Gemini by default). No microphone capture, frontend,
or hardware connection: you supply recorded `.m4a` files and inspect terminal
output and JSON logs. No pump module is imported. Gemini receives the audio,
existing acoustic analysis, baseline recipe, and accumulated session context.

## Setup (Windows PowerShell)

Run from `D:\HackMIT26\vocal-LM-exp`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

FFmpeg is bundled by `imageio-ffmpeg`; no separate install or PATH change is
required. An existing FFmpeg on PATH is preferred when available. FFmpeg decodes
M4A to a temporary mono 16 kHz WAV for the existing analyzer.
The original M4A is sent through OpenRouter as base64 `input_audio` with format `m4a`; clips must be <=25 seconds
and <=15 MiB. The temporary WAV is deleted after analysis.

The sibling `../voice_decipher_2` folder must be present, including its VAD model.
No Whisper installation is necessary: Gemini hears the words directly. The
baseline uses `local_bartender.recipe(features)` without VADER sentiment.

Set your key in the terminal (not in a tracked file):

```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-key"
$env:OPENROUTER_MODEL = "google/gemini-2.5-flash"
.\.venv\Scripts\python.exe harness.py
```

The model is configurable with `--model` or `OPENROUTER_MODEL`. Choose an OpenRouter
model/provider supporting both M4A audio input and tool calling. Requests use
`https://openrouter.ai/api/v1/chat/completions`, with provider routing requiring
support for the supplied parameters. The integration uses Python's standard
library; no Google SDK or OpenAI SDK is required. The default is a starting candidate,
not a verified guarantee of availability for your account. The first live run
is also the audio + function-calling capability test. API calls require internet
and may incur charges. Audio is sent through OpenRouter to the selected provider;
local logs contain model outputs,
measurements, and inferred/transcribed speech. Keys and audio bytes are not logged.

## Record These Files

Save the following files directly in this folder, using the exact filenames.
They are recording instructions, not existing or generated audio assets.

Use iPhone Voice Memos (or any recorder exporting real AAC/M4A). Tap Record,
wait about half a second, speak the exact transcript, wait half a second, stop,
then Share/Save to Files and transfer here. Renaming a WAV to `.m4a` does not
convert it. Use the same room, mic, distance (about 15-20 cm), and input settings.
Do not add music or artificial voice effects. Avoid clipping. Check every clip
plays correctly and is under 25 seconds; most should be 3-12 seconds.

| Filename | Exact transcript | Delivery |
|---|---|---|
| `flat.m4a` | "I'm fine, really. It's been a long day, and I finally have a minute to take a break. Make me something that fits." | Tired, quiet but clearly audible, narrow pitch range. Pause briefly after "really" and "day". Do not whisper. |
| `bright.m4a` | "I'm fine, really. It's been a long day, and I finally have a minute to take a break. Make me something that fits." | Upbeat, lively pitch changes, naturally quicker and fluid. Same words and mic distance as `flat.m4a`; do not shout. |
| `tart.m4a` | "Actually, make it less sweet and a little more tart. Keep it about the same size." | Clear, friendly, decisive. This requests a genuine recipe revision. |
| `gentle.m4a` | "That's a little too sharp. Make it gentler, but don't make it very sweet." | Thoughtful, mildly dissatisfied, not angry. Pretend this is feedback on the preceding sample. |
| `exclude.m4a` | "I don't want grapefruit in this drink. Please leave it out of every version." | Neutral and firm; emphasize "don't" and "every version". |
| `correct.m4a` | "I'm not tired, just speaking quietly. I'd like something bright and lively." | Quiet but confident. Correct the interpretation without raising volume. |
| `vague.m4a` | "Could you make it a bit more interesting?" | Curious and uncertain. Deliberately omit what "interesting" means. |
| `accept.m4a` | "Yes, that version sounds good. Finish the conversation and pour the final drink." | Unambiguous, calm acceptance. |
| `cancel.m4a` | "Cancel the drink. End the conversation without pouring the final drink." | Firm, clearly articulated; no joking tone. |
| `fourth.m4a` | "One more change, please. Add more apple juice and give me another sample." | Friendly and persistent. Use only after three proposals to test the limit. |

Record both paired clips yourself in one sitting. Acoustic differences are
observations, not validated emotional labels. Tiny differences between recipes
are possible; do not expect a particular drink name or identical model wording.

## Run the Experiments

First verify local acoustic processing without an API key:

```powershell
.\.venv\Scripts\python.exe harness.py --analyze flat.m4a
```

Then run `harness.py` and enter filenames at `Audio or action>` one at a time.
Wait for the tool results and current recipe before entering the next file.
Paths are relative to this experiment folder, regardless of terminal directory.

1. **Negotiation:** `flat.m4a`, `tart.m4a`, `gentle.m4a`, `accept.m4a`, then `/pour`.
   Expect initial proposal plus two revisions, each with one simulated sample.
   Acceptance requests finalization; `/pour` supplies explicit local confirmation.
2. **Limit:** repeat the first three clips above, then `fourth.m4a`.
   No fourth revision or sample may execute. End with `/pour` or `/cancel`.
3. **Constraint memory:** new session: `exclude.m4a`, `tart.m4a`, `gentle.m4a`.
   Channel 3 must remain absent. The current experiment asks the model to preserve
   semantic constraints; the validator does NOT independently understand exclusions.
   If it reintroduces grapefruit, record that as a failed experiment.
4. **Correction:** new session: `flat.m4a`, `correct.m4a`.
   Expect acknowledgement of the correction without insisting the user is tired.
5. **Ambiguity:** new session: `flat.m4a`, `vague.m4a`.
   Prefer clarification rather than an arbitrary recipe change.
6. **Cancellation:** new session: `flat.m4a`, `cancel.m4a`.
   The sample remains in the ledger, but no final pour should appear.
7. **Voice comparison:** separate fresh sessions with `flat.m4a` and `bright.m4a`.
   Compare acoustic measurements, explanations, and ingredient amounts.

For input ablations, repeat a FIRST-TURN comparison in fresh sessions:

```powershell
.\.venv\Scripts\python.exe harness.py --mode combined
.\.venv\Scripts\python.exe harness.py --mode audio-only
.\.venv\Scripts\python.exe harness.py --mode features-only
```

`audio-only` omits raw measurements/axes but retains the decipher baseline recipe
needed by the editing tools. Thus it is an ablation of explicit measurement
context, NOT an independent audio-only scientific control. `features-only`
omits audio and transcript entirely, so it cannot test verbal negotiation.
All modes still compute the local baseline. Repeat runs to distinguish useful
patterns from model variability; the model is not deterministic.

## Tools and Rules

- `revise_drink`: version-checked ingredient edits using `set_amounts`. Amounts are absolute ml in
  the FINAL serving, not deltas. Zero removes an ingredient. Unchanged channels
  retain their amounts. An empty edit commits the initial baseline.
  The result contains the full validated recipe in `state.recipe` plus a
  server-computed `amount_diff`: channel, ingredient, before/after mL, and signed
  delta mL for each changed ingredient. `diff_base` distinguishes the internal
  first-proposal baseline from a previous sampled recipe. CLI output and Markdown
  reports show these verified changes. The report reader also supports older
  logs using the former `changes` argument name.
- `pour_sample`: one simulated 15 ml sample per version in a separate tasting cup.
  Proportions are scaled from the full recipe. Duplicate calls do not dispense again.
- `finish_and_pour`: requests finalization; `/pour` actually records the simulated
  final pour. The operator can use `/pour` at any prompt once a proposal exists.
- `cancel_session` or `/cancel`: closes the session without a final serving.

Three proposals maximum, including the initial one; at most one revision per
audio turn. After sampling, one additional model call (tools disabled) produces
a conversational description and feedback question. The prompt requests one or
two short sentences, under 50 words; this is a model instruction, not a sentence
counter. The message is printed under `BARTENDER MESSAGE | YOUR TURN`, saved in
history and the run log, and appears in reports as another response in that step.
This adds one API call per sample. A failed/empty follow-up uses a brief fallback
without repeating the pour. Finalization requests and cancellation still stop immediately.

The bartender speaks as the complete MixMind product. The first proposal describes
the whole drink, not modifications to the hidden decipher baseline. Later replies
may describe changes relative to the previous sampled recipe and the guest's
feedback. This applies to recipe explanations as well as the closing message.
To evaluate this, start a fresh session with `flat.m4a`: expect an introduction
such as "Your drink pairs citrus with ginger ale...", only if those ingredients
are present, rather than "I added ginger ale...". Then use `tart.m4a`: comparative
language should refer to the drink proposed in the preceding turn.
Each subsequent utterance is evidence, not a replacement for the negotiated recipe.
Final recipes require 2-6 distinct channels, 10-80 ml each, <=220 ml total.

The ingredient catalog is explicitly hypothetical and mirrors the current kiosk
labels. Actual bottle sweetness must be checked before evaluating taste claims.
The simulated samples deliberately bypass the hardware's 10 ml PER INGREDIENT
minimum. They are not executable hardware recipes. Minimum repeatable doses,
cup handling, and taste fidelity require a separate physical experiment.

## Inspect and Verify

### Saved-run report viewer

```powershell
.\.venv\Scripts\python.exe report.py
```

Pick one numbered run from the newest-first list. The CLI writes a concise
Markdown report with chronological steps: simulated user script and delivery,
model responses, provider reasoning, tool requests/results, and operator actions.
Recipes become short ingredient lists and usage becomes a single token/cost line.
Structured details are linked to exact source JSON lines using absolute local
`path:line` Markdown links (supported by the workspace viewer; ordinary Markdown
viewers may not support jumping to local line numbers).
Readable reasoning is displayed once; encrypted
reasoning is displayed as `encrypted`, never as ciphertext. No model/API call
is made. Scripts come from this README's recording table and are explicitly
not verified transcripts. Unknown filenames are marked as having no script.

Each selection writes one UTF-8 file, `reports/<run>.md`, and prints its path.
The CLI no longer dumps the report into the terminal. You may also pass a
specific JSON log and optional Markdown destination:

```powershell
.\.venv\Scripts\python.exe report.py runs\YOUR-RUN.json
.\.venv\Scripts\python.exe report.py runs\YOUR-RUN.json --output reports\demo.md
```

### Live output

The CLI separates each turn into **INPUTS**, **RAW RESPONSE**, **BARTENDER
MESSAGE**, **TOOL CALL**, and **SESSION RESULT** sections. Inputs show the audio
path, model, mode, prior-message count, and structured context sent to the model
(audio bytes are omitted). Every API response is printed as indented JSON,
including metadata and tool-call arguments. Tool executions show parsed arguments,
success/errors, and recipe tables with ingredient names and before/after changes.
The final summary shows the current recipe, remaining updates, and pour counts.
The raw response section is the actual API response, not a generated summary.

Each session writes `runs/<timestamp>-<unique-id>.json`: input features, full model
responses, response latency per call, tool arguments/results, final recipe,
and dispensing ledger. Audio is not embedded in this log. Recipe revisions and
sample events are visible in terminal output; there is no frontend integration.
An API failure ends the run and preserves the log; it does not retry physical
actions or resume conversation state. Start a fresh run to retry.

Use this pass checklist: meaningful recipe change for explicit feedback;
preferences retained; no fourth proposal; exactly one sample per version;
no final event before `/pour`; no final event after cancellation. Judge ingredient
meaning manually against the catalog rather than comparing generated prose.

Offline tool-state tests need only standard Python:

```powershell
python -B -m unittest -v test_harness
```

These tests exercise atomic invalid edits, version checks, limits, continuity,
duplicate dispensing, explicit final confirmation, and cancellation. They do not
verify Gemini, FFmpeg, acoustic accuracy, or real pumps.
