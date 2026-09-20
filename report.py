"""Select a saved experiment and render a readable, chronological cascade."""
import argparse
import json
from pathlib import Path
import bisect
import re
from urllib.parse import quote

from harness import CATALOG

ROOT = Path(__file__).resolve().parent


def recording_scripts():
    scripts = {}
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| `"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == 3 and cells[0].endswith(".m4a`"):
                scripts[cells[0].strip("`")] = cells[1:]
    return scripts


def clean(value):
    # Saved model output is untrusted terminal text; remove terminal controls.
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
    return "".join(c for c in value if c in "\n\t" or (ord(c) >= 32 and ord(c) != 127))


def reasoning(message):
    output, seen = [], set()
    for detail in message.get("reasoning_details") or []:
        if not isinstance(detail, dict):
            continue
        if "encrypted" in detail.get("type", ""):
            text = "encrypted"
        else:
            text = detail.get("text") or detail.get("summary")
        if text and str(text).strip() not in seen:
            output.append(str(text).strip())
            seen.add(str(text).strip())
    for key in ("reasoning", "reasoning_content"):
        text = message.get(key)
        if text and str(text).strip() not in seen:
            output.append(str(text).strip())
            seen.add(str(text).strip())
    return output or ["Not supplied by provider."]


def source_lines(text):
    """Index JSON value paths against their actual source offsets, including compact JSON."""
    tokens = list(re.finditer(r'"(?:\\.|[^"\\])*"|[{}\[\]:,]|[^\s{}\[\]:,]+', text))
    newlines = [m.start() for m in re.finditer('\n', text)]
    positions, cursor = {}, 0

    def walk(path):
        nonlocal cursor
        token = tokens[cursor]
        positions[path] = bisect.bisect_left(newlines, token.start()) + 1
        cursor += 1
        if token.group() == '{':
            while tokens[cursor].group() != '}':
                key = json.loads(tokens[cursor].group())
                cursor += 2  # key and colon
                walk(path + (key,))
                if tokens[cursor].group() == ',':
                    cursor += 1
            cursor += 1
        elif token.group() == '[':
            index = 0
            while tokens[cursor].group() != ']':
                walk(path + (index,))
                index += 1
                if tokens[cursor].group() == ',':
                    cursor += 1
            cursor += 1
    walk(())
    return positions


def prose(value):
    text = re.sub(r'\s+', ' ', clean(value)).strip()
    for char in ('\\', '`', '*', '_', '[', ']', '<', '>'):
        text = text.replace(char, '\\' + char)
    return text


def pours(items):
    return '; '.join('CH %s (%s): %s mL' %
                     (p.get('channel', '?'), CATALOG.get(p.get('channel'), {}).get('name', 'Unknown'),
                      p.get('ml', '?')) for p in items) or 'No ingredients recorded'


def outcome(result):
    if not result.get('ok'):
        return 'Failed: ' + prose(result.get('error', 'unspecified error'))
    prefix = 'Already executed; no extra pour. ' if result.get('duplicate') else ''
    if result.get('state'):
        state = result['state']
        recipe = state.get('recipe') or {}
        return prefix + 'Recipe v%s: %s. %s.' % (state.get('version', '?'),
                    prose(recipe.get('name', 'Unnamed')), pours(recipe.get('pours', [])))
    event = result.get('event', {})
    if event:
        return prefix + 'Simulated %s, recipe v%s: %s.' % (event.get('kind', 'pour'),
                event.get('version', '?'), pours(event.get('pours', event.get('recipe', {}).get('pours', []))))
    return prefix + prose(result.get('message') or result.get('status') or 'Completed.')


def render(data, filename, scripts=None, source_text=None, source_path=None):
    scripts = scripts or {}
    positions = source_lines(source_text) if source_text is not None else {}
    target = quote(Path(source_path or filename).resolve().as_posix(), safe='/:')

    def link(label, path):
        if path not in positions:
            return label
        return '[%s](<%s:%d>)' % (label, target, positions[path])

    lines = ['# MixMind experiment', '', '**%s** | %s | %s.' %
             (prose(filename), prose(data.get('model', 'unknown')), prose(data.get('mode', 'unknown'))),
             '', 'All pours are simulated. User scripts below are planned inputs, not verified transcripts.', '']
    step, round_number = 0, 0
    for index, event in enumerate(data.get("events", [])):
        path = ('events', index)
        kind = event.get("kind")
        if kind in ("input", "operator"):
            step += 1
            round_number = 0
            lines.extend(['## Step %d: %s' % (step, prose(event.get('file', event.get('command', 'Input')))), ''])
        if kind == "input":
            name = event.get("file", "unknown")
            script = scripts.get(Path(name).name)
            if script:
                lines.append('**User script:** %s **Delivery:** %s' % (prose(script[0]), prose(script[1])))
            else:
                lines.append('User script unavailable; no transcript stored in this log.')
            evidence = event.get("evidence", {})
            recipe = evidence.get('decipher_proposal', {})
            lines.extend(['', '**Decipher:** %s. %s. %s' % (prose(recipe.get('name', 'No proposal')),
                          pours(recipe.get('pours', [])), link('Input measurements and proposal', path + ('evidence',))), ''])
        elif kind == "model_response":
            round_number += 1
            response = event.get("response", {})
            lines.extend(['### Response %d (%s s)' % (round_number, event.get('latency_s', '?')), ''])
            choices = response.get("choices") or []
            if not choices:
                lines.append('No message returned. ' + link('Response details', path + ('response',)))
            for choice_index, choice in enumerate(choices):
                message_path = path + ('response', 'choices', choice_index, 'message')
                message = choice.get("message") or {}
                lines.append('**Assistant:** %s %s' % (prose(message.get('content') or 'Tool-only response.'),
                              link('Source', message_path)))
                if message.get("refusal"):
                    lines.append('**Refusal:** ' + prose(message['refusal']))
                for item in reasoning(message):
                    reasoning_path = message_path + ('reasoning_details',) if message.get('reasoning_details') else message_path + ('reasoning',)
                    lines.append('**Reasoning:** %s %s' % (prose(item), link('Source', reasoning_path)))
                for call_index, call in enumerate(message.get("tool_calls") or []):
                    function = call.get("function", {})
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except ValueError:
                            pass
                    summary = ''
                    if isinstance(arguments, dict):
                        summary = prose(arguments.get('explanation', ''))
                        amounts = arguments.get('set_amounts', arguments.get('changes', []))
                        if amounts:
                            summary += ' Set ' + pours(amounts) + '.'
                        if 'recipe_version' in arguments:
                            summary += ' Recipe v%s.' % arguments['recipe_version']
                    lines.append('**Tool request:** %s. %s %s' % (prose(function.get('name', '?')), summary,
                                 link('Arguments', message_path + ('tool_calls', call_index, 'function', 'arguments'))))
            if response.get("usage"):
                usage = response['usage']
                summary = '%s input + %s output = %s tokens' % (usage.get('prompt_tokens', '?'),
                          usage.get('completion_tokens', '?'), usage.get('total_tokens', '?'))
                if isinstance(usage.get('cost'), (int, float)):
                    summary += '; $%.6f' % usage['cost']
                lines.append('**Usage:** %s. %s' % (summary, link('Token and cost details', path + ('response', 'usage'))))
            lines.append('')
        elif kind == "tool":
            result = event.get("result", {})
            lines.extend(['**Tool result (%s):** %s %s' % (prose(event.get('name', '?')), outcome(result),
                          link('Full tool response', path + ('result',))), ''])
            if 'amount_diff' in result:
                changes = '; '.join('CH %s (%s): %s -> %s mL (%+g mL)' %
                          (d['channel'], prose(d['ingredient']), d['before_ml'], d['after_ml'], d['delta_ml'])
                          for d in result['amount_diff']) or 'No ingredient amounts changed'
                base = 'internal baseline' if result.get('diff_base') == 'internal_baseline' else 'previous sampled recipe'
                lines.extend(['**Verified changes from %s:** %s. %s' %
                              (base, changes, link('Computed diff', path + ('amount_diff',))), ''])
        elif kind == "operator":
            lines.extend([outcome(event.get('result', {})) + ' ' + link('Action details', path), ''])
        else:
            lines.extend([link('Other event', path), ''])
    if not data.get("events"):
        lines.append('No events recorded. This run stopped before an input was logged.')
    state = data.get('state', {})
    recipe = state.get('recipe') or {}
    lines.extend(['## Outcome', '', 'Session %s; recipe v%s. %s' % (prose(state.get('status', 'unknown')),
                  state.get('version', '?'), link('Final session state', ('state',)))])
    if recipe:
        lines.append('%s. %s.' % (prose(recipe.get('name', 'Unnamed')), pours(recipe.get('pours', []))))
    return "\n".join(lines) + "\n"


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        raise ValueError("Not an experiment log (expected an events list)")
    return data


def choose():
    paths = sorted((ROOT / "runs").glob("*.json"), reverse=True)
    if not paths:
        print("No saved runs in %s" % (ROOT / "runs"))
        return None
    print("\nSAVED EXPERIMENTS (newest first)\n")
    for index, path in enumerate(paths, 1):
        try:
            data = load(path)
            clips = [e.get("file", "?") for e in data["events"] if e.get("kind") == "input"]
            label = "%s | %s | %d inputs | %s" % (data.get("model", "?"), data.get("mode", "?"),
                      len(clips), data.get("state", {}).get("status", "unknown"))
            print("%2d. %s\n    %s\n    %s" % (index, path.name, label, ", ".join(clips) or "No audio inputs"))
        except (OSError, ValueError) as exc:
            print("%2d. %s [unreadable: %s]" % (index, path.name, exc))
    while True:
        answer = input("\nSelect log number (q to quit)> ").strip()
        if answer.lower() in ("q", "quit", "exit"):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(paths):
            return paths[int(answer)-1]
        print("Enter a number from 1 to %d, or q." % len(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, help="Skip picker and open this JSON log")
    parser.add_argument("--output", type=Path, help="Markdown destination (default: reports/<run>.md)")
    args = parser.parse_args()
    try:
        path = args.log or choose()
        if path is None:
            return
        data = load(path)
        report = render(data, path.name, recording_scripts(), path.read_text(encoding='utf-8'), path)
        output = args.output or ROOT / "reports" / (path.stem + ".md")
        if output.suffix.lower() != '.md':
            raise ValueError('Report destination must end in .md')
        if output.resolve() == path.resolve():
            raise ValueError("Report destination cannot overwrite its source log")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print("\nReport saved:", output.resolve())
    except (OSError, ValueError) as exc:
        parser.exit(1, "Report error: %s\n" % exc)
    except (EOFError, KeyboardInterrupt):
        print("\nClosed report picker.")


if __name__ == "__main__":
    main()
