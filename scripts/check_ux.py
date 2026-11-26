"""UX ratchet and shared-piece gates for docs/UX_PRINCIPLES.md.

    python scripts/check_ux.py              # the ratchet (same as "ratchet")
    python scripts/check_ux.py --lower      # write lower counts into the baseline
    python scripts/check_ux.py <mode>       # one gate, prints its own OK token

The ratchet counts, per template, the patterns the principles retire (see RULES).
scripts/ux_baseline.json holds the counts the repository started from. A count
above its baseline fails: that is new debt, and every chat's gate run catches
it. A count below its baseline also fails until the baseline is lowered with
--lower in the same commit, so paid-off debt cannot quietly come back.

Modes, each a gate in run_all_gates.py:
    ratchet     no template is above or below its baseline        UX_RATCHET_OK
    control     the scanner finds every rule in a bad fixture and nothing in a clean one   UX_CONTROL_OK
    principles  the doc names all 20 laws and the shared pieces    UX_PRINCIPLES_OK
    claude-md   CLAUDE.md points chats at the doc and this check   UX_CLAUDE_MD_OK
    shell       base.html loads window.Insights once, keeps the live region, Bootstrap once, and every page renders it   UX_SHELL_OK
    behavior    the shared pieces behave under node, and broken copies of them are caught   UX_BEHAVIOR_OK
    targets     the CSS gives small and icon-only controls a 40px target, 44px by touch   UX_TARGETS_OK
    registered  every built mode above is registered in run_all_gates.py   UX_REGISTERED_OK
Later rollout phases (zero, disabled-states, flows, undo, harness) are listed in
docs/ux-rollout/GATES.md and fail here as not built yet.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATES = os.path.join(ROOT, "templates")
BASELINE = os.path.join(HERE, "ux_baseline.json")
FIXTURES = os.path.join(HERE, "ux_fixtures")
UI_JS = os.path.join(ROOT, "static", "insights-ui.js")
BASE = os.path.join(TEMPLATES, "base.html")
DOC = os.path.join(ROOT, "docs", "UX_PRINCIPLES.md")


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# ------------------------------------------------------------------ the scanner

BUTTON = re.compile(r"<button\b([^>]*)>(.*?)</button>", re.S | re.I)
HAS_TEXT = re.compile(r"\w", re.U)


def _icon_only_buttons(text):
    count = 0
    for match in BUTTON.finditer(text):
        attrs, inner = match.group(1), match.group(2)
        if re.search(r"\baria-label(ledby)?\s*=|\btitle\s*=", attrs, re.I):
            continue
        if "{{" in inner or "${" in inner or "{%" in inner:
            continue  # text supplied by Jinja or a template literal
        visible = re.sub(r"<[^>]+>", "", inner)
        visible = re.sub(r"&[a-z]+;|&#\d+;", "", visible)
        if not HAS_TEXT.search(visible):
            count += 1
    return count


def _inline_hex(text):
    total = 0
    for match in re.finditer(r"""\bstyle\s*=\s*("[^"]*"|'[^']*')""", text, re.I):
        total += len(re.findall(r"#[0-9a-fA-F]{3,8}\b", match.group(1)))
    return total


# name -> (counter, what to do instead). Order is the report order.
RULES = {
    "alert": (lambda t: len(re.findall(r"(?<![\w$.])alert\s*\(|\bwindow\.alert\s*\(", t)),
              "show it with Insights.toast(message, {type: 'error'}) or Insights.fieldError next to the control"),
    "confirm": (lambda t: len(re.findall(r"(?<![\w$.])confirm\s*\(|\bwindow\.confirm\s*\(", t)),
                "use Insights.deferred (Undo) inside Insights, or await Insights.confirm({...}) for what leaves it"),
    "local_toast": (lambda t: len(re.findall(r"function\s+showToast\s*\(|\bshowToast\s*=\s*(?:function|\()", t)),
                    "delete the page's copy; showToast and Insights.toast come from base.html"),
    "button_no_type": (lambda t: len(re.findall(r"<button\b(?![^>]*\btype\s*=)[^>]*>", t, re.I)),
                       'give the <button> type="button" (or type="submit" if it submits)'),
    "icon_button_no_label": (_icon_only_buttons,
                             "give the icon-only button an aria-label that says what it does"),
    "img_no_alt": (lambda t: len(re.findall(r"<img\b(?![^>]*\balt\s*=)[^>]*>", t, re.I)),
                   'give the <img> an alt (alt="" when it is decoration)'),
    "inline_hex_color": (_inline_hex,
                         "use a token from base.html (var(--ok), var(--muted), ...) in a class, not a hex in style="),
}


def scan_text(text):
    return {name: counter(text) for name, (counter, _) in RULES.items()}


def template_files():
    return sorted(glob.glob(os.path.join(TEMPLATES, "**", "*.html"), recursive=True))


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def scan_templates():
    counts = {}
    for path in template_files():
        found = {k: v for k, v in scan_text(read(path)).items() if v}
        if found:
            counts[rel(path)] = found
    return counts


def load_baseline():
    if not os.path.exists(BASELINE):
        fail("scripts/ux_baseline.json is missing; run python scripts/check_ux.py --init once")
    data = json.loads(read(BASELINE))
    return data.get("files", {})


def write_baseline(files):
    data = {
        "about": "Per-template counts of retired UI patterns. Only ever goes down. "
                 "See scripts/check_ux.py and docs/UX_PRINCIPLES.md.",
        "rules": list(RULES),
        "files": {path: files[path] for path in sorted(files) if files[path]},
    }
    with open(BASELINE, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")


def compare(current, baseline):
    above, below = [], []
    for path in sorted(set(current) | set(baseline)):
        for rule in RULES:
            now = current.get(path, {}).get(rule, 0)
            was = baseline.get(path, {}).get(rule, 0)
            if now > was:
                above.append((path, rule, now, was))
            elif now < was:
                below.append((path, rule, now, was))
    return above, below


# ------------------------------------------------------------------------ modes

def mode_ratchet():
    current, baseline = scan_templates(), load_baseline()
    above, below = compare(current, baseline)
    for path, rule, now, was in above:
        print(f"  NEW DEBT {path}: {rule} {now} > {was}. Fix: {RULES[rule][1]}")
    for path, rule, now, was in below:
        print(f"  PAID OFF {path}: {rule} {now} < {was}. Lock it in: python scripts/check_ux.py --lower")
    if above or below:
        fail(f"{len(above)} count(s) above and {len(below)} below the baseline "
             f"(docs/UX_PRINCIPLES.md says what to use instead)")
    total = sum(sum(v.values()) for v in current.values())
    print(f"{len(template_files())} templates at baseline; {total} retired patterns left to migrate")
    print("UX_RATCHET_OK")


def mode_lower():
    current, baseline = scan_templates(), load_baseline()
    above, below = compare(current, baseline)
    if above:
        for path, rule, now, was in above:
            print(f"  NEW DEBT {path}: {rule} {now} > {was}. Fix: {RULES[rule][1]}")
        fail("--lower only lowers; remove the new debt first")
    lowered = {}
    for path in set(baseline) | set(current):
        row = {rule: min(baseline.get(path, {}).get(rule, 0), current.get(path, {}).get(rule, 0))
               for rule in RULES}
        lowered[path] = {k: v for k, v in row.items() if v}
    write_baseline(lowered)
    for path, rule, now, was in below:
        print(f"  lowered {path}: {rule} {was} -> {now}")
    print(f"baseline lowered in {len(below)} place(s)")


def mode_init():
    if os.path.exists(BASELINE):
        fail("the baseline already exists; it only goes down (use --lower)")
    write_baseline(scan_templates())
    print("baseline written")


def mode_control():
    bad, clean = (read(os.path.join(FIXTURES, name)) for name in ("bad.html", "clean.html"))
    bad_counts, clean_counts = scan_text(bad), scan_text(clean)
    missed = [rule for rule, n in bad_counts.items() if n == 0]
    if missed:
        fail(f"the scanner found nothing for {', '.join(missed)} in the bad fixture")
    flagged = {rule: n for rule, n in clean_counts.items() if n}
    if flagged:
        fail(f"the scanner flagged the clean fixture: {flagged}")
    # And the ratchet itself must fail on new debt and on unlowered progress.
    above, below = compare({"t.html": {"alert": 2}}, {"t.html": {"alert": 1}})
    if not above or below:
        fail("compare() did not report a count above its baseline")
    above, below = compare({}, {"t.html": {"alert": 1}})
    if above or not below:
        fail("compare() did not report a count below its baseline")
    print(f"bad fixture trips all {len(RULES)} rules; clean fixture trips none")
    print("UX_CONTROL_OK")


LAWS = ["Hick", "Fitts", "Jakob", "Proximity", "Miller", "Doherty", "Von Restorff",
        "Fitts", "Serial Position", "Peak-End", "Zeigarnik", "Prägnanz", "Sensible Defaults",
        "Prevent Errors", "Errors Recoverable", "Similarity", "Uniform Connectedness",
        "Parkinson", "Tesler", "Goal-Gradient"]
PIECES = ["Insights.toast", "Insights.confirm", "Insights.busy", "Insights.explain",
          "Insights.fieldError", "Insights.deferred"]


def mode_principles():
    text = read(DOC)
    source = text.split("## Source text", 1)
    if len(source) != 2:
        fail("the doc lost its verbatim source section")
    headings = re.findall(r"^## (\d+)\. (.+)$", source[1], re.M)
    if [int(n) for n, _ in headings] != list(range(1, 21)):
        fail(f"the source text does not have laws 1 to 20 in order ({len(headings)} found)")
    for number, (law, (_, heading)) in enumerate(zip(LAWS, headings), 1):
        if law.lower() not in heading.lower():
            fail(f"law {number} should be {law!r}, the doc says {heading!r}")
    checklist = re.findall(r"^(\d+)\. \*\*", source[0], re.M)
    if [int(n) for n in checklist] != list(range(1, 21)):
        fail("the Insights checklist does not have 20 numbered items")
    for piece in PIECES:
        if piece not in source[0]:
            fail(f"the doc does not tell chats about {piece}")
    if "PLANNED, not built" in source[0]:
        fail("the doc still says the shared pieces are not built")
    if "scripts/check_ux.py" not in source[0]:
        fail("the doc does not name the ratchet")
    print("20 laws in order, a checklist item for each, and every shared piece documented")
    print("UX_PRINCIPLES_OK")


def mode_claude_md():
    text = read(os.path.join(ROOT, "CLAUDE.md"))
    section = text.split("## UI work follows the UX principles", 1)
    if len(section) != 2:
        fail("CLAUDE.md lost its 'UI work follows the UX principles' section")
    body = section[1].split("\n## ", 1)[0]
    for needle in ("docs/UX_PRINCIPLES.md", "alert(", "toast", "window.Insights", "scripts/check_ux.py"):
        if needle not in body:
            fail(f"the UI section of CLAUDE.md does not mention {needle}")
    print("CLAUDE.md sends UI work to the doc, the shared pieces and the ratchet")
    print("UX_CLAUDE_MD_OK")


BOOTSTRAP_JS = "bootstrap.bundle.min.js"


def mode_shell():
    base = read(BASE)
    head, _, body = base.partition("</head>")
    ui_tag = re.compile(r"<script[^>]*\bsrc=[^>]*insights-ui\.js")
    boot_tag = re.compile(r"<script[^>]*\bsrc=[^>]*" + re.escape(BOOTSTRAP_JS))
    if len(ui_tag.findall(base)) != 1 or not ui_tag.search(head):
        fail("base.html must load insights-ui.js exactly once, in <head>")
    if len(boot_tag.findall(base)) != 1:
        fail("base.html must load Bootstrap's JS exactly once")
    if boot_tag.search(head).start() > ui_tag.search(head).start():
        fail("insights-ui.js must load after Bootstrap's JS")
    region = re.search(r'<div[^>]*id="insightsToasts"[^>]*>', body)
    if not region or 'aria-live="polite"' not in region.group(0):
        fail("base.html lost the #insightsToasts live region")
    for path in template_files():
        if path == BASE:
            continue
        text = read(path)
        if ui_tag.search(text):
            fail(f"{rel(path)} loads insights-ui.js again; base.html already does")
        if boot_tag.search(text) and "{% extends" in text:
            fail(f"{rel(path)} extends base.html and loads Bootstrap's JS a second time")
    js = read(UI_JS)
    for name in ("toast", "confirm", "busy", "explain", "fieldError", "deferred"):
        if not re.search(rf"\b{name}: {name}\b", js):
            fail(f"insights-ui.js does not export Insights.{name}")
    if "global.showToast = function" not in js:
        fail("insights-ui.js lost the showToast(message, type) wrapper")

    # Render real pages: the tag is there and the file is served.
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    cwd = os.getcwd()
    try:
        from _gate_common import isolated_app
        directory, _database, web, client = isolated_app()
        try:
            pages = ["/compose", "/schedule", "/accounts", "/inbox", "/outreach", "/sources"]
            for page in pages:
                response = client.get(page)
                html = response.get_data(as_text=True)
                if response.status_code != 200:
                    fail(f"{page} did not render ({response.status_code})")
                if len(ui_tag.findall(html)) != 1 or len(boot_tag.findall(html)) != 1:
                    fail(f"{page} does not load insights-ui.js and Bootstrap's JS exactly once each")
                if 'id="insightsToasts"' not in html:
                    fail(f"{page} has no toast live region")
            static = client.get("/static/insights-ui.js")
            served = static.get_data(as_text=True)
            static.close()
            if static.status_code != 200 or "window.Insights" not in served and "global.Insights" not in served:
                fail("/static/insights-ui.js is not served")
        finally:
            os.chdir(cwd)
            shutil.rmtree(directory, ignore_errors=True)
    except SystemExit:
        raise
    print(f"base.html loads Insights once after Bootstrap; {len(pages)} pages render it with one live region")
    print("UX_SHELL_OK")


def _node():
    node = shutil.which("node")
    if not node:
        fail("node is required for this gate and was not found on PATH")
    return node


def _run_behavior(js_path):
    result = subprocess.run([_node(), os.path.join(HERE, "ux_behavior.js"), js_path],
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        fail(f"the behavior harness crashed:\n{result.stderr[-2000:]}")
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        fail(f"the behavior harness printed no result:\n{result.stdout[-1000:]}{result.stderr[-1000:]}")


# Each mutant breaks one promise; the harness must notice every one of them.
MUTANTS = [
    ("errors vanish after 5 s", "error: 0 }", "error: 5000 }"),
    ("busy leaves the button enabled", "btn.disabled = true;", "btn.disabled = false;"),
    ("busy runs a second click", "if (running) return running;", ""),
    ("deferred sends at once", "hide();\n            pending.push(entry);", "hide();\n            pending.push(entry); commit(false);"),
    ("a risky confirm starts on its action", "(opts.danger ? d.cancel : d.ok).focus();", "d.ok.focus();"),
    ("explain passes raw errors", "if (err.message && !TECHNICAL.test(err.message)", "if (err.message"),
    ("fieldError forgets aria-invalid", "input.setAttribute('aria-invalid', 'true');", ""),
    ("toast renders HTML", "var body = make('div', 'insights-toast-body', message);",
     "var body = make('div', 'insights-toast-body'); body.innerHTML = message;"),
]


def mode_behavior():
    results = _run_behavior(UI_JS)
    broken = {name: why for name, why in results.items() if why is not True}
    if broken:
        for name, why in broken.items():
            print(f"  {name}: {why}")
        fail(f"{len(broken)} of {len(results)} behavior checks failed")
    source = read(UI_JS)
    scratch = tempfile.mkdtemp(prefix="ux_mutant_")
    try:
        for label, old, new in MUTANTS:
            if source.count(old) != 1:
                fail(f"mutant {label!r} no longer matches insights-ui.js; update MUTANTS")
            path = os.path.join(scratch, "insights-ui.js")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source.replace(old, new))
            if all(v is True for v in _run_behavior(path).values()):
                fail(f"the harness did not catch a broken copy: {label}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"{len(results)} behavior checks pass; {len(MUTANTS)} broken copies are each caught")
    print("UX_BEHAVIOR_OK")


def mode_targets():
    base = read(BASE)
    style = base.split("<style>", 1)[1].split("</style>", 1)[0]
    if not re.search(r"--target-min:\s*40px", style) or not re.search(r"--target-touch:\s*44px", style):
        fail("base.html lost the --target-min (40px) / --target-touch (44px) tokens")
    icon = re.search(r"\.btn-icon\s*\{([^}]*)\}", style)
    if not icon or "min-width: var(--target-min)" not in icon.group(1) or "min-height: var(--target-min)" not in icon.group(1):
        fail(".btn-icon does not hold the 40px floor")
    if not re.search(r"\.btn-close::after\s*\{[^}]*inset:\s*calc\(\(var\(--target-min\)", style):
        fail(".btn-close lost its 40px hit area")
    coarse = re.search(r"@media \(pointer: coarse\)\s*\{(.*?)\n        \}", style, re.S)
    if not coarse:
        fail("there is no (pointer: coarse) block")
    block = coarse.group(1)
    for selector, needle in ((".btn-sm", "min-height: var(--target-touch)"),
                             (".btn-icon", "min-width: var(--target-touch)"),
                             (".btn-close::after", "var(--target-touch)")):
        rules = re.finditer(r"([^{}]*)\{([^}]*)\}", block)
        held = any(selector in [s.strip() for s in r.group(1).split(",")] and needle in r.group(2)
                   for r in rules)
        if not held:
            fail(f"touch screens do not give {selector} a 44px target")
    print("40px floor for icon-only and close buttons; 44px for small buttons by touch")
    print("UX_TARGETS_OK")


BUILT = {
    "ratchet": "UX_RATCHET_OK", "control": "UX_CONTROL_OK", "principles": "UX_PRINCIPLES_OK",
    "claude-md": "UX_CLAUDE_MD_OK", "shell": "UX_SHELL_OK", "behavior": "UX_BEHAVIOR_OK",
    "targets": "UX_TARGETS_OK", "registered": "UX_REGISTERED_OK",
}
LATER = {"zero": "Phase 3", "disabled-states": "Phase 3", "flows": "Phase 4",
         "undo": "Phase 3", "harness": "Phase 3"}


def mode_registered():
    sys.path.insert(0, HERE)
    import run_all_gates
    have = {}
    for entry in run_all_gates.ALL_GATES:
        filename, token, _claim, args = run_all_gates._unpack(entry)
        if filename == "check_ux.py":
            have[tuple(args)[0] if args else "ratchet"] = token
    missing = {m: t for m, t in BUILT.items() if have.get(m) != t}
    if missing:
        fail(f"run_all_gates.py does not register check_ux.py {sorted(missing)} with their tokens")
    print(f"all {len(BUILT)} check_ux.py modes are registered under their tokens")
    print("UX_REGISTERED_OK")


MODES = {
    "ratchet": mode_ratchet, "control": mode_control, "principles": mode_principles,
    "claude-md": mode_claude_md, "shell": mode_shell, "behavior": mode_behavior,
    "targets": mode_targets, "registered": mode_registered,
    "--lower": mode_lower, "--init": mode_init,
}


def main(argv):
    mode = argv[1] if len(argv) > 1 else "ratchet"
    if mode in LATER:
        fail(f"mode {mode!r} is not built yet (rollout {LATER[mode]}, docs/ux-rollout/PLAN.md)")
    if mode not in MODES:
        fail(f"unknown mode {mode!r}; one of {', '.join(MODES)}")
    MODES[mode]()


if __name__ == "__main__":
    main(sys.argv)
