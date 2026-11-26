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
    zero        no template has alert(, confirm( or its own showToast   UX_ZERO_DEBT_OK
    harness     harnesses that fake the dialogs also fake Insights into the same log   UX_HARNESS_PARITY_OK
    undo        deletes wait behind Insights.deferred; exits ask with a named confirm   UX_UNDO_OK
    disabled-states  compose, schedule, accounts disable what cannot work and say why   UX_DISABLED_STATES_OK
    flows       getting-started card from live data; queue/publish end on the next step   UX_FLOWS_OK
zero and undo take template paths after the mode to check only those files.
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


ONLY = []   # zero/undo accept template paths after the mode, to check just those


def template_files():
    every = sorted(glob.glob(os.path.join(TEMPLATES, "**", "*.html"), recursive=True))
    if ONLY:
        wanted = {os.path.abspath(os.path.join(ROOT, p)) for p in ONLY}
        unknown = sorted(wanted - {os.path.abspath(p) for p in every})
        if unknown:
            # A path that names no template would otherwise scan nothing and pass.
            fail(f"not a template under templates/: {', '.join(os.path.relpath(u, ROOT) for u in unknown)}")
        return [p for p in every if os.path.abspath(p) in wanted]
    return every


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
          "Insights.fieldError", "Insights.deferred", "Insights.done", "data-confirm", "data-undo"]


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
    for name in ("toast", "confirm", "busy", "explain", "fieldError", "deferred", "done"):
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
    ("a data-confirm form submits without asking", "if (!wantsConfirm && !wantsUndo) return;\n        event.preventDefault();",
     "if (!wantsConfirm && !wantsUndo) return;"),
    ("a data-undo row stays visible", "hide: function () { row.hidden = true; },", "hide: function () {},"),
    ("done hides the next step", "opts.action = { label: next.label, run: next.run || function () { global.location.href = next.href; } };", ""),
    ("a data-confirm button runs its handler on Cancel", "if (event.stopImmediatePropagation) event.stopImmediatePropagation();", ""),
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


# ------------------------------------------------------------- zero debt (G7)

ZERO_RULES = ("alert", "confirm", "local_toast")


def mode_zero():
    left = {}
    for path in template_files():
        found = scan_text(read(path))
        bad = {k: found[k] for k in ZERO_RULES if found[k]}
        if bad:
            left[rel(path)] = bad
    if left:
        for name, bad in sorted(left.items()):
            print(f"  {name}: {bad}")
        fail(f"{sum(sum(b.values()) for b in left.values())} alert(/confirm(/local showToast left in {len(left)} templates")
    print(f"no alert(, confirm( or local showToast in {len(template_files())} templates")
    print("UX_ZERO_DEBT_OK")


# --------------------------------------------------------- harness parity (G15)

STUBS = re.compile(r"\b(?:confirm|alert|showToast)\s*[:=]\s*(?:\(|async\b|function\b|\w+\s*=>)")
CHECK_LINE = re.compile(r"^\s*(?:check|ok|expect|assert)\s*\(|^\s*assert\b", re.M)


def _main_copy(relpath):
    run = subprocess.run(["git", "-C", ROOT, "show", f"origin/main:{relpath}"],
                         capture_output=True, text=True)
    return run.stdout if run.returncode == 0 else None


def mode_harness():
    problems, seen = [], 0
    for path in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        name = os.path.basename(path)
        if name in ("check_ux.py", "_insights_stub.py"):
            continue
        text = read(path)
        if not STUBS.search(text):
            continue
        seen += 1
        if "INSIGHTS_STUB_JS" not in text or not re.search(r"makeInsights\(", text):
            problems.append(f"{name}: stubs alert/confirm/showToast but not Insights (use _insights_stub.INSIGHTS_STUB_JS)")
            continue
        if not re.search(r"Insights\s*[:=]\s*makeInsights\(", text):
            problems.append(f"{name}: builds makeInsights but never puts it in the world as Insights")
        before = _main_copy(f"scripts/{name}")
        if before is not None:
            was, now = len(CHECK_LINE.findall(before)), len(CHECK_LINE.findall(text))
            if now < was:
                problems.append(f"{name}: {now} assertion lines, main has {was}; an assertion was removed")
    if seen == 0:
        fail("no harness stubs alert/confirm/showToast; the scan pattern is broken")
    if problems:
        for p in problems:
            print(f"  {p}")
        fail(f"{len(problems)} harness parity problems")
    print(f"{seen} harnesses stub the dialogs and Insights into one log; none lost an assertion")
    print("UX_HARNESS_PARITY_OK")


# --------------------------------------------------------- undo and confirm (G14)

def _scripts_of(html):
    """(offset, body) for each inline <script> (no src=)."""
    out = []
    for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", html, re.S | re.I):
        if re.search(r"\bsrc\s*=", m.group(1)):
            continue
        out.append((m.start(2), m.group(2)))
    return out


NOT_FUNCTIONS = {"if", "for", "while", "switch", "catch", "with", "return", "typeof"}


def _function_spans(js):
    """[(start, end)] of every function body, found by a small tokenizer that skips
    strings, template literals (with ${} nesting) and comments."""
    spans, stack, i, n = [], [], 0, len(js)
    tmpl = []          # brace depth at which each open template literal's ${ began
    parens = []        # positions of open '('
    closed_paren = {}  # index of ')' -> index of its '('
    while i < n:
        c = js[i]
        if c == "/" and js.startswith("//", i):
            j = js.find("\n", i); i = n if j == -1 else j; continue
        if c == "/" and js.startswith("/*", i):
            j = js.find("*/", i + 2); i = n if j == -1 else j + 2; continue
        if c in "'\"":
            j = i + 1
            while j < n and js[j] != c and js[j] != "\n":
                j += 2 if js[j] == "\\" else 1
            i = j + 1; continue
        if c == "`" or (c == "}" and tmpl and tmpl[-1] == len(stack)):
            if c == "}":
                tmpl.pop()
            j = i + 1
            while j < n:
                if js[j] == "\\":
                    j += 2; continue
                if js[j] == "`":
                    break
                if js.startswith("${", j):
                    tmpl.append(len(stack)); j += 1
                    break
                j += 1
            i = j + 1; continue
        if c == "(":
            parens.append(i)
        elif c == ")" and parens:
            closed_paren[i] = parens.pop()
        elif c == "{":
            k = i - 1
            while k >= 0 and js[k].isspace():
                k -= 1
            is_fn = False
            if k >= 1 and js[k - 1:k + 1] == "=>":
                is_fn = True
            elif k >= 0 and js[k] == ")" and k in closed_paren:
                p = closed_paren[k] - 1
                while p >= 0 and js[p].isspace():
                    p -= 1
                e = p + 1
                while p >= 0 and (js[p].isalnum() or js[p] in "_$"):
                    p -= 1
                word = js[p + 1:e]
                is_fn = word not in NOT_FUNCTIONS
            stack.append((i, is_fn))
        elif c == "}" and stack:
            start, is_fn = stack.pop()
            if is_fn:
                spans.append((start, i))
        i += 1
    return spans


def _enclosing(spans, pos, depth=3):
    inner = sorted((s for s in spans if s[0] <= pos <= s[1]), key=lambda s: s[1] - s[0])
    return inner[:depth]


STAYS_INSIDE = re.compile(
    r"/(?:delete|bulk-delete|delete-selected|delete-bulk|delete-prompts?-bulk|delete-prompt|clear|clear-queue|"
    r"clear-all|clear-prompts|remove-from-queue|unqueue|cancel|bulk-remove-platform)\b"
    r"|method\s*:\s*['\"]DELETE['\"]")
LEAVES = re.compile(r"/(?:post-now|publish|disconnect|send)\b|/(?:linkedin|threads|facebook|twitter|instagram)/post/")
URL_LITERAL = re.compile(r"""`[^`]*`|'[^'\n]*'|"[^"\n]*"|method\s*:\s*['"]DELETE['"]""")
INSIDE_NAME = re.compile(r"""url_for\(\s*["'][^"']*(?:delete|remove|clear|unqueue|cancel)[^"']*["']""")
LEAVES_NAME = re.compile(r"""url_for\(\s*["'][^"']*(?:disconnect|publish|post_now|send)[^"']*["']""")
GENERIC_LABELS = {"ok", "yes", "confirm", "continue", "proceed", "sure"}
UNDO_MARK = re.compile(r"ux-undo-ok:\s*\S[^\n]{0,100}")
CONFIRM_MARK = re.compile(r"ux-confirm-ok:\s*\S[^\n]{0,100}")


def _call_text(js, start):
    """The text of a call from its '(' to the matching ')', roughly."""
    depth, i = 0, js.index("(", start)
    for j in range(i, min(len(js), i + 3000)):
        if js[j] == "(":
            depth += 1
        elif js[j] == ")":
            depth -= 1
            if depth == 0:
                return js[i:j + 1]
    return js[i:i + 3000]


def undo_problems(path):
    html, name, problems = read(path), rel(path), []
    marks = []
    for offset, js in _scripts_of(html):
        spans = _function_spans(js)
        line_of = lambda pos: html.count("\n", 0, offset + pos) + 1
        for m in URL_LITERAL.finditer(js):
            target = m.group(0)
            if "<" in target and ">" in target:
                continue  # markup built in a script; the tag scan below reads it
            scopes = [js[s:e + 1] for s, e in _enclosing(spans, m.start())]
            if STAYS_INSIDE.search(target) or INSIDE_NAME.search(target):
                if any("Insights.deferred(" in s for s in scopes):
                    continue
                if scopes and UNDO_MARK.search(scopes[0]):
                    marks.append(f"{name}:{line_of(m.start())} {UNDO_MARK.search(scopes[0]).group(0)}")
                    continue
                problems.append(f"{name}:{line_of(m.start())}: {target[:70]} deletes without Insights.deferred (Undo)")
            elif LEAVES.search(target) or LEAVES_NAME.search(target):
                if any("Insights.confirm(" in s for s in scopes):
                    continue
                if scopes and CONFIRM_MARK.search(scopes[0]):
                    marks.append(f"{name}:{line_of(m.start())} {CONFIRM_MARK.search(scopes[0]).group(0)}")
                    continue
                problems.append(f"{name}:{line_of(m.start())}: {target[:70]} leaves Insights without Insights.confirm")
        for m in re.finditer(r"Insights\.confirm\(", js):
            call = _call_text(js, m.start())
            label = re.search(r"confirmLabel\s*:\s*(`[^`]*`|'[^']*'|\"[^\"]*\"|[\w.]+)", call)
            if not label:
                problems.append(f"{name}:{line_of(m.start())}: Insights.confirm without a confirmLabel naming the action")
            elif label.group(1).strip("'\"`").strip().lower() in GENERIC_LABELS:
                problems.append(f"{name}:{line_of(m.start())}: confirmLabel {label.group(1)} does not name the action")
    for m in re.finditer(r"<(form|a)\b([^>]*)>", html, re.I):
        tag, attrs = m.group(1).lower(), m.group(2)
        url = re.search(r"""\b(?:action|href)\s*=\s*("[^"]*"|'[^']*')""", attrs)
        if not url:
            continue
        target = url.group(1)
        line = html.count("\n", 0, m.start()) + 1
        inside = STAYS_INSIDE.search(target) or re.search(r"url_for\(\s*'[^']*(?:delete|remove|clear)[^']*'", target)
        leaves = LEAVES.search(target) or re.search(r"url_for\(\s*'[^']*(?:disconnect|publish|post_now|send)[^']*'", target)
        if inside and "data-undo" not in attrs:
            problems.append(f"{name}:{line}: <{tag}> to {target[:60]} deletes without data-undo")
        if leaves and "data-confirm" not in attrs and tag == "form":
            problems.append(f"{name}:{line}: <form> to {target[:60]} leaves Insights without data-confirm")
        if "data-confirm" in attrs and "data-confirm-label" not in attrs:
            problems.append(f"{name}:{line}: data-confirm without data-confirm-label naming the action")
    return problems, marks


def mode_undo():
    problems, marks = [], []
    for path in template_files():
        p, m = undo_problems(path)
        problems += p
        marks += m
    caught = undo_problems(os.path.join(FIXTURES, "bad.html"))[0]
    if len(caught) < 4:
        fail(f"the undo scan found {len(caught)} of the 4 planted problems in ux_fixtures/bad.html; it is blind")
    if undo_problems(os.path.join(FIXTURES, "clean.html"))[0]:
        fail("the undo scan flags ux_fixtures/clean.html; it cries wolf")
    for m in marks:
        print(f"  marked: {m}")
    if problems:
        for p in problems:
            print(f"  {p}")
        fail(f"{len(problems)} deletes without Undo or exits without a named confirm")
    print(f"every delete inside Insights is deferred with Undo and every exit is confirmed by name; {len(marks)} marked exceptions")
    print("UX_UNDO_OK")


# ------------------------------------------------------------- rendered pages

from html.parser import HTMLParser

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class _El:
    def __init__(self, tag, attrs, parent):
        self.tag, self.attrs, self.parent, self.kids, self.text = tag, dict(attrs), parent, [], []

    def all(self):
        for k in self.kids:
            yield k
            yield from k.all()

    def find(self, pred):
        return [e for e in self.all() if pred(e)]

    def content(self):
        return " ".join(" ".join(self.text).split() + [k.content() for k in self.kids]).strip()

    def has(self, name):
        return name in self.attrs

    def cls(self, name):
        return name in (self.attrs.get("class") or "").split()


class _Tree(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = _El("#root", {}, None)
        self.cur = self.root
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        el = _El(tag, attrs, self.cur)
        self.cur.kids.append(el)
        if tag not in VOID:
            self.cur = el

    def handle_endtag(self, tag):
        node = self.cur
        while node is not self.root and node.tag != tag:
            node = node.parent
        if node is not self.root:
            self.cur = node.parent

    def handle_data(self, data):
        if self.cur.tag not in ("script", "style"):
            self.cur.text.append(data)


def _dom(html):
    return _Tree(html).root


def _by_id(root, ident):
    hit = root.find(lambda e: e.attrs.get("id") == ident)
    return hit[0] if hit else None


def _reason_for(root, el):
    """The element that says why `el` is disabled, via aria-describedby, or None."""
    for ident in (el.attrs.get("aria-describedby") or "").split():
        target = _by_id(root, ident)
        if target is not None and target.content():
            return target
    return None


def _app():
    sys.path.insert(0, HERE)
    import _gate_common
    return _gate_common.isolated_app()


# ------------------------------------------------- disabled with a reason (G9)

def mode_disabled_states():
    problems = []
    _dir, database, web, client = _app()
    db = database.DB_PATH
    post = database.add_standalone_post("freeform", "x", "linkedin", "A post for LinkedIn", db_path=db)

    def card(html):
        root = _dom(html)
        hit = root.find(lambda e: e.cls("post-item") and e.attrs.get("data-post-id") == str(post))
        return root, (hit[0] if hit else None)

    # 1. Nothing connected: posting or queueing to LinkedIn cannot work yet.
    root, c = card(client.get("/compose").get_data(as_text=True))
    if c is None:
        problems.append("compose: the seeded post's card did not render")
    else:
        post_now = c.find(lambda e: e.tag == "button" and "Post to" in e.content() and "now" in e.content()
                          and "LinkedIn" in e.content())
        queue = c.find(lambda e: e.tag == "button" and e.cls("chip-queue"))
        for label, found in (("the Post to LinkedIn now item", post_now), ("the chip's queue button", queue)):
            if not found:
                problems.append(f"compose: {label} is missing")
                continue
            b = found[0]
            if not b.has("disabled"):
                problems.append(f"compose: {label} is clickable with no LinkedIn account connected")
            why = _reason_for(root, b)
            if why is None or "connect" not in why.content().lower():
                problems.append(f"compose: {label} does not say why (aria-describedby to text naming Connect)")
            elif not why.find(lambda e: e.tag == "a" and (e.attrs.get("href") or "").startswith("/accounts")):
                problems.append("compose: the reason does not link to Accounts, where it gets fixed")

    # 2. Connected: the same controls work and no reason is shown.
    database.upsert_social_account("linkedin", "urn:li:person:gate", display_name="Gate", db_path=db)
    root, c = card(client.get("/compose").get_data(as_text=True))
    if c is not None:
        post_now = c.find(lambda e: e.tag == "button" and "Post to" in e.content() and "LinkedIn" in e.content())
        if post_now and post_now[0].has("disabled"):
            problems.append("compose: Post to LinkedIn stays disabled once LinkedIn is connected")
        if c.find(lambda e: e.cls("card-why")):
            problems.append("compose: the 'connect first' reason still shows once LinkedIn is connected")

    # 3. Schedule with nothing pending: Clear Pending has nothing to clear.
    root = _dom(client.get("/schedule").get_data(as_text=True))
    clear = root.find(lambda e: e.tag == "button" and "Clear Pending" in e.content())
    if not clear:
        problems.append("schedule: the Clear Pending button is missing")
    elif not clear[0].has("disabled") or _reason_for(root, clear[0]) is None:
        problems.append("schedule: Clear Pending is clickable (or unexplained) with an empty queue")
    database.add_scheduled_post(scheduled_for="2099-01-01T09:00:00", post_type="standalone", standalone_post_id=post, platform="linkedin",
                                status="pending", db_path=db)
    root = _dom(client.get("/schedule").get_data(as_text=True))
    clear = root.find(lambda e: e.tag == "button" and "Clear Pending" in e.content())
    if clear and clear[0].has("disabled"):
        problems.append("schedule: Clear Pending stays disabled with a post pending")

    # 4. Accounts: a platform without API credentials cannot be connected.
    real = web._platform_is_configured
    web._platform_is_configured = lambda platform: platform != "facebook"
    try:
        root = _dom(client.get("/accounts").get_data(as_text=True))
    finally:
        web._platform_is_configured = real
    block = root.find(lambda e: e.cls("account-platform") and e.attrs.get("data-platform") == "facebook")
    if not block:
        problems.append("accounts: the Facebook block is missing")
    else:
        btn = block[0].find(lambda e: e.tag == "button" and "Connect" in e.content())
        if not btn or not btn[0].has("disabled"):
            problems.append("accounts: an unconfigured platform has no disabled Connect button")
        else:
            why = _reason_for(root, btn[0])
            if why is None or ".env" not in why.content():
                problems.append("accounts: the disabled Connect does not say the API keys go in .env")
        live = root.find(lambda e: e.cls("account-platform") and e.attrs.get("data-platform") == "linkedin")
        if live and live[0].find(lambda e: e.tag == "button" and "Connect" in e.content() and e.has("disabled")):
            problems.append("accounts: a configured platform's Connect is disabled")

    if problems:
        for p in problems:
            print(f"  {p}")
        fail(f"{len(problems)} impossible actions are clickable or unexplained")
    print("compose, schedule and accounts disable what cannot work yet and say why, next to it, with the fix")
    print("UX_DISABLED_STATES_OK")


# ----------------------------------------------------------------- flows (G11)

def _first_run(html):
    root = _dom(html)
    card = _by_id(root, "first-run")
    return root, card


def mode_flows():
    problems = []
    _dir, database, web, client = _app()
    db = database.DB_PATH

    root, card = _first_run(client.get("/compose").get_data(as_text=True))
    if card is None:
        problems.append("a fresh install's Compose has no getting-started card")
    else:
        if "0 of 4 done" not in card.content():
            problems.append(f"the fresh checklist does not read 0 of 4 done: {card.content()[:120]}")
        go = card.find(lambda e: e.tag == "a" and e.cls("btn-primary"))
        if len(go) != 1 or go[0].attrs.get("href") != "/accounts":
            problems.append("the fresh checklist's one primary action is not Connect an account (/accounts)")

    account = database.upsert_social_account("linkedin", "urn:li:person:flow", display_name="Flow", db_path=db)
    post = database.add_standalone_post("freeform", "x", "linkedin", "First post", db_path=db, account_id=account)
    sched = database.add_scheduled_post(scheduled_for="2099-01-01T09:00:00", post_type="standalone", standalone_post_id=post,
                                        platform="linkedin", status="pending", db_path=db)
    root, card = _first_run(client.get("/compose").get_data(as_text=True))
    if card is None:
        problems.append("the checklist vanished before the last step was done")
    else:
        if "3 of 4 done" not in card.content():
            problems.append(f"after connecting, writing and queueing it does not read 3 of 4 done: {card.content()[:120]}")
        done_steps = [e.attrs.get("data-step") for e in card.find(lambda e: e.cls("is-done"))]
        if done_steps != ["accounts", "posts", "queued"]:
            problems.append(f"the done steps are {done_steps}, not accounts, posts, queued")
        go = card.find(lambda e: e.tag == "a" and e.cls("btn-primary"))
        if len(go) != 1 or go[0].attrs.get("href") != "/schedule":
            problems.append("with only 'see it go out' left, the primary action is not Open Schedule")

    database.update_scheduled_post_status(sched, "posted", db_path=db)
    root, card = _first_run(client.get("/compose").get_data(as_text=True))
    if card is not None:
        problems.append("the checklist still shows once every step is done")

    # The end of the queue and publish flows: what happened, and the next step.
    js = "\n".join(body for _, body in _scripts_of(read(os.path.join(TEMPLATES, "compose.html"))))
    spans = _function_spans(js)
    for name in ("toggleQueue", "queueCardToAll", "postNowFromCard"):
        m = re.search(r"(?:async\s+)?function\s+" + name + r"\s*\(", js)
        if not m:
            problems.append(f"compose.html has no function {name}")
            continue
        body = [js[s:e + 1] for s, e in spans if s > m.start() and s - m.end() < 400]
        body = min(body, key=len) if body else ""
        if not re.search(r"Insights\.done\([^;]*\{[^}]*label\s*:[^}]*(href|run)\s*:", body, re.S):
            problems.append(f"{name} does not end with Insights.done(message, {{label, href}}) naming the next step")

    if problems:
        for p in problems:
            print(f"  {p}")
        fail(f"{len(problems)} flow problems")
    print("the getting-started card counts real progress and leaves when done; queue and publish end on the next step")
    print("UX_FLOWS_OK")


BUILT = {
    "ratchet": "UX_RATCHET_OK", "control": "UX_CONTROL_OK", "principles": "UX_PRINCIPLES_OK",
    "claude-md": "UX_CLAUDE_MD_OK", "shell": "UX_SHELL_OK", "behavior": "UX_BEHAVIOR_OK",
    "targets": "UX_TARGETS_OK", "registered": "UX_REGISTERED_OK",
    "zero": "UX_ZERO_DEBT_OK", "harness": "UX_HARNESS_PARITY_OK", "undo": "UX_UNDO_OK",
    "disabled-states": "UX_DISABLED_STATES_OK", "flows": "UX_FLOWS_OK",
}
LATER = {}


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
    "zero": mode_zero, "harness": mode_harness, "undo": mode_undo,
    "disabled-states": mode_disabled_states, "flows": mode_flows,
    "--lower": mode_lower, "--init": mode_init,
}


def main(argv):
    mode = argv[1] if len(argv) > 1 else "ratchet"
    if mode in ("zero", "undo") and len(argv) > 2:
        ONLY.extend(argv[2:])
    if mode in LATER:
        fail(f"mode {mode!r} is not built yet (rollout {LATER[mode]}, docs/ux-rollout/PLAN.md)")
    if mode not in MODES:
        fail(f"unknown mode {mode!r}; one of {', '.join(MODES)}")
    MODES[mode]()


if __name__ == "__main__":
    main(sys.argv)
