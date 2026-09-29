"""Gates for writing posts by hand, and for the model adapters behind generation.

    python scripts/check_manual_posts.py route     MANUAL_POSTS_ROUTE_OK
    python scripts/check_manual_posts.py wiring    MANUAL_POSTS_WIRING_OK
    python scripts/check_manual_posts.py targets   MANUAL_POSTS_TARGETS_OK
    python scripts/check_manual_posts.py video     MANUAL_POSTS_VIDEO_OK
    python scripts/check_manual_posts.py llm       LLM_ADAPTERS_OK

``route`` drives the real /compose/post/create endpoint with the same targets the
composer sends and checks the rows it writes. ``wiring`` renders the real
/compose page and checks the manual tab is first, is the default, hides what only
a model needs, and is bound to that endpoint. ``targets`` runs the composer's
target-building JavaScript under node. ``video`` runs the tab's video handlers
and Save under node against a stub page, then replays exactly what Save sent to
the real route. ``llm`` exercises both model adapters
against recording fakes: they are what let the call sites, written for GPT-4-era
``max_tokens``/``temperature``, keep working on GPT-6 and Claude 5.

All five run on a throwaway database with fake clients, so none needs a key,
makes a network call, or touches a real insights.db.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import isolated_app, connect, check  # noqa: E402


def rig():
    directory, database, web, publisher, client = isolated_app()
    # Creating a post fires a background og:image fetch that uploads to the
    # real Cloudinary when the repo .env is found; a gate must not.
    web._maybe_attach_link_image = lambda *a, **k: None
    ids = {
        "work": connect(database, "linkedin", "li-work", "Work"),
        "studio": connect(database, "linkedin", "li-studio", "Studio"),
        "threads": connect(database, "threads", "th-1", "brandco"),
    }
    return database, web, client, ids


def post(client, targets, content="A post written by hand", image=None):
    data = {"content": content, "targets": list(targets)}
    if image:
        data["image_url"] = image
    return client.post("/compose/post/create", data=data)


# ── route ────────────────────────────────────────────────────────────────

def section_route():
    database, web, client, ids = rig()
    image = "https://example.test/pic.jpg"
    response = post(client, [f"linkedin:{ids['work']}", f"linkedin:{ids['studio']}", "threads"],
                    image=image)
    body = response.get_json()
    check(response.status_code == 200 and body.get("success"), f"create failed: {body}")
    check(len(body["post_ids"]) == 3, f"expected one row per target, got {body['post_ids']}")

    rows = [dict(database.get_standalone_post(i, db_path=database.DB_PATH)) for i in body["post_ids"]]
    check({r["source_type"] for r in rows} == {"manual"}, "a hand-written post must be source_type 'manual'")
    check({r["content"] for r in rows} == {"A post written by hand"}, "every row must carry the same copy")
    check({r["image_url"] for r in rows} == {image}, "every row must carry the chosen image")
    li = sorted(r["account_id"] for r in rows if r["platform"] == "linkedin")
    check(li == sorted([ids["work"], ids["studio"]]),
          f"both LinkedIn accounts must get their own row, got {li}")
    th = [r for r in rows if r["platform"] == "threads"]
    check(len(th) == 1 and th[0]["account_id"] == ids["threads"],
          "a bare platform name must land on that platform's default account")

    groups = web._group_standalone_posts(
        database.list_standalone_posts(db_path=database.DB_PATH))
    check(len(groups) == 1, f"three rows of one post must show as ONE card, got {len(groups)}")

    # Rejections: each must say why and write nothing.
    before = len(database.list_standalone_posts(db_path=database.DB_PATH))
    empty = post(client, ["threads"], content="   ")
    check(empty.status_code == 400 and "Content" in empty.get_json()["error"],
          "blank copy must be refused")
    wrong = post(client, [f"threads:{ids['work']}"])
    check(wrong.status_code == 400, "an account belonging to another platform must be refused")
    none = client.post("/compose/post/create", data={"content": "x"})
    check(none.status_code == 400, "a post with no platform must be refused")
    check(len(database.list_standalone_posts(db_path=database.DB_PATH)) == before,
          "a refused request must not write any row")

    # The Recent Prompts list is built from freeform source_content only.
    check(not database.list_recent_prompts(db_path=database.DB_PATH),
          "hand-written posts leaked into the recent-prompts list")
    print("manual create: one row per ticked account, one card, refusals write nothing")
    print("MANUAL_POSTS_ROUTE_OK")


# ── wiring ───────────────────────────────────────────────────────────────

def function_body(html, name):
    """The source of ``function name(...) {...}`` up to the next top-level function."""
    match = re.search(rf"(?:async\s+)?function\s+{name}\s*\(", html)
    check(match, f"the page does not define {name}()")
    rest = html[match.start():]
    nxt = re.search(r"\n(?:async\s+)?function\s+\w+\s*\(", rest[1:])
    return rest[: nxt.start() + 1] if nxt else rest


def tag_containing(html, marker):
    """The opening tag that carries ``marker``."""
    at = html.index(marker)
    return html[html.rindex("<", 0, at): html.index(">", at) + 1]


def section_wiring():
    database, web, client, ids = rig()
    html = client.get("/compose").get_data(as_text=True)
    check(client.get("/compose").status_code == 200, "/compose did not render")

    # --- tab order and default -----------------------------------------------
    tabs = re.findall(r'<button class="nav-link[^"]*" id="([a-z-]+-tab)"', html)
    tabs = [t for t in tabs if t in {"manual-tab", "freeform-tab", "url-tab", "text-tab",
                                     "image-tab", "file-tab", "saved-tab", "import-tab"}]
    check(tabs and tabs[0] == "manual-tab", f"the manual tab must come first, order is {tabs}")
    check("active" in tag_containing(html, 'id="manual-tab"'), "manual must be the default tab")
    check("active" not in tag_containing(html, 'id="freeform-tab"'),
          "Freeform must no longer be the default tab")
    check("show active" in tag_containing(html, 'id="manual-content"'),
          "the manual pane must be the one shown")
    check("show active" not in tag_containing(html, 'id="freeform-content"'),
          "two panes are shown at once")

    # A "use this saved source" link must still win over the manual default.
    src = database.add_url_source("https://example.test/a", "T", "D", "body", db_path=database.DB_PATH)
    linked = client.get(f"/compose?source_id={src}").get_data(as_text=True)
    check("active" in tag_containing(linked, 'id="saved-tab"'), "source_id no longer opens the saved tab")
    check("active" not in tag_containing(linked, 'id="manual-tab"'),
          "manual and saved source are both active")

    # --- what a model needs is hidden for manual, what manual needs is not ----
    for marker in ('AI Provider:', 'for="posts-per-platform"', 'Tone & Style',
                   'for="extra-context"', 'id="generate-btn"'):
        at = html.index(marker)
        check("ai-only" in html[max(0, at - 350):at],
              f"{marker!r} is not inside an ai-only wrapper, so the manual tab would show it")
    save = tag_containing(html, 'id="manual-save-btn"')
    check('onclick="saveManualPost()"' in save, "the Save button is not bound to saveManualPost()")
    wrapper_at = html.rindex('<div class="manual-only">', 0, html.index('id="manual-save-btn"'))
    check(html.index('id="generate-btn"') > html.index("</div>", wrapper_at),
          "the Generate button ended up inside the manual-only wrapper")
    check("#generator-body.mode-manual .ai-only { display: none !important; }" in html,
          "no rule hides the model controls in manual mode")
    check("#generator-body .manual-only { display: none; }" in html
          and "#generator-body.mode-manual .manual-only { display: grid; }" in html,
          "no rule shows Save only in manual mode")

    # --- the shared pickers are the ones the manual tab reads -----------------
    check('id="platform-checkboxes"' in html and 'id="final-image-url"' in html,
          "the shared platform and image pickers are gone")
    save_src = function_body(html, "saveManualPost")
    for token in ("/compose/post/create", "formData.append('targets'",
                  "formData.append('content'", "formData.append('image_url'",
                  "final-image-url", "manualPostTargets()"):
        check(token in save_src, f"saveManualPost() never uses `{token}`")
    check("getActiveSourceType" in function_body(html, "applyGeneratorMode")
          and "'manual'" in function_body(html, "applyGeneratorMode"),
          "the mode switch does not key off the active tab")
    check("id === 'manual-tab'" in function_body(html, "getActiveSourceType"),
          "getActiveSourceType() does not know the manual tab")
    check("manual-input" in function_body(html, "getContent"), "getContent() does not read the manual box")

    # --- one declaration each, none shadowing an older one --------------------
    for name in ("applyGeneratorMode", "tickedPlatforms", "manualPostTargets",
                 "updateManualComposerCount", "saveManualPost", "createManualPost",
                 "updateManualCharCount", "showAddPostForm", "manualVideoChoice",
                 "refreshManualVideoUI", "onManualVideoFile", "onManualVideoUrl",
                 "clearManualVideo"):
        found = len(re.findall(rf"function\s+{name}\s*\(", html))
        check(found == 1, f"{name}() is declared {found} times")

    # --- the video control ------------------------------------------------------
    pane = html[html.index('id="manual-content"'): html.index('id="freeform-content"')]
    for needle in ('id="manual-video-file"', 'accept="video/mp4,video/quicktime',
                   'onchange="onManualVideoFile(this)"', 'id="manual-video-url"',
                   'oninput="onManualVideoUrl(this)"', 'onclick="clearManualVideo()"',
                   'id="manual-video-note"', "Reel"):
        check(needle in pane, f"the manual pane's video control is missing `{needle}`")
    for token in ("formData.append('video'", "formData.append('video_url'", "video_warnings",
                  "manualVideoChoice()"):
        check(token in save_src, f"saveManualPost() never uses `{token}`")
    # The in-list form's own ids must be untouched: its gate keys on them.
    for needle in ('id="new-post-video-file"', 'id="new-post-video-url"',
                   'onchange="onNewPostVideoFile(this)"'):
        check(needle in html, f"the in-list composer lost `{needle}`")

    # --- the older in-list composer is still there ----------------------------
    check('id="add-post-btn"' in html or "new_post_composer" in html or "Write a New Post" in html,
          "the in-list composer was removed")
    print("manual tab: first, default, hides model controls, shows Save, bound to /compose/post/create")
    print("MANUAL_POSTS_WIRING_OK")


# ── targets (JavaScript, under node) ─────────────────────────────────────

def section_targets():
    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    database, web, client, ids = rig()
    html = client.get("/compose").get_data(as_text=True)
    source = function_body(html, "manualPostTargets")

    # The page's own function, run against stand-ins for the two pickers it reads.
    script = ("let selectedAccountTargets, tickedPlatforms;\n" + source + """
    const cases = JSON.parse(process.argv[1]);
    console.log(JSON.stringify(cases.map(c => {
        selectedAccountTargets = () => c.accounts;
        tickedPlatforms = () => c.ticked;
        return manualPostTargets();
    })));
    """)
    cases = [
        # two LinkedIn accounts and Threads: accounts named, Threads bare
        {"accounts": ["linkedin:1", "linkedin:2"], "ticked": ["linkedin", "threads"],
         "want": ["linkedin:1", "linkedin:2", "threads"]},
        # a single-account platform has no account buttons: it goes bare
        {"accounts": [], "ticked": ["twitter"], "want": ["twitter"]},
        # one of two accounts ticked
        {"accounts": ["linkedin:2"], "ticked": ["linkedin"], "want": ["linkedin:2"]},
        # every account of a platform unticked: default account, so bare
        {"accounts": [], "ticked": ["linkedin", "threads"], "want": ["linkedin", "threads"]},
        # nothing ticked: nothing to save
        {"accounts": [], "ticked": [], "want": []},
    ]
    import json
    result = subprocess.run([node, "-e", script, json.dumps(cases)], capture_output=True, text=True)
    check(result.returncode == 0, f"node failed:\n{result.stderr.strip()[:600]}")
    got = json.loads(result.stdout.strip().splitlines()[-1])
    for case, actual in zip(cases, got):
        check(actual == case["want"], f"{case['ticked']} + {case['accounts']} -> {actual}, wanted {case['want']}")

    # And the server accepts exactly what that produced, end to end.
    good = post(client, [f"linkedin:{ids['work']}", "threads", "twitter"])
    check(good.status_code == 200 and len(good.get_json()["post_ids"]) == 3,
          f"the server refused targets the composer builds: {good.get_json()}")
    print(f"target building: {len(cases)} cases, and the server accepts what they produce")
    print("MANUAL_POSTS_TARGETS_OK")


# ── video (JavaScript under node, then the real route) ───────────────────

VIDEO_HARNESS = r"""
// A stub page: just the elements the manual tab's video code reads and writes.
const els = {};
const make = (id) => els[id] = {
  id, value: '', files: [], textContent: '', innerHTML: '', disabled: false,
  classes: new Set(['d-none']),
  classList: {
    add: (c) => els[id].classes.add(c), remove: (c) => els[id].classes.delete(c),
    contains: (c) => els[id].classes.has(c),
    toggle: (c, on) => { (on === undefined ? !els[id].classes.has(c) : on) ? els[id].classes.add(c) : els[id].classes.delete(c); },
  },
  focus() {},
};
['manual-video-file', 'manual-video-url', 'manual-video-name', 'manual-video-clear',
 'manual-video-note', 'manual-input', 'manual-save-btn', 'final-image-url'].forEach(make);
// A real file input drops its files when its value is set to ''.
(() => {
  const input = els['manual-video-file'];
  let value = '';
  Object.defineProperty(input, 'value', {
    get: () => value,
    set: (v) => { value = v; if (v === '') input.files = []; },
  });
})();
const toasts = [], alerts = [], sent = [];
const document = { getElementById: (id) => els[id] };
const showToast = (m, t) => toasts.push([t, m]);
const alert = (m) => alerts.push(m);
const confirm = () => true;
const platformCharLimits = { linkedin: 3000, threads: 500, twitter: 280, facebook: 5000, instagram: 2200 };
const PLATFORM_LABELS = { linkedin: 'LinkedIn', threads: 'Threads', twitter: 'X', facebook: 'Facebook', instagram: 'Instagram' };
const counts = [];
const flush = () => new Promise((resolve) => setImmediate(resolve));
let updateManualComposerCount = () => counts.push(1);
let selectedAccountTargets = () => [];
let tickedPlatforms = () => ['linkedin', 'threads'];
let reloaded = 0;
const location = { reload: () => { reloaded++; } };
const setTimeout = (fn) => fn();
let reply = { success: true, post_ids: [1, 2], video_warnings: [] };
const fetch = (url, opts) => {
  sent.push({ url, entries: Array.from(opts.body.entries()).map(([k, v]) => [k, v && v.name ? 'FILE:' + v.name : v]) });
  return Promise.resolve({ json: () => Promise.resolve(reply) });
};
const file = (name, size) => { const f = new File(['x'], name); Object.defineProperty(f, 'size', { value: size }); return f; };
const state = () => ({
  fileName: els['manual-video-file'].files[0] ? els['manual-video-file'].files[0].name : null,
  fileValue: els['manual-video-file'].value, url: els['manual-video-url'].value,
  name: els['manual-video-name'].textContent,
  clearHidden: els['manual-video-clear'].classes.has('d-none'),
  noteHidden: els['manual-video-note'].classes.has('d-none'),
});
const setFile = (f) => { els['manual-video-file'].files = [f]; els['manual-video-file'].value = 'C:\\fake\\' + f.name; };
const reset = () => {
  ['manual-video-file', 'manual-video-url', 'final-image-url', 'manual-input'].forEach((id) => { els[id].value = ''; });
  els['manual-video-file'].files = [];
  toasts.length = 0; alerts.length = 0; sent.length = 0; counts.length = 0; reloaded = 0;
  reply = { success: true, post_ids: [1, 2], video_warnings: [] };
  els['manual-save-btn'].disabled = false;
};
"""


def section_video():
    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    database, web, client, ids = rig()
    html = client.get("/compose").get_data(as_text=True)

    def body(name):
        """Exactly one function's source, found by matching its braces."""
        text = function_body(html, name)
        start = text.index("{", text.index(")"))
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                return text[: i + 1]
        check(False, f"{name}() has unbalanced braces")

    source = "\n".join(body(n) for n in (
        "manualVideoChoice", "refreshManualVideoUI", "onManualVideoFile", "onManualVideoUrl",
        "clearManualVideo", "manualPostTargets", "saveManualPost"))
    const = re.search(r"const MANUAL_VIDEO_MAX_BYTES = [^;]+;", html)
    check(const, "the upload cap constant is gone")

    script = VIDEO_HARNESS + const.group(0) + "\n" + source + r"""
(async () => {
  const out = {};
  const MB = 1024 * 1024;

  reset(); setFile(file('clip.avi', 1000)); onManualVideoFile(els['manual-video-file']);
  out.badType = { ...state(), toasts: toasts.slice() };

  reset(); setFile(file('clip.mp4', 101 * MB)); onManualVideoFile(els['manual-video-file']);
  out.tooBig = { ...state(), toasts: toasts.slice() };

  reset(); els['manual-video-url'].value = 'https://old.test/x.mp4';
  setFile(file('clip.mp4', 5 * MB)); onManualVideoFile(els['manual-video-file']);
  out.fileClearsUrl = state();

  reset(); setFile(file('old.mp4', 1000));
  els['manual-video-url'].value = 'https://cdn.test/y.mp4'; onManualVideoUrl(els['manual-video-url']);
  out.urlClearsFile = state();

  reset(); setFile(file('clip.mov', 1000)); onManualVideoFile(els['manual-video-file']);
  clearManualVideo(); out.cleared = state();
  out.recounted = counts.length;

  // Save: a URL video with an image also chosen sends the video and NOT the image.
  reset(); els['manual-input'].value = 'A post with a clip';
  els['final-image-url'].value = 'https://img.test/a.jpg';
  els['manual-video-url'].value = ' https://cdn.test/y.mp4 ';
  saveManualPost(); await flush();
  out.saveUrl = { sent: sent.slice(), reloaded };

  // Save: an uploaded file goes as `video`, no video_url, no image.
  reset(); els['manual-input'].value = 'A post with an upload';
  els['final-image-url'].value = 'https://img.test/a.jpg';
  setFile(file('clip.mp4', 1234));
  saveManualPost(); const busy = els['manual-save-btn'].innerHTML;
  await flush();
  out.saveFile = { sent: sent.slice(), busy };

  // Save: no video keeps sending the image, exactly as before.
  reset(); els['manual-input'].value = 'A plain post';
  els['final-image-url'].value = 'https://img.test/a.jpg';
  saveManualPost(); await flush();
  out.savePlain = sent.slice();

  // Save: the platforms' warnings about the video are shown before the reload wipes them.
  reset(); els['manual-input'].value = 'Short clip';
  els['manual-video-url'].value = 'https://cdn.test/short.mp4';
  reply = { success: true, post_ids: [1], video_warnings: [
    { level: 'warning', message: 'LinkedIn wants at least 3 seconds' },
    { level: 'error', message: 'Instagram cannot take this' }] };
  saveManualPost(); await flush();
  out.warnings = { alerts: alerts.slice(), reloaded };

  console.log(JSON.stringify(out));
})();
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True)
    check(result.returncode == 0, f"node failed:\n{result.stderr.strip()[:800]}")
    out = json.loads(result.stdout.strip().splitlines()[-1])

    bad = out["badType"]
    check(bad["fileValue"] == "" and bad["clearHidden"] and bad["toasts"]
          and "MP4 or MOV" in bad["toasts"][0][1], f"an AVI should be refused and cleared: {bad}")
    big = out["tooBig"]
    check(big["fileValue"] == "" and big["clearHidden"] and "100MB" in big["toasts"][0][1],
          f"a file over 100MB should be refused and cleared: {big}")
    fc = out["fileClearsUrl"]
    check(fc["fileName"] == "clip.mp4" and fc["url"] == "" and "5.0 MB" in fc["name"]
          and not fc["clearHidden"] and not fc["noteHidden"],
          f"choosing a file should clear the URL and show the file: {fc}")
    uc = out["urlClearsFile"]
    check(uc["fileValue"] == "" and uc["url"] == "https://cdn.test/y.mp4" and not uc["noteHidden"],
          f"typing a URL should clear the file: {uc}")
    cl = out["cleared"]
    check(cl["fileValue"] == "" and cl["url"] == "" and cl["clearHidden"] and cl["noteHidden"],
          f"Remove should empty the control: {cl}")
    check(out["recounted"] == 2, "the character/Instagram hint is not refreshed as the video changes")

    url_send = out["saveUrl"]["sent"]
    check(len(url_send) == 1 and url_send[0]["url"] == "/compose/post/create", f"Save did not post to the route: {url_send}")
    fields = url_send[0]["entries"]
    check(["video_url", "https://cdn.test/y.mp4"] in fields, f"the URL was not sent trimmed: {fields}")
    check(not any(k in ("image_url", "video") for k, _ in fields),
          f"a video post must not also send an image or a file: {fields}")
    check(["targets", "linkedin"] in fields and ["targets", "threads"] in fields
          and ["content", "A post with a clip"] in fields, f"targets/content missing: {fields}")
    check(out["saveUrl"]["reloaded"] == 1, "the page should reload after a successful save")

    file_send = out["saveFile"]["sent"][0]["entries"]
    check(["video", "FILE:clip.mp4"] in file_send and not any(k in ("video_url", "image_url") for k, _ in file_send),
          f"an uploaded video should go as `video` alone: {file_send}")
    check("Uploading video" in out["saveFile"]["busy"], "the button should say the video is uploading")
    plain = out["savePlain"][0]["entries"]
    check(["image_url", "https://img.test/a.jpg"] in plain and not any(k.startswith("video") for k, _ in plain),
          f"a post without a video must still send its image: {plain}")
    w = out["warnings"]
    check(len(w["alerts"]) == 1 and "3 seconds" in w["alerts"][0] and "⛔" in w["alerts"][0]
          and "⚠️" in w["alerts"][0], f"video warnings must be shown before the reload: {w}")

    # --- replay what Save sent against the real route ---------------------------
    web.video_media.head_size = lambda url: 5 * 1024 * 1024   # no network
    created = []
    web._maybe_attach_link_image = lambda *a, **k: created.append(a)
    replay = {}
    for k, v in fields:
        replay.setdefault(k, []).append(v)
    replay["content"] = "Replayed manual video post"
    replay["video_url"] = "http://93.184.216.34/clip.mp4"   # a literal public IP: no DNS
    response = client.post("/compose/post/create", data=replay, content_type="multipart/form-data")
    body_json = response.get_json()
    check(response.status_code == 200 and body_json.get("success"), f"the route refused what Save sends: {body_json}")
    rows = [dict(r) for r in database.list_standalone_posts(db_path=database.DB_PATH)
            if r["content"] == "Replayed manual video post"]
    check(len(rows) == 2 and all(r["video_url"] == "http://93.184.216.34/clip.mp4" and not r["image_url"]
                                 and r["source_type"] == "manual" for r in rows),
          f"every ticked account's row should carry the video and no image: {rows}")
    check(len(web._group_standalone_posts(database.list_standalone_posts(db_path=database.DB_PATH))) == 1,
          "the video post should be one card")
    check(not created, "a post with a video must not go looking for a link image")
    print("video: the tab validates, excludes file/URL, sends the video instead of the image, and the route takes it")
    print("MANUAL_POSTS_VIDEO_OK")


# ── llm ──────────────────────────────────────────────────────────────────

def section_llm():
    import httpx
    import openai
    import insights
    import usage_meter

    # --- defaults and dropdown ------------------------------------------------
    check(insights.MODEL_CHOICES["openai"][0] == "gpt-6.1-sol", "OpenAI list should lead with gpt-6.1-sol")
    check("claude-opus-5-5" in insights.MODEL_CHOICES["anthropic"], "Opus 5.5 missing from the Claude list")
    for provider, stale in (("openai", ("gpt-4o", "gpt-4.1")), ("anthropic", ("claude-opus-4", "claude-sonnet-4"))):
        for model in insights.MODEL_CHOICES[provider]:
            check(not model.startswith(stale), f"{model} is an old model in the {provider} dropdown")
    for provider, models in insights.provider_model_options()["models"].items():
        check(len(set(models)) == len(models), f"{provider} dropdown lists a model twice: {models}")

    # --- the factory hands out the adapters ------------------------------------
    os.environ.setdefault("OPENAI_API_KEY", "gate-key")
    os.environ.setdefault("ANTHROPIC_API_KEY", "gate-key")
    client, model, provider = insights._get_llm_client(provider="openai")
    check(isinstance(client, insights._OpenAIChatClient) and provider == "openai" and model == insights.OPENAI_MODEL,
          "the OpenAI client is not wrapped, so call sites would send max_tokens/temperature to GPT-6")
    client, model, provider = insights._get_llm_client(provider="anthropic")
    check(isinstance(client, insights._AnthropicChatClient) and model == insights.ANTHROPIC_MODEL,
          "the Anthropic client is not the adapter")
    client, _, provider = insights._get_llm_client(provider="local")
    check(provider == "ollama" and not isinstance(client, insights._OpenAIChatClient),
          "the local Ollama client must not be wrapped: its models take temperature and max_tokens")
    # A model of the other provider must not cross over (a stale dropdown value).
    check(insights._get_llm_client(provider="openai", model="claude-opus-5-5")[1] == insights.OPENAI_MODEL,
          "a Claude model was sent to OpenAI")
    check(insights._get_llm_client(provider="anthropic", model="gpt-6-luna")[1] == insights.ANTHROPIC_MODEL,
          "a GPT model was sent to Anthropic")

    # --- OpenAI adapter -------------------------------------------------------
    class FakeOpenAI:
        def __init__(self, reject=None):
            self.calls, self.reject = [], list(reject or [])
            self.responses = "passthrough"
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if self.reject:
                code, param = self.reject.pop(0)
                request = httpx.Request("POST", "http://openai.test")
                raise openai.BadRequestError(
                    f"rejected {param}", response=httpx.Response(400, request=request),
                    body={"code": code, "param": param, "message": f"rejected {param}"})
            return "reply"

    def run(model, reject=None, **kwargs):
        fake = FakeOpenAI(reject)
        adapter = insights._OpenAIChatClient(fake)
        result = adapter.chat.completions.create(model=model, messages=[], **kwargs)
        return fake, result

    fake, result = run("gpt-6.1-sol", temperature=0.8, max_tokens=3000)
    sent = fake.calls[0]
    check(result == "reply" and len(fake.calls) == 1, "a GPT-6 request should go out once")
    check("temperature" not in sent and "max_tokens" not in sent,
          f"GPT-6 rejects temperature and max_tokens, but sent {sorted(sent)}")
    check(sent["max_completion_tokens"] == 3000 + insights.REASONING_TOKEN_HEADROOM,
          "reasoning tokens must have room on top of the caller's max_tokens")
    check(sent["reasoning_effort"] == insights.OPENAI_REASONING_EFFORT, "no reasoning effort was set")
    for model in ("gpt-6-astra", "gpt-6-luna", "gpt-5", "o3"):
        check("temperature" not in run(model, temperature=0.2, max_tokens=100)[0].calls[0],
              f"{model} should be treated as a reasoning model")

    fake, _ = run("gpt-4o", temperature=0.8, max_tokens=3000)
    check(fake.calls[0] == {"model": "gpt-4o", "messages": [], "temperature": 0.8, "max_tokens": 3000},
          f"a non-reasoning model's request must pass through untouched, got {fake.calls[0]}")

    fake, result = run("brand-new-model", reject=[("unsupported_value", "temperature")],
                       temperature=0.2, max_tokens=50)
    check(len(fake.calls) == 2 and "temperature" not in fake.calls[1] and fake.calls[1]["max_tokens"] == 50,
          "an unknown model that refuses temperature should be retried without it")
    fake, result = run("brand-new-model", reject=[("unsupported_parameter", "max_tokens")], max_tokens=50)
    check(fake.calls[1].get("max_completion_tokens") == 50 and "max_tokens" not in fake.calls[1],
          "a refused max_tokens should be renamed, not dropped")
    try:
        run("gpt-6.1-sol", reject=[("invalid_api_key", None)], max_tokens=5)
        check(False, "an unrelated 400 must propagate")
    except openai.BadRequestError:
        pass
    check(insights._OpenAIChatClient(FakeOpenAI()).responses == "passthrough",
          "other client attributes (responses, audio…) must pass through")

    # --- Anthropic adapter ----------------------------------------------------
    class FakeMessage:
        def __init__(self, text="hi", stop_reason="end_turn", category=None):
            self.content = [types.SimpleNamespace(type="thinking", thinking=""),
                            types.SimpleNamespace(type="text", text=text)]
            self.usage = types.SimpleNamespace(input_tokens=3, output_tokens=4)
            self.model = "m"
            self.stop_reason = stop_reason
            self.stop_details = types.SimpleNamespace(category=category, explanation=None) if category else None

    class FakeAnthropic:
        def __init__(self, message=None):
            self.calls, self.message = [], message or FakeMessage()
            create = lambda kind: (lambda **kw: (self.calls.append((kind, kw)), self.message)[1])
            self.messages = types.SimpleNamespace(create=create("plain"))
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=create("beta")))

    def ask(model, message=None, fallback=None, **kwargs):
        if fallback is not None:
            os.environ["ANTHROPIC_REFUSAL_FALLBACK"] = fallback
        try:
            adapter = insights._AnthropicChatClient("key")
        finally:
            os.environ.pop("ANTHROPIC_REFUSAL_FALLBACK", None)
        adapter._client = FakeAnthropic(message)
        result = adapter.chat.completions.create(
            model=model, messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            temperature=0.8, **kwargs)
        return adapter._client.calls, result

    calls, result = ask("claude-opus-5-5", max_tokens=160)
    kind, sent = calls[0]
    check(len(calls) == 1 and kind == "beta" and sent["fallbacks"] == "default"
          and sent["betas"] == [insights._AnthropicChatClient.FALLBACK_BETA],
          f"Opus 5.5 should request the server-side refusal fallback, got {kind} {sorted(sent)}")
    check(sent["max_tokens"] == 160 + insights.REASONING_TOKEN_HEADROOM,
          "thinking models need headroom above a small max_tokens")
    check(sent["output_config"] == {"effort": insights.ANTHROPIC_EFFORT}, "no effort was set")
    check("temperature" not in sent and sent["system"] == "s", "sampling must be dropped, system lifted")
    check(result.choices[0].message.content == "hi", "thinking blocks must not leak into the reply text")

    for model in ("claude-sonnet-5-5", "claude-fable-5-1"):
        check(ask(model, max_tokens=10)[0][0][0] == "beta", f"{model} should use the fallback")
    calls, _ = ask("claude-sonnet-5", max_tokens=10)
    check(calls[0][0] == "plain" and "output_config" in calls[0][1],
          "Sonnet 5 thinks by default (effort) but has no refusal fallback")
    calls, _ = ask("claude-haiku-4-5", max_tokens=160)
    check(calls[0][0] == "plain" and calls[0][1]["max_tokens"] == 160 and "output_config" not in calls[0][1],
          "Haiku 4.5 does not think or take effort: its request must stay as written")
    calls, _ = ask("claude-opus-4-8", max_tokens=160)
    check(calls[0][0] == "plain" and "output_config" not in calls[0][1], "Opus 4.8 must be left alone")
    calls, _ = ask("claude-opus-5-5", fallback="0", max_tokens=10)
    check(calls[0][0] == "plain" and "fallbacks" not in calls[0][1] and "output_config" in calls[0][1],
          "ANTHROPIC_REFUSAL_FALLBACK=0 should turn only the fallback off")

    try:
        ask("claude-opus-5-5", message=FakeMessage(text="", stop_reason="refusal", category="cyber"), max_tokens=10)
        check(False, "a refusal must raise, not return an empty reply")
    except insights.LLMRefusal as exc:
        check("cyber" in str(exc), f"the refusal should say why: {exc}")

    # --- metering -------------------------------------------------------------
    price = usage_meter._price_for_anthropic
    check(price("claude-opus-5-5") == {"in": 4.0, "out": 20.0}, "Opus 5.5 is mispriced")
    check(price("claude-opus-4-8") == {"in": 5.0, "out": 25.0}, "Opus 4.8 must keep its own price")
    check(price("claude-sonnet-5-5") == {"in": 2.0, "out": 10.0}, "Sonnet 5.5 is mispriced")
    check(price("claude-fable-5-1") == {"in": 10.0, "out": 50.0}, "Fable 5.1 is mispriced")
    check(usage_meter._price_for("gpt-6.1-sol") == {"in": 2.0, "out": 10.0}, "GPT-6.1 Sol is mispriced")
    check(usage_meter._price_for("gpt-6-luna") == {"in": 0.1, "out": 0.5}, "GPT-6 Luna is mispriced")
    check(usage_meter._price_for("gpt-6-astra") == {"in": 10.0, "out": 50.0}, "GPT-6 Astra is mispriced")
    print("adapters: GPT-6/Claude 5 requests are translated, older models untouched, prices resolve")
    print("LLM_ADAPTERS_OK")


SECTIONS = {"route": section_route, "wiring": section_wiring,
            "targets": section_targets, "video": section_video, "llm": section_llm}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_manual_posts.py {'|'.join(SECTIONS)}")
        sys.exit(2)
    SECTIONS[name]()
