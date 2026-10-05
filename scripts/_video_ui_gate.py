"""The UI half of the video-posting gates (``check_video.py ui``).

A helper module, not a gate of its own: it is named with a leading underscore so
the gate inventory in run_all_gates.py does not expect to run it directly.

Renders the real /compose and /schedule pages and checks four things:

* the video control, player and remove button are on the card, and every
  handler the markup names is a function the page really declares;
* the page's scripts still parse, and no function is declared twice (a second
  declaration silently replaces the first in a classic script);
* the video functions themselves behave: they are lifted out of the rendered
  page and run in node against a stubbed DOM and ``fetch``, so what they send
  and how the card responds is checked and not just how they are spelled;
* the schedule page marks a queued post that carries a video.

Runs on a throwaway database with fake platform clients: no network, no real
insights.db, no real account.
"""

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _video_gate import check, fail  # noqa: E402

VIDEO = "https://cdn.example.test/clip.mp4"


def inline_scripts(html):
    blocks = []
    for match in re.finditer(r"<script([^>]*)>(.*?)</script>", html, re.S | re.I):
        attrs, body = match.group(1), match.group(2)
        if "src=" in attrs or not body.strip():
            continue
        kind = re.search(r'type="([^"]+)"', attrs)
        if kind and "javascript" not in kind.group(1) and kind.group(1) != "module":
            continue
        blocks.append(body)
    return blocks


def node_check(node, blocks, label):
    work = tempfile.mkdtemp(prefix="video_ui_js_")
    try:
        for index, body in enumerate(blocks):
            path = os.path.join(work, f"block{index}.js")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body)
            result = subprocess.run([node, "--check", path], capture_output=True, text=True)
            check(result.returncode == 0,
                  f"{label} inline script #{index} does not parse:\n{result.stderr.strip()[:600]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# A stub DOM just big enough for the video functions: elements with a class
# set, a dataset, children, and the few methods the functions call.
HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

class El {
  constructor(id) {
    this.id = id; this.classes = new Set(); this.dataset = {}; this.children = [];
    this.value = ''; this.textContent = ''; this.className = ''; this.attrs = {}; this.q = {};
    this.files = [];
    const self = this;
    this.classList = {
      add: (...c) => c.forEach(x => self.classes.add(x)),
      remove: (...c) => c.forEach(x => self.classes.delete(x)),
      toggle: (c, force) => { const on = force === undefined ? !self.classes.has(c) : force;
                              on ? self.classes.add(c) : self.classes.delete(c); return on; },
      contains: (c) => self.classes.has(c),
    };
  }
  querySelector(sel) { return this.q[sel]; }
  appendChild(c) { this.children.push(c); }
  replaceChildren() { this.children = []; }
  removeAttribute(a) { delete this.attrs[a]; this.src = undefined; }
  load() {}
}

function build(hasVideo) {
  const els = {};
  const make = (name, classes = []) => { const e = new El(name); classes.forEach(c => e.classes.add(c)); els[name] = e; return e; };
  const section = make('video-section-4'); section.dataset.videoUrl = hasVideo ? 'https://old.test/a.mp4' : '';
  const preview = make('video-preview-4', hasVideo ? ['d-flex'] : ['d-none']);
  preview.q.video = new El('player'); preview.q.span = new El('label');
  make('video-note-4', hasVideo ? [] : ['d-none']);
  make('video-warnings-4');
  make('video-edit-form-4', ['d-none']);
  make('add-video-btn-4', hasVideo ? ['d-none'] : []);
  make('video-progress-4', ['d-none']);
  make('video-url-input-4'); make('video-file-input-4'); make('video-library-4');
  return els;
}

async function run(scenario) {
  const els = build(scenario.hasVideo);
  const log = { fetch: [], toast: [], confirm: 0 };
  const ctx = {
    console, Promise,
    FormData: class {
      constructor() { this.entries = []; }
      append(k, v) { this.entries.push([k, v]); }
      get(k) { const h = this.entries.find(e => e[0] === k); return h ? h[1] : null; }
    },
    document: {
      getElementById: (id) => els[id] || null,
      createElement: () => new El('li'),
    },
    cardIdsParam: () => '4,5,6',
    // renderPostVideo tells the Instagram people block the card's video changed
    // (a Reel takes tags); that block is covered by check_ig_people.py js.
    igRefreshPeople: () => {},
    showToast: (message, type) => log.toast.push([type, message]),
    confirm: () => { log.confirm++; return scenario.confirm !== false; },
    fetch: async (url, init) => {
      log.fetch.push({ url, method: init.method, body: init.body });
      if (scenario.reject) throw new Error('offline');
      return { ok: scenario.ok !== false, json: async () => scenario.reply };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  scenario.setup && scenario.setup(els, ctx);
  let threw = null;
  try { await scenario.act(ctx, els); } catch (e) { threw = e.message; }
  await new Promise(r => setTimeout(r, 0));
  const f = log.fetch[0];
  return {
    threw, toasts: log.toast, confirms: log.confirm, fetches: log.fetch.length,
    url: f && f.url, method: f && f.method,
    body: f && Object.fromEntries(f.body.entries),
    section: els['video-section-4'].dataset.videoUrl,
    previewHidden: els['video-preview-4'].classes.has('d-none'),
    previewFlex: els['video-preview-4'].classes.has('d-flex'),
    noteHidden: els['video-note-4'].classes.has('d-none'),
    formHidden: els['video-edit-form-4'].classes.has('d-none'),
    addHidden: els['add-video-btn-4'].classes.has('d-none'),
    progressHidden: els['video-progress-4'].classes.has('d-none'),
    playerSrc: els['video-preview-4'].q.video.src,
    label: els['video-preview-4'].q.span.textContent,
    warnings: els['video-warnings-4'].children.map(c => [c.className, c.textContent]),
    fileValue: els['video-file-input-4'].value,
    libraryValue: els['video-library-4'].value,
  };
}

const file = (name, size) => ({ name, size });
const results = {};
(async () => {
  const good = { success: true, video_url: 'https://cdn.test/new.mp4',
    warnings: [{ level: 'error', platform: 'linkedin', message: 'LinkedIn needs 3s' },
               { level: 'warn', platform: 'twitter', message: 'X may reject it' }] };

  results.saveUrl = await run({ hasVideo: false, reply: good,
    act: async (c, e) => { e['video-url-input-4'].value = '  https://cdn.test/new.mp4 '; c.savePostVideoUrl(4); } });
  results.saveEmpty = await run({ hasVideo: false, reply: good,
    act: async (c, e) => { e['video-url-input-4'].value = '   '; c.savePostVideoUrl(4); } });
  results.saveRefused = await run({ hasVideo: false, ok: false, reply: { error: 'URL host resolves to non-public address' },
    act: async (c, e) => { e['video-url-input-4'].value = 'http://10.0.0.1/x.mp4'; c.savePostVideoUrl(4); } });
  results.offline = await run({ hasVideo: false, reject: true, reply: {},
    act: async (c, e) => { e['video-url-input-4'].value = 'https://cdn.test/new.mp4'; c.savePostVideoUrl(4); } });

  results.removeCancelled = await run({ hasVideo: true, confirm: false, reply: { success: true, video_url: null },
    act: async (c) => { c.removePostVideo(4); } });
  results.remove = await run({ hasVideo: true, reply: { success: true, video_url: null, warnings: [] },
    act: async (c) => { c.removePostVideo(4); } });

  results.uploadBadType = await run({ hasVideo: false, reply: good,
    act: async (c, e) => { const i = { files: [file('clip.avi', 1000)], value: 'C:\\clip.avi' }; c.uploadPostVideo(4, i); results._i1 = i.value; } });
  results.uploadTooBig = await run({ hasVideo: false, reply: good,
    act: async (c, e) => { const i = { files: [file('clip.mp4', 101 * 1024 * 1024)], value: 'x' }; c.uploadPostVideo(4, i); results._i2 = i.value; } });
  results.upload = await run({ hasVideo: false, reply: good,
    act: async (c, e) => { const i = { files: [file('Clip.MOV', 5 * 1024 * 1024)], value: 'x' }; c.uploadPostVideo(4, i); results._i3 = i.value; } });
  results.uploadFails = await run({ hasVideo: false, ok: false, reply: { error: 'Video upload needs Cloudinary.' },
    act: async (c, e) => { const i = { files: [file('clip.mp4', 1000)], value: 'x' }; c.uploadPostVideo(4, i); await new Promise(r => setTimeout(r, 5)); results._i4 = i.value; } });

  results.pick = await run({ hasVideo: false, reply: good,
    act: async (c, e) => { const s = e['video-library-4']; s.value = 'https://cdn.test/lib.mp4'; c.pickPostVideoFromLibrary(4, s); } });
  results.pickNothing = await run({ hasVideo: false, reply: good,
    act: async (c, e) => { const s = e['video-library-4']; s.value = ''; c.pickPostVideoFromLibrary(4, s); } });

  results.cancelEmpty = await run({ hasVideo: false, reply: good,
    setup: (e) => { e['add-video-btn-4'].classes.add('d-none'); e['video-edit-form-4'].classes.delete('d-none'); },
    act: async (c) => { c.cancelPostVideo(4); } });
  results.cancelWithVideo = await run({ hasVideo: true, reply: good,
    setup: (e) => { e['video-edit-form-4'].classes.delete('d-none'); },
    act: async (c) => { c.cancelPostVideo(4); } });
  results.edit = await run({ hasVideo: false, reply: good, act: async (c) => { c.editPostVideo(4); } });

  console.log(JSON.stringify(results));
})();
"""


def section_ui():
    from _accounts_gate import connect
    from check_bulk_platform import Rig

    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    rig = Rig()
    db, P, web, client = rig.database, rig.P, rig.web, rig.client

    with_video = rig.card("Card that carries a video", ["linkedin", "instagram"], image_url=None)
    for row_id in with_video.values():
        db.update_standalone_post_video(row_id, VIDEO, db_path=P)
    without = rig.card("Card with no video", ["linkedin"], image_url=None)

    html = client.get("/compose").get_data(as_text=True)
    vid = db.get_standalone_post(with_video["instagram"], db_path=P)["id"]   # the IG row is the card's primary
    plain = without["linkedin"]

    def region(marker_id):
        start = html.index(f'id="video-section-{marker_id}"')
        return html[start: html.index("</div>\n                            <div class=\"edit-form", start)]

    # --- the markup -----------------------------------------------------
    card = region(vid)
    check(f'<video src="{VIDEO}"' in card, "the card with a video must render a player for it")
    check(f'id="video-preview-{vid}"' in card and "d-none" not in re.search(
        rf'<div class="([^"]*)"\s+id="video-preview-{vid}"', card).group(1),
          "the preview of a card with a video must be visible")
    check(f'id="add-video-btn-{vid}"' in card and "d-none" in re.search(
        rf'class="([^"]*)"\s+id="add-video-btn-{vid}"', card).group(1),
          "a card that already has a video should hide its Add Video button")
    check("goes out as a Reel" in card, "an Instagram card should say the video posts as a Reel")
    check('accept="video/mp4,video/quicktime' in card, "the file picker must accept MP4/MOV")

    bare = region(plain)
    check('<video src=""' in bare and "d-none" in re.search(
        rf'<div class="([^"]*)"\s+id="video-preview-{plain}"', bare).group(1),
          "a card with no video must keep its player hidden")
    check("Add Video" in bare and "d-none" not in re.search(
        rf'class="([^"]*)"\s+id="add-video-btn-{plain}"', bare).group(1),
          "a card with no video must offer Add Video")
    check("Reel" not in bare, "a card without Instagram must not mention Reels")

    # --- every control is bound to its handler, not just present ----------------
    for label, call in (
        ("Add Video", f"editPostVideo({vid})"), ("Change video", f"editPostVideo({vid})"),
        ("Remove video", f"removePostVideo({vid})"), ("Cancel", f"cancelPostVideo({vid})"),
        ("Save URL", f"savePostVideoUrl({vid})"), ("Upload", f"uploadPostVideo({vid}, this)"),
        ("Library", f"pickPostVideoFromLibrary({vid}, this)"),
    ):
        check(re.search(rf'on\w+="[^"]*{re.escape(call)}', card),
              f"the {label} control is not wired to {call}")

    # --- every handler the markup calls exists exactly once ----------------
    blocks = inline_scripts(html)
    check(blocks and max(len(b) for b in blocks) > 100_000, "the page's main script is missing")
    declared = collections.Counter(
        name for body in blocks
        for name in re.findall(r"^(?:async\s+)?function\s+(\w+)\s*\(", body, re.M))
    handlers = set(re.findall(r'on(?:click|change|keydown)="[^"]*?\b(\w+Video\w*|pickPostVideoFromLibrary)\(', card + bare))
    handlers |= {"editPostVideo", "cancelPostVideo", "removePostVideo", "savePostVideoUrl", "uploadPostVideo",
                 "pickPostVideoFromLibrary", "renderPostVideo", "sendPostVideo", "loadPostVideoLibrary",
                 "postVideoEls"}
    for name in sorted(handlers):
        check(declared[name] == 1, f"{name}() is declared {declared[name]} times on the page, expected exactly once")
    dupes = {n: c for n, c in declared.items() if c > 1}
    check(not dupes, f"declared more than once, so the later one silently wins: {dupes}")
    node_check(node, blocks, "/compose")

    # --- the functions run ----------------------------------------------
    source = "\n".join(blocks)
    start = source.index("// Video attachment.")
    end = source.index("function savePostEdit(", start)
    work = tempfile.mkdtemp(prefix="video_ui_")
    try:
        fn_path = os.path.join(work, "video_fns.js")
        harness_path = os.path.join(work, "harness.js")
        with open(fn_path, "w", encoding="utf-8") as handle:
            handle.write(source[start:end])
        with open(harness_path, "w", encoding="utf-8") as handle:
            handle.write(HARNESS)
        run = subprocess.run([node, harness_path, fn_path], capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check(run.returncode == 0, f"the harness crashed:\n{run.stderr.strip()[:800]}")
    r = json.loads(run.stdout.strip().splitlines()[-1])

    def ok(name):
        check(r[name]["threw"] is None, f"{name} threw: {r[name]['threw']}")
        return r[name]

    s = ok("saveUrl")
    check(s["fetches"] == 1 and s["url"] == "/compose/post/4/video" and s["method"] == "POST",
          f"saving a URL must POST to the card's video route: {s['url']} {s['method']}")
    check(s["body"] == {"video_url": "https://cdn.test/new.mp4", "post_ids": "4,5,6"},
          f"the request must carry the trimmed URL and every row of the card: {s['body']}")
    check(s["section"] == "https://cdn.test/new.mp4" and s["playerSrc"] == "https://cdn.test/new.mp4"
          and s["label"] == "https://cdn.test/new.mp4", "the card did not show the new video")
    check(not s["previewHidden"] and s["previewFlex"] and not s["noteHidden"] and s["addHidden"] and s["formHidden"],
          f"the card is in the wrong state after attaching: {s}")
    check(s["warnings"] == [["text-danger", "⛔ LinkedIn needs 3s"], ["text-warning", "⚠️ X may reject it"]],
          f"warnings should list the platform problems, errors in red: {s['warnings']}")
    check(s["toasts"] == [["success", "Video attached!"]], f"wrong toast: {s['toasts']}")

    e = ok("saveEmpty")
    check(e["fetches"] == 0 and e["toasts"] and e["toasts"][0][0] == "warning", "an empty URL should not be sent")

    bad = ok("saveRefused")
    check(bad["fetches"] == 1 and bad["section"] == "" and bad["previewHidden"] and not bad["addHidden"]
          and "non-public" in bad["toasts"][0][1] and bad["toasts"][0][0] == "error",
          f"a refused URL must say why and leave the card as it was: {bad}")
    off = ok("offline")
    check(off["section"] == "" and off["toasts"][0][0] == "error" and off["progressHidden"],
          f"a network failure must be reported and leave the card alone: {off}")

    c = ok("removeCancelled")
    check(c["confirms"] == 1 and c["fetches"] == 0 and c["section"] == "https://old.test/a.mp4",
          "cancelling the confirm must remove nothing")
    rm = ok("remove")
    check(rm["fetches"] == 1 and rm["body"] == {"video_url": "", "post_ids": "4,5,6"},
          f"removing must send an empty video for every row of the card: {rm['body']}")
    check(rm["section"] == "" and rm["previewHidden"] and not rm["previewFlex"] and rm["noteHidden"]
          and not rm["addHidden"] and rm["toasts"] == [["success", "Video removed!"]],
          f"the card did not return to having no video: {rm}")

    for name, note in (("uploadBadType", "an AVI"), ("uploadTooBig", "a 101 MB file")):
        u = ok(name)
        check(u["fetches"] == 0 and u["toasts"] and u["toasts"][0][0] == "error",
              f"{note} must be refused in the browser before any upload")
    check(r["_i1"] == "" and r["_i2"] == "", "a refused file must be cleared from the picker")
    up = ok("upload")
    check(up["fetches"] == 1 and up["url"] == "/compose/post/4/video" and up["body"]["post_ids"] == "4,5,6"
          and up["body"]["video"] == {"name": "Clip.MOV", "size": 5 * 1024 * 1024},
          f"an upload must send the file with the whole card's ids: {up['body']}")
    check(up["section"] == "https://cdn.test/new.mp4" and up["toasts"] == [["success", "Video uploaded and attached!"]],
          "the card did not show the uploaded video")
    fail_up = ok("uploadFails")
    check("Cloudinary" in fail_up["toasts"][0][1] and fail_up["section"] == "" and r["_i4"] == "",
          f"a failed upload must say why, leave the card alone and clear the picker: {fail_up}")

    p = ok("pick")
    check(p["body"] == {"video_url": "https://cdn.test/lib.mp4", "post_ids": "4,5,6"} and p["libraryValue"] == "",
          f"picking from the library must attach that video and reset the menu: {p}")
    check(ok("pickNothing")["fetches"] == 0, "the menu's placeholder must not send anything")

    ce = ok("cancelEmpty")
    check(ce["formHidden"] and not ce["addHidden"], "cancelling on a card with no video must bring back Add Video")
    cv = ok("cancelWithVideo")
    check(cv["formHidden"] and cv["addHidden"], "cancelling on a card with a video must keep Add Video hidden")
    ed = ok("edit")
    check(not ed["formHidden"] and ed["addHidden"], "Add Video must open the form and hide itself")

    # --- the schedule page marks a video post ---------------------------
    queued = rig.card("Queued with a video", ["linkedin"], image_url=None)
    db.update_standalone_post_video(queued["linkedin"], VIDEO, db_path=P)
    queued_plain = rig.card("Queued without a video", ["linkedin"], image_url=None)
    for row_id in (queued["linkedin"], queued_plain["linkedin"]):
        db.add_scheduled_post(social_post_id=None, article_id=None, standalone_post_id=row_id,
                              post_type="standalone", platform="linkedin",
                              scheduled_for="2099-01-01T09:00:00", status="pending", db_path=P)
    page = client.get("/schedule").get_data(as_text=True)
    markup = re.sub(r"<script.*?</script>", "", page, flags=re.S | re.I)
    cells = re.findall(r'<td class="description-cell"[^>]*>(.*?)</td>', markup, re.S)
    marked = {("Queued with a video" in cell, "video-badge" in cell) for cell in cells}
    check(len(cells) == 2 and marked == {(True, True), (False, False)},
          f"only the queued post with a video should be marked, and it should be marked: {len(cells)} rows, {marked}")
    listing = client.get("/schedule/list-json").get_json()
    rows = {p["standalone_content"]: p for p in listing["posts"]}
    check(rows["Queued with a video"]["standalone_video_url"] == VIDEO
          and not rows["Queued without a video"]["standalone_video_url"],
          "the schedule feed must carry each entry's video")
    schedule_blocks = inline_scripts(page)
    check("video-badge" in "\n".join(schedule_blocks), "the schedule page's own renderer does not mark videos")
    node_check(node, schedule_blocks, "/schedule")

    print("the card shows, changes and clears its video across all its rows; the functions run; the schedule marks video posts")
    print("VIDEO_UI_OK")


# ---------------------------------------------------------------------------
# The "Write a New Post" card (``check_video.py newpost``)
# ---------------------------------------------------------------------------

NEWPOST_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

class El {
  // Like a real <input type=file>: assigning '' to value empties files too.
  get value() { return this._value; }
  set value(v) { this._value = v; if (v === '') this.files = []; }
  constructor(id) {
    this.id = id; this.classes = new Set(); this._value = ''; this.textContent = '';
    this.files = []; this.innerHTML = ''; this.disabled = false; this.dataset = {};
    const self = this;
    this.classList = {
      add: (...c) => c.forEach(x => self.classes.add(x)),
      remove: (...c) => c.forEach(x => self.classes.delete(x)),
      toggle: (c, force) => { const on = force === undefined ? !self.classes.has(c) : force;
                              on ? self.classes.add(c) : self.classes.delete(c); return on; },
      contains: (c) => self.classes.has(c),
    };
  }
  focus() {}
}

function build() {
  const els = {};
  ['new-post-video-file', 'new-post-video-url', 'new-post-video-name', 'new-post-video-clear',
   'new-post-video-note', 'add-post-textarea', 'add-post-charcount', 'add-post-btn', 'add-post-form']
    .forEach(id => { els[id] = new El(id); });
  els['new-post-video-clear'].classes.add('d-none');
  els['new-post-video-note'].classes.add('d-none');
  const save = new El('save'); save.innerHTML = '💾 Save Post';
  els['add-post-textarea'].closest = () => ({ querySelector: () => save });
  els.save = save;
  return els;
}

async function run(scenario) {
  const els = build();
  const log = { fetch: [], toast: [], alert: [], reload: 0 };
  const chips = [{ dataset: { platform: 'linkedin', accountId: '7' } }, { dataset: { platform: 'threads', accountId: '' } }];
  const ctx = {
    console, Promise,
    FormData: class {
      constructor() { this.entries = []; }
      append(k, v) { this.entries.push([k, v]); }
      getAll(k) { return this.entries.filter(e => e[0] === k).map(e => e[1]); }
    },
    document: {
      getElementById: (id) => els[id] || null,
      querySelectorAll: () => chips,
    },
    platformCharLimits: { linkedin: 3000, threads: 500 },
    PLATFORM_LABELS: { linkedin: 'LinkedIn', threads: 'Threads' },
    showToast: (m, t) => log.toast.push([t, m]),
    alert: (m) => log.alert.push(m),
    location: { reload: () => { log.reload++; } },
    fetch: async (url, init) => {
      log.fetch.push({ url, method: init.method, body: init.body });
      if (scenario.reject) throw new Error('offline');
      return { json: async () => scenario.reply };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  scenario.setup && scenario.setup(els);
  let threw = null;
  try { await scenario.act(ctx, els); } catch (e) { threw = e.message; }
  await new Promise(r => setTimeout(r, 5));
  const f = log.fetch[0];
  const entries = f ? f.body.entries.map(e => [e[0], (e[1] && e[1].name) ? { file: e[1].name } : e[1]]) : null;
  return {
    threw, toasts: log.toast, alerts: log.alert, reloads: log.reload,
    fetches: log.fetch.length, url: f && f.url, method: f && f.method, entries,
    fileValue: els['new-post-video-file'].value, urlValue: els['new-post-video-url'].value,
    name: els['new-post-video-name'].textContent,
    clearHidden: els['new-post-video-clear'].classes.has('d-none'),
    noteHidden: els['new-post-video-note'].classes.has('d-none'),
    saveText: els.save.innerHTML, saveDisabled: els.save.disabled,
  };
}

const file = (name, size) => ({ name, size });
const results = {};
(async () => {
  const withFile = (els, f) => { els['new-post-video-file'].files = [f]; els['new-post-video-file'].value = 'C:\\fake\\' + f.name; };
  const ready = (els) => { els['add-post-textarea'].value = 'Words for the post'; };

  results.badType = await run({ act: async (c, e) => { withFile(e, file('clip.avi', 1000)); c.onNewPostVideoFile(e['new-post-video-file']); } });
  results.tooBig = await run({ act: async (c, e) => { withFile(e, file('clip.mp4', 101 * 1024 * 1024)); c.onNewPostVideoFile(e['new-post-video-file']); } });
  results.goodFile = await run({
    setup: (e) => { e['new-post-video-url'].value = 'https://old.test/x.mp4'; },
    act: async (c, e) => { withFile(e, file('clip.mp4', 5 * 1024 * 1024)); c.onNewPostVideoFile(e['new-post-video-file']); } });
  results.typeUrl = await run({
    setup: (e) => { withFile(e, file('clip.mp4', 1000)); },
    act: async (c, e) => { e['new-post-video-url'].value = 'https://cdn.test/y.mp4'; c.onNewPostVideoUrl(e['new-post-video-url']); } });
  results.emptyUrl = await run({
    act: async (c, e) => { e['new-post-video-url'].value = '   '; c.onNewPostVideoUrl(e['new-post-video-url']); } });
  results.cleared = await run({
    setup: (e) => { withFile(e, file('clip.mp4', 1000)); e['new-post-video-url'].value = ''; },
    act: async (c, e) => { c.clearNewPostVideo(); } });
  results.reopened = await run({
    setup: (e) => { e['new-post-video-url'].value = 'https://left.over/z.mp4'; },
    act: async (c, e) => { c.showAddPostForm(); } });

  const ok = { success: true, post_ids: [1, 2], video_warnings: [] };
  results.saveFile = await run({ reply: { ...ok, video_warnings: [{ level: 'error', message: 'LinkedIn needs 3 seconds' }] },
    setup: (e) => { ready(e); withFile(e, file('clip.mp4', 1234)); },
    act: async (c) => { c.createManualPost(); } });
  results.saveUrl = await run({ reply: ok,
    setup: (e) => { ready(e); e['new-post-video-url'].value = ' https://cdn.test/y.mp4 '; },
    act: async (c) => { c.createManualPost(); } });
  results.saveNone = await run({ reply: ok, setup: (e) => { ready(e); }, act: async (c) => { c.createManualPost(); } });
  results.saveFails = await run({ reply: { error: 'URL host resolves to non-public address' },
    setup: (e) => { ready(e); e['new-post-video-url'].value = 'http://10.0.0.1/x.mp4'; },
    act: async (c) => { c.createManualPost(); } });
  results.saveOffline = await run({ reject: true, reply: {},
    setup: (e) => { ready(e); withFile(e, file('clip.mp4', 1234)); }, act: async (c) => { c.createManualPost(); } });
  results.saveNoText = await run({ reply: ok, setup: (e) => { withFile(e, file('clip.mp4', 1234)); }, act: async (c) => { c.createManualPost(); } });

  console.log(JSON.stringify(results));
})();
"""


def section_newpost():
    import io
    from check_bulk_platform import Rig
    from _video_gate import MB, make_sample_mp4

    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    rig = Rig()
    db, P, web, client = rig.database, rig.P, rig.web, rig.client
    web.video_media.head_size = lambda url: 5 * MB
    link_image_calls = []
    web._maybe_attach_link_image = lambda *a, **k: link_image_calls.append(a)
    uploads = []

    def fake_upload(data, **kwargs):
        uploads.append({"bytes": len(data), **kwargs})
        return {"secure_url": "https://res.cloudinary.test/insights/np1.mp4",
                "public_id": "insights/np1", "bytes": len(data)}

    web.cloudinary.uploader.upload = fake_upload
    web.CLOUDINARY_CONFIGURED = True
    sample = make_sample_mp4(rig.directory, seconds=2)
    with open(sample, "rb") as fh:
        clip = fh.read()

    def rows():
        return [dict(r) for r in db.list_standalone_posts(db_path=P)]

    def create(content="A brand new post", targets=("linkedin", "threads"), **fields):
        data = {"content": content, "targets": list(targets), **fields}
        response = client.post("/compose/post/create", data=data, content_type="multipart/form-data")
        return response.status_code, (response.get_json(silent=True) or {})

    def cards():
        return web._group_standalone_posts(db.list_standalone_posts(db_path=P))

    # --- a video by URL lands on every ticked account's post ---------------
    url = "http://93.184.216.34/new-post.mp4"
    status, res = create(video_url=url)
    check(status == 200 and res.get("success"), f"create with a video URL failed: {status} {res}")
    made = [r for r in rows() if r["content"] == "A brand new post"]
    check(sorted(r["platform"] for r in made) == ["linkedin", "threads"]
          and all(r["video_url"] == url for r in made), f"every ticked account's post should carry the video: {made}")
    check(res["post"]["video_url"] == url and res["video_warnings"] == [], f"reply wrong: {res}")
    check(len(cards()) == 1 and len(cards()[0]["platforms"]) == 2, "the new video post should be one card")
    check(not link_image_calls, "a post with a video must not go fetching a link image it will never use")

    # control: a post with no video is unchanged, and does go looking for a link image
    status, res = create(content="Plain new post")
    plain = [r for r in rows() if r["content"] == "Plain new post"]
    check(status == 200 and len(plain) == 2 and all(r["video_url"] is None for r in plain),
          f"a post without a video changed: {status} {plain}")
    check(len(link_image_calls) == 1, "a post without a video should still look for a link image")

    # --- a video by file ------------------------------------------------
    status, res = create(content="Uploaded video post", video=(io.BytesIO(clip), "clip.mp4"))
    check(status == 200 and res.get("success"), f"create with a video file failed: {status} {res}")
    check(len(uploads) == 1 and uploads[0]["resource_type"] == "video" and uploads[0]["bytes"] == len(clip),
          f"the file should be uploaded once as a video: {uploads}")
    made = [r for r in rows() if r["content"] == "Uploaded video post"]
    check(len(made) == 2 and all(r["video_url"] == "https://res.cloudinary.test/insights/np1.mp4" for r in made),
          "the uploaded video did not reach every ticked account's post")
    short = sorted(w["platform"] for w in res["video_warnings"] if "at least 3 seconds" in w["message"])
    check(short == ["instagram", "linkedin"], f"a 2s clip should warn LinkedIn and Instagram: {res['video_warnings']}")

    # --- a video that can't be used leaves no post behind ---------------------
    before = len(rows())
    uploaded = len(uploads)
    for label, fields in (
        ("a private-network URL", {"video_url": "http://10.0.0.5/x.mp4"}),
        ("a non-http URL", {"video_url": "ftp://93.184.216.34/x.mp4"}),
        ("a script URL", {"video_url": "javascript:alert(1)"}),
        ("an AVI file", {"video": (io.BytesIO(clip), "clip.avi")}),
        ("an empty file", {"video": (io.BytesIO(b""), "empty.mp4")}),
    ):
        status, res = create(content=f"Should not exist {label}", **fields)
        check(status == 400 and res.get("error") and not res.get("success"),
              f"{label} should be refused: {status} {res}")
        check(len(rows()) == before, f"{label} was refused but a post was still created")
    check(len(uploads) == uploaded, "a refused video must not have been uploaded")

    web.cloudinary.uploader.upload = lambda data, **kw: (_ for _ in ()).throw(RuntimeError("cloudinary is down"))
    status, res = create(content="Upload will fail", video=(io.BytesIO(clip), "clip.mp4"))
    check(status == 400 and "cloudinary is down" in res["error"] and len(rows()) == before,
          f"a failed upload must say why and create nothing: {status} {res}")
    web.cloudinary.uploader.upload = fake_upload
    web.CLOUDINARY_CONFIGURED = False
    status, res = create(content="No cloudinary", video=(io.BytesIO(clip), "clip.mp4"))
    check(status == 400 and "Cloudinary" in res["error"] and len(rows()) == before,
          f"without Cloudinary a file should be refused cleanly: {status} {res}")
    web.CLOUDINARY_CONFIGURED = True

    # a request that was going to fail anyway must not spend an upload first
    uploaded = len(uploads)
    status, _ = create(content="", video=(io.BytesIO(clip), "clip.mp4"))
    check(status == 400 and len(uploads) == uploaded, "an empty post must be refused before its video is uploaded")
    status, _ = create(content="No targets", targets=(), video=(io.BytesIO(clip), "clip.mp4"))
    check(status == 400 and len(uploads) == uploaded, "a post with no accounts must be refused before its video is uploaded")

    # --- the card and its markup ---------------------------------------------
    html = client.get("/compose").get_data(as_text=True)
    composer = html[html.index('id="new-post-video"'): html.index('id="add-post-charcount"')]
    for needle in ('id="new-post-video-file"', 'accept="video/mp4,video/quicktime',
                   'onchange="onNewPostVideoFile(this)"', 'id="new-post-video-url"',
                   'oninput="onNewPostVideoUrl(this)"', 'onclick="clearNewPostVideo()"',
                   'id="new-post-video-note"', "Reel"):
        check(needle in composer, f"the new post card is missing {needle}")
    blocks = inline_scripts(html)
    declared = collections.Counter(
        name for body in blocks for name in re.findall(r"^(?:async\s+)?function\s+(\w+)\s*\(", body, re.M))
    for name in ("onNewPostVideoFile", "onNewPostVideoUrl", "clearNewPostVideo", "refreshNewPostVideoUI",
                 "createManualPost", "showAddPostForm"):
        check(declared[name] == 1, f"{name}() is declared {declared[name]} times, expected exactly once")
    check(not {n: c for n, c in declared.items() if c > 1}, "a function on the page is declared twice")
    node_check(node, blocks, "/compose")

    # --- the composer's functions run --------------------------------------------
    source = "\n".join(blocks)
    start = source.index("// ============ Writing a new post ============")
    end = source.index("/* ── Import File handling", start)
    work = tempfile.mkdtemp(prefix="video_newpost_")
    try:
        fn_path = os.path.join(work, "newpost.js")
        harness_path = os.path.join(work, "harness.js")
        with open(fn_path, "w", encoding="utf-8") as handle:
            handle.write(source[start:end])
        with open(harness_path, "w", encoding="utf-8") as handle:
            handle.write(NEWPOST_HARNESS)
        run = subprocess.run([node, harness_path, fn_path], capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check(run.returncode == 0, f"the harness crashed:\n{run.stderr.strip()[:800]}")
    r = json.loads(run.stdout.strip().splitlines()[-1])

    def ok(name):
        check(r[name]["threw"] is None, f"{name} threw: {r[name]['threw']}")
        return r[name]

    for name, what in (("badType", "an AVI"), ("tooBig", "a 101 MB file")):
        v = ok(name)
        check(v["toasts"] and v["toasts"][0][0] == "error" and v["fileValue"] == "" and v["clearHidden"],
              f"{what} must be refused and cleared from the picker: {v}")
    g = ok("goodFile")
    check(g["urlValue"] == "" and g["name"] == "clip.mp4 (5.0 MB)" and not g["clearHidden"] and not g["noteHidden"],
          f"choosing a file must clear the URL, show the file and offer removal: {g}")
    t = ok("typeUrl")
    check(t["fileValue"] == "" and not t["clearHidden"] and not t["noteHidden"] and t["name"] == "",
          f"typing a URL must clear the file: {t}")
    e = ok("emptyUrl")
    check(e["clearHidden"] and e["noteHidden"], "a blank URL is not a video")
    c = ok("cleared")
    check(c["fileValue"] == "" and c["urlValue"] == "" and c["name"] == "" and c["clearHidden"] and c["noteHidden"],
          f"clearing must reset the whole control: {c}")
    o = ok("reopened")
    check(o["urlValue"] == "" and o["clearHidden"], "opening the composer must not keep the last post's video")

    sf = ok("saveFile")
    keys = [k for k, _ in sf["entries"]]
    check(sf["url"] == "/compose/post/create" and sf["method"] == "POST", f"wrong request: {sf['url']} {sf['method']}")
    check(keys.count("video") == 1 and "video_url" not in keys and ["video", {"file": "clip.mp4"}] in sf["entries"],
          f"a chosen file must be sent as the video: {sf['entries']}")
    check(["targets", "linkedin:7"] in sf["entries"] and ["targets", "threads"] in sf["entries"]
          and ["content", "Words for the post"] in sf["entries"], f"the post itself must still be sent: {sf['entries']}")
    check(sf["saveText"] == "<span class=\"spinner-border spinner-border-sm\"></span> Uploading video..."
          and sf["reloads"] == 1, f"saving a file should say it is uploading, then reload: {sf}")
    check(len(sf["alerts"]) == 1 and "LinkedIn needs 3 seconds" in sf["alerts"][0] and "⛔" in sf["alerts"][0],
          f"the platform warnings must be shown before the reload wipes them: {sf['alerts']}")

    su = ok("saveUrl")
    check(["video_url", "https://cdn.test/y.mp4"] in su["entries"] and "video" not in [k for k, _ in su["entries"]]
          and su["reloads"] == 1 and su["alerts"] == [], f"a pasted URL must be sent trimmed as video_url: {su['entries']}")
    sn = ok("saveNone")
    check(not {"video", "video_url"} & {k for k, _ in sn["entries"]} and sn["reloads"] == 1
          and "Uploading" not in sn["saveText"], f"a post with no video must send none: {sn['entries']}")
    sx = ok("saveFails")
    check(sx["reloads"] == 0 and sx["alerts"] == ["Error: URL host resolves to non-public address"]
          and not sx["saveDisabled"] and "Save Post" in sx["saveText"],
          f"a refused video must be reported and leave the form usable: {sx}")
    so = ok("saveOffline")
    check(so["reloads"] == 0 and so["alerts"] == ["Error: offline"] and not so["saveDisabled"],
          f"a network failure must be reported and leave the form usable: {so}")
    check(ok("saveNoText")["fetches"] == 0, "a post with no text must not be sent, video or not")

    print("a new post takes a video by file or URL, refuses an unusable one before creating anything, and the card's functions run")
    print("VIDEO_NEWPOST_OK")
