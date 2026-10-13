"""Gates for queueing from a filtered Compose list.

    python scripts/check_queue_filter.py route    QUEUE_FILTER_ROUTE_OK
    python scripts/check_queue_filter.py bulk     QUEUE_FILTER_BULK_OK
    python scripts/check_queue_filter.py recheck  QUEUE_FILTER_RECHECK_OK

The bug: under "Unused" + "Not Queued", ticking posts and pressing Add to Queue
left every card on the list. Two causes. The page's own list of queueable
platforms still said LinkedIn and Threads, so X (and Facebook and Instagram)
were never queued, and a card queued on Threads but not X still belongs under
"Not Queued". And nothing asked whether a card the page had just changed still
matched the filters, so even a fully queued card stayed until a reload.

``route`` proves the card route answers "does this card still match" with the
list's own predicate, by comparing it with what /compose/posts/more lists.
``bulk`` runs the page's Add to Queue under node: every ticked platform the
server can schedule is sent, an unconnected one is skipped and named rather than
blocking the rest, the server's reasons are shown, and every card is re-checked.
``recheck`` runs the re-check and the counting under node.

All of it runs on a throwaway database with fake platform clients.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import check  # noqa: E402
from check_accounts_js import inline_scripts, node_path  # noqa: E402
from check_bulk_platform import Rig  # noqa: E402


# ---------------------------------------------------------------------------
# route: the card route's `matches` is the list's predicate
# ---------------------------------------------------------------------------
def section_route():
    rig = Rig()
    web, db, P = rig.web, rig.database, rig.P

    # The screenshot's card: Threads and X, unused, nothing queued.
    ids = rig.card("Most testing adds payloads.", ["threads", "twitter"])
    th, tw = ids["threads"], ids["twitter"]
    other = rig.card("The quiet part of the CVE.", ["threads", "twitter"])

    def card(filters, post_ids=(th, tw)):
        data = {"post_ids": ",".join(map(str, post_ids))}
        if filters is not None:
            data["filters"] = filters if isinstance(filters, str) else json.dumps(filters)
        response = rig.client.post(f"/compose/post/{post_ids[0]}/card", data=data)
        check(response.status_code == 200, f"the card route returned {response.status_code}")
        return response.get_json()

    def listed(filters):
        query = "&".join(f"{k}={v}" for k, v in filters.items() if v)
        body = rig.client.get(f"/compose/posts/more?offset=0&limit=100&{query}").get_json()
        return f'data-post-id="{th}"' in body["html"]

    def agree(filters, expected, why):
        reply = card(filters)
        check(reply.get("matches") is expected, f"{why}: the card route said {reply.get('matches')!r}")
        check(listed(filters) is expected,
              f"{why}: the list disagrees with the card route ({listed(filters)} vs {expected})")
        check('class="post-item' in reply["html"], f"{why}: the card was not redrawn")

    view = {"used": "unused", "queued": "not-queued"}
    agree(view, True, "nothing queued yet, so the card is in the list")

    rig.client.post(f"/compose/post/{th}/queue", data={"platform": "threads"})
    agree(view, True, "queued on Threads alone, X still waits, so it stays under Not Queued")

    rig.client.post(f"/compose/post/{tw}/queue", data={"platform": "twitter"})
    agree(view, False, "queued on both, so it leaves Not Queued")
    agree({"queued": "queued"}, True, "and it now belongs under Queued")

    agree({"used": "unused"}, True, "not used yet")
    rig.client.post("/compose/posts/bulk-toggle-used", json={"post_ids": [th, tw], "used": True})
    agree({"used": "unused"}, False, "marked used, so it leaves Unused")

    agree({"platform": "linkedin"}, False, "a card with no LinkedIn row is not in a LinkedIn list")
    agree({"platform": "twitter"}, True, "it is in the X list")

    # Without filters the reply is what it always was, so older callers are unchanged.
    plain = card(None)
    check("matches" not in plain and plain["success"] and plain["html"], f"no filters, no verdict: {plain.keys()}")
    for bad in ("not json", "[1, 2]", "null"):
        reply = card(bad)
        check("matches" not in reply and reply["success"], f"unreadable filters {bad!r} must be ignored: {reply.keys()}")

    # The other card was never touched and still matches the original view.
    reply = card(view, (other["threads"], other["twitter"]))
    check(reply["matches"] is True, "an untouched card must still match")

    # The page's queueable platforms are the server's, so none is left out again.
    html = rig.client.get("/compose").get_data(as_text=True)
    found = re.search(r"^const SCOPE_QUEUEABLE_PLATFORMS = (.*);$", html, re.M)
    check(found and json.loads(found.group(1)) == web.SCHEDULABLE_PLATFORMS,
          f"the page's queueable platforms are not the server's: {found and found.group(1)}")

    print("route: the card route's verdict matches what the filtered list shows, through queueing on one "
          "platform then both, marking used, and a platform filter; no filters means no verdict")
    print("QUEUE_FILTER_ROUTE_OK")


# ---------------------------------------------------------------------------
# bulk + recheck: the page's functions, under node
# ---------------------------------------------------------------------------
def page_source():
    rig = Rig()
    rig.card("Script gate card", ["threads", "twitter"])
    html = rig.client.get("/compose").get_data(as_text=True)
    return rig.web, "\n".join(inline_scripts(html))


def lift(source, names):
    pieces = []
    for name in names:
        found = re.findall(rf"^(?:async\s+)?function {name}\(", source, re.M)
        check(len(found) == 1, f"{name} is declared {len(found)} times on the page, expected once")
        match = re.search(rf"^(?:async\s+)?function {name}\(.*?^\}}\n", source, re.M | re.S)
        check(match, f"could not lift {name}")
        pieces.append(match.group(0))
    return "\n".join(pieces)


def lift_const(source, name):
    match = re.search(rf"^const {name} = .*?;\n", source, re.M | re.S)
    check(match, f"could not lift {name}")
    return match.group(0)


def platform_tables(source):
    match = re.search(r"^const PLATFORM_META = .*?^\}\);\n", source, re.M | re.S)
    check(match, "could not lift the PLATFORM_META tables")
    return match.group(0)


def run_node(harness, functions):
    node = node_path()
    work = tempfile.mkdtemp(prefix="queue_filter_js_")
    try:
        paths = {}
        for name, body in (("functions.js", functions), ("harness.js", harness)):
            paths[name] = os.path.join(work, name)
            with open(paths[name], "w", encoding="utf-8") as handle:
                handle.write(body)
        run = subprocess.run([node, paths["harness.js"], paths["functions.js"]],
                             capture_output=True, text=True, timeout=120)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check(run.returncode == 0, f"the harness crashed:\n{run.stderr.strip()[:1500]}")
    return json.loads(run.stdout.strip().splitlines()[-1])


BULK_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

function chip(platform, id, queued) {
  const queue = { dataset: { queued: queued ? 'true' : 'false' }, platform, id };
  return { dataset: { platform, platformPostId: String(id) },
           querySelector: (sel) => sel === '.chip-queue' ? queue : null, queue };
}

function world(opts) {
  const w = { fetches: [], toasts: [], alerts: [], chipUI: [], rechecked: [], toggled: 0 };
  // Two cards like the screenshot (Threads + X), one also on Instagram and
  // Facebook; the second card's Threads is already queued.
  const cards = [
    { postItem: { n: 1 }, chips: [chip('threads', 11, false), chip('twitter', 12, false),
                                  chip('instagram', 13, false), chip('facebook', 14, false)] },
    { postItem: { n: 2 }, chips: [chip('threads', 21, true), chip('twitter', 22, false)] },
  ];
  const button = { innerHTML: 'Add to Queue', disabled: false };
  const ctx = {
    console, Promise, JSON, Set, Object, Array, String, parseInt,
    selectAllScope: null,
    linkedinConnected: true, threadsConnected: true, twitterConnected: true,
    instagramConnected: true, facebookConnected: false,
    selectedCards: () => cards.map(c => ({ postItem: c.postItem, chips: c.chips })),
    selectedChips: (card) => card.chips,
    setChipQueuedUI: (btn, queued) => w.chipUI.push([btn.id, queued]),
    recheckCardFilters: async (postItem) => { w.rechecked.push(postItem.n); },
    toggleSelectMode: () => { w.toggled++; },
    showToast: (m, t) => w.toasts.push([t, m]),
    alert: (m) => w.alerts.push(m),
    document: { getElementById: (id) => id === 'queue-selected-btn' ? button : null },
    FormData: class {
      constructor() { this.entries = []; }
      append(k, v) { this.entries.push([k, String(v)]); }
      get(k) { const e = this.entries.find(x => x[0] === k); return e ? e[1] : null; }
    },
    fetch: async (url, init) => {
      const platform = init.body.get('platform');
      w.fetches.push([url, platform]);
      if (platform === 'instagram') {
        return { ok: false, json: async () => ({ error: 'Instagram needs an image or a video' }) };
      }
      return { ok: true, json: async () => ({ success: true, scheduled_for: '2026-10-14T09:00' }) };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  w.ctx = ctx;
  w.button = button;
  return w;
}

(async () => {
  const w = world();
  await vm.runInContext('queueSelectedPosts()', w.ctx);
  console.log(JSON.stringify({
    fetches: w.fetches, toasts: w.toasts, alerts: w.alerts, chipUI: w.chipUI,
    rechecked: w.rechecked, toggled: w.toggled, button: w.button,
  }));
})().catch(e => { console.error(e.stack); process.exit(1); });
"""


def section_bulk():
    web, source = page_source()
    functions = "\n".join([
        platform_tables(source),
        lift_const(source, "SCOPE_QUEUEABLE_PLATFORMS"),
        lift(source, ["queueSelectedPosts", "platformConnected", "platformList"]),
    ])
    r = run_node(BULK_HARNESS, functions)

    sent = [platform for _, platform in r["fetches"]]
    check(sorted(sent) == ["instagram", "threads", "twitter", "twitter"],
          f"every unqueued, connected platform the server can schedule must be sent, X included: {r['fetches']}")
    check(all(url == f"/compose/post/{pid}/queue" for (url, _), pid in zip(r["fetches"], (11, 12, 13, 22))),
          f"each platform is queued through its own row: {r['fetches']}")
    check(sorted(r["chipUI"]) == [[11, True], [12, True], [22, True]],
          f"only the platforms that went in are drawn as queued: {r['chipUI']}")
    check(sorted(r["rechecked"]) == [1, 2],
          f"every selected card must be re-checked against the filters: {r['rechecked']}")
    check(len(r["alerts"]) == 1 and "Instagram: Instagram needs an image or a video" in r["alerts"][0]
          and "Not connected, so not queued: Facebook" in r["alerts"][0],
          f"the server's reason and the skipped platform must be shown: {r['alerts']}")
    check(r["toasts"] == [["warning", "Added 3, failed 1"]], f"the tally is reported: {r['toasts']}")
    check(r["toggled"] == 1 and not r["button"]["disabled"] and r["button"]["innerHTML"] == "Add to Queue",
          f"select mode ends and the button comes back: {r}")

    print("bulk: Add to Queue sends X, Threads and Instagram, skips and names unconnected Facebook instead of "
          "blocking, shows the server's reason, and re-checks every card")
    print("QUEUE_FILTER_BULK_OK")


RECHECK_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

function world(filters) {
  const w = { refreshed: [] };
  const values = Object.assign({ 'filter-platform': '', 'filter-used': '', 'filter-queued': '',
    'filter-image': '', 'filter-source': '', 'filter-brief': '', 'filter-sort': '' }, filters);
  const badge = { textContent: '13 of 2045 posts' };
  const results = { textContent: 'Showing 13 of 2045' };
  const more = { dataset: { offset: '13' }, disabled: false, innerHTML: '' };
  const all = { disabled: false, innerHTML: '' };
  const wrap = { dataset: { total: '40' }, querySelector: (sel) => sel === '.load-more-btn' ? more : all };
  const ctx = {
    console, Promise, JSON, Math, parseInt, Object,
    refreshCard: async (card) => { w.refreshed.push(card.n); },
    autoLoadEnabled: () => false,
    document: {
      getElementById: (id) => id in values ? { value: values[id] }
        : id === 'posts-count' ? badge : id === 'filter-results-count' ? results : null,
      querySelector: (sel) => sel === '#saved-posts-container .load-more-container' ? wrap : null,
    },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  Object.assign(w, { ctx, badge, results, more, all, wrap });
  return w;
}

(async () => {
  const out = {};
  let w = world({});
  await w.ctx.recheckCardFilters({ n: 1 });
  await w.ctx.recheckCardFilters(null);
  out.noFilter = w.refreshed;

  w = world({ 'filter-sort': 'oldest' });
  await w.ctx.recheckCardFilters({ n: 2 });
  out.sortOnly = w.refreshed;

  w = world({ 'filter-queued': 'not-queued', 'filter-used': 'unused' });
  await w.ctx.recheckCardFilters({ n: 3 });
  out.filtered = w.refreshed;

  // A card that stopped matching: one fewer shown, the saved total unchanged,
  // and Load more starts one earlier.
  w = world({ 'filter-queued': 'not-queued' });
  w.ctx.noteCardFilteredOut();
  // (A real dataset holds strings; this stub keeps whatever it is given.)
  out.filteredOut = { badge: w.badge.textContent, results: w.results.textContent,
                      offset: String(w.more.dataset.offset), total: String(w.wrap.dataset.total),
                      more: w.more.innerHTML };

  // A deleted card: both numbers drop.
  w = world({ 'filter-queued': 'not-queued' });
  w.ctx.noteCardDeleted(true);
  out.deleted = { badge: w.badge.textContent, results: w.results.textContent,
                  offset: String(w.more.dataset.offset), total: String(w.wrap.dataset.total) };
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e.stack); process.exit(1); });
"""


def section_recheck():
    _, source = page_source()
    functions = "\n".join([
        lift_const(source, "NARROWING_FILTER_KEYS"),
        lift(source, ["recheckCardFilters", "anyFilterActive", "currentPostFilters",
                      "noteCardFilteredOut", "noteCardDeleted", "noteCardLeftList",
                      "adjustPostCountBadge", "loadMoreControls", "setLoadMoreLabels"]),
    ])
    r = run_node(RECHECK_HARNESS, functions)

    check(r["noFilter"] == [] and r["sortOnly"] == [],
          f"with no narrowing filter every card belongs, so nothing is redrawn: {r}")
    check(r["filtered"] == [3], f"under a filter the card is redrawn and judged by the server: {r}")

    out = r["filteredOut"]
    check(out["badge"] == "12 of 2045 posts" and out["results"] == "Showing 12 of 2045",
          f"a card that stops matching takes one off the matching count and none off the saved total: {out}")
    check(out["offset"] == "12" and out["total"] == "39" and out["more"] == "Load more (12 of 39) ▼",
          f"and Load more starts one earlier so nothing is skipped: {out}")

    gone = r["deleted"]
    check(gone["badge"] == "12 of 2044 posts" and gone["results"] == "Showing 12 of 2044"
          and gone["offset"] == "12" and gone["total"] == "39",
          f"a deleted card takes one off both numbers: {gone}")

    print("recheck: only a narrowing filter triggers a redraw; a card filtered out drops the matching count, "
          "not the saved total, and moves Load more back one")
    print("QUEUE_FILTER_RECHECK_OK")


SECTIONS = {
    "route": section_route,
    "bulk": section_bulk,
    "recheck": section_recheck,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_queue_filter.py {'|'.join(SECTIONS)}")
        sys.exit(2)
    SECTIONS[name]()
