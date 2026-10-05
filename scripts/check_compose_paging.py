"""Gates for loading the Compose saved-posts list: Load more, Load all, scrolling.

    python scripts/check_compose_paging.py route     PAGING_ROUTE_OK
    python scripts/check_compose_paging.py wiring    PAGING_WIRING_OK
    python scripts/check_compose_paging.py loadall   PAGING_LOADALL_OK
    python scripts/check_compose_paging.py scroll    PAGING_SCROLL_OK

The list shows 20 cards and a "Load more" button. These gates prove the other two
ways to get the rest: a "Load all" button that loads every post matching the
current filters, and loading the next page when the list is scrolled to its end.

The browser code is run, not read. The page's own functions are lifted out and run
under node against a fake of the DOM, a fake of the server that pages a list of
cards exactly as /compose/posts/more does, and an IntersectionObserver that can
report "the row is in view". What is asserted is what the page requests, what it
puts on screen, and what it does when a request fails, when the filters change
under a request that is in flight, and when the user clicks twice.

Everything runs on a throwaway database with fake platform clients.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import (  # noqa: E402
    isolated_app, install_fake_clients, connect, check,
)
from check_accounts_js import inline_scripts, node_path  # noqa: E402


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------

def seeded_app(threads=120, linkedin=40):
    """A Compose page with ``threads`` Threads-only cards and ``linkedin`` LinkedIn-only."""
    directory, database, web, publisher, client = isolated_app()
    install_fake_clients(publisher, web)
    web._maybe_attach_link_image = lambda *a, **k: None
    connect(database, "linkedin", "li-work", "Work")
    for i in range(linkedin):
        database.add_standalone_post("manual", "gate", "linkedin", f"LinkedIn card {i:03d}",
                                     db_path=database.DB_PATH)
    for i in range(threads):
        database.add_standalone_post("manual", "gate", "threads", f"Threads card {i:03d}",
                                     db_path=database.DB_PATH)
    return database, web, client


def card_ids(html):
    return [int(i) for i in re.findall(r'class="post-item" data-post-id="(\d+)"', html)]


def page_html(client):
    response = client.get("/compose")
    check(response.status_code == 200, f"/compose returned {response.status_code}")
    return response.get_data(as_text=True)


# ---------------------------------------------------------------------------
# route: batch size and paging
# ---------------------------------------------------------------------------

def fetch(client, **params):
    response = client.get("/compose/posts/more", query_string=params)
    check(response.status_code == 200, f"/compose/posts/more {params} returned {response.status_code}")
    return response.get_json()


def walk(client, limit, **filters):
    """Every card id, paging with ``limit`` the way the page does."""
    seen, offset, total, calls = [], 0, None, 0
    while True:
        data = fetch(client, offset=offset, limit=limit, **filters)
        calls += 1
        ids = card_ids(data["html"])
        seen.extend(ids)
        check(total is None or data["total"] == total, "the total must not change between pages")
        total = data["total"]
        check(data["next_offset"] == offset + len(ids), f"next_offset {data['next_offset']} after {offset}+{len(ids)}")
        check(data["has_more"] == (data["next_offset"] < total), "has_more must say whether cards remain")
        offset = data["next_offset"]
        if not data["has_more"]:
            return seen, total, calls
        check(len(ids) > 0, "a page with more to come must not be empty")
        check(calls < 1000, "paging does not terminate")


def section_route():
    database, web, client = seeded_app()
    check(web.POSTS_PAGE_SIZE == 20 and web.POSTS_MAX_BATCH == 100,
          "the page size is 20 and the batch ceiling 100")

    # batch size: default, explicit, clamped, junk
    def count(**params):
        return len(card_ids(fetch(client, offset=0, **params)["html"]))
    check(count() == 20, "the default batch is the page size")
    check(count(limit=5) == 5 and count(limit=1) == 1, "an explicit limit is honoured")
    check(count(limit=100) == 100, "the ceiling itself is allowed")
    check(count(limit=1000) == 100, "a limit above the ceiling is clamped to it")
    check(count(limit="abc") == 20 and count(limit="") == 20 and count(limit=0) == 20,
          "a junk or zero limit falls back to the page size")
    check(count(limit=-3) == 1, "a negative limit is clamped up to 1")

    # paging covers every card once, in one order, whatever the batch size
    ids_by_size = {}
    for limit in (20, 37, 100):
        ids, total, calls = walk(client, limit)
        check(total == 160 and len(ids) == 160 and len(set(ids)) == 160,
              f"limit {limit}: expected 160 distinct cards, got {len(ids)} ({len(set(ids))} distinct) of {total}")
        ids_by_size[limit] = ids
    check(ids_by_size[20] == ids_by_size[37] == ids_by_size[100], "every batch size lists the same order")
    check(walk(client, 100)[2] == 2, "Load all's 100-per-request batches need two requests for 160 cards")

    # filters: paging stays inside the filtered set
    ids, total, _ = walk(client, 100, platform="threads")
    check(total == 120 and len(ids) == 120 and len(set(ids)) == 120,
          f"platform=threads: expected 120 cards, got {len(ids)} of {total}")
    linkedin_ids, linkedin_total, _ = walk(client, 100, platform="linkedin")
    check(linkedin_total == 40 and not set(ids) & set(linkedin_ids),
          "a filtered walk must never include the other platform's cards")
    sorted_ids, _, _ = walk(client, 25, sort="shortest")
    check(len(sorted_ids) == 160 and sorted_ids != ids_by_size[20] and set(sorted_ids) == set(ids_by_size[20]),
          "sorting reorders the same cards, and paging keeps that order")
    shortest_first = fetch(client, offset=0, limit=100, sort="shortest")
    check(card_ids(shortest_first["html"])[:3] == sorted_ids[:3], "sorted pages agree with themselves")

    # edges: past the end, exact multiples
    data = fetch(client, offset=500, limit=100)
    check(data["html"].strip() == "" and data["has_more"] is False and data["total"] == 160,
          "an offset past the end is an empty, finished page")
    data = fetch(client, offset=60, limit=100)
    check(len(card_ids(data["html"])) == 100 and data["has_more"] is False and data["next_offset"] == 160,
          "the last batch reports has_more false")
    data = fetch(client, offset=0, limit=80, platform="linkedin")
    check(len(card_ids(data["html"])) == 40 and data["has_more"] is False,
          "a limit larger than the filtered set returns the set and stops")

    # the page itself still renders its first 20 and tells the browser the sizes
    html = page_html(client)
    check(len(card_ids(html)) == 20, "the page still renders only the first page of cards")
    check(re.search(r"const POSTS_PAGE_SIZE = 20;", html) and re.search(r"const POSTS_MAX_BATCH = 100;", html),
          "the page tells its script the page size and the batch ceiling")

    print("route: limit defaults to 20, clamps to 1..100, and paging covers each matching card once")
    print("PAGING_ROUTE_OK")


# ---------------------------------------------------------------------------
# node plumbing
# ---------------------------------------------------------------------------

def compose_script(**kw):
    database, web, client = seeded_app(**kw)
    html = page_html(client)
    return html, "\n".join(inline_scripts(html))


def lift_region(source):
    """The page's loading code, from POSTS_PAGE_SIZE to the end of its init listener,
    plus the functions it leans on, exactly as the page has them."""
    start = source.index("const POSTS_PAGE_SIZE")
    marker = "document.addEventListener('DOMContentLoaded', () => {\n    document.querySelectorAll('.auto-load-toggle')"
    end = source.index(marker)
    end = source.index("    watchLoadMore();\n});\n", end) + len("    watchLoadMore();\n});\n")
    pieces = [source[start:end]]
    for name in ("currentPostFilters", "postFilterQuery", "syncLoadMore", "anyFilterActive",
                 "applyPostFilters"):
        found = re.findall(rf"^(?:async\s+)?function {name}\(", source, re.M)
        check(len(found) == 1, f"{name} is declared {len(found)} times on the page, expected once")
        match = re.search(rf"^(?:async\s+)?function {name}\(.*?^\}}\n", source, re.M | re.S)
        check(match, f"could not lift {name}")
        pieces.append(match.group(0))
    keys = re.search(r"^const NARROWING_FILTER_KEYS = .*?;\n", source, re.M | re.S)
    check(keys, "could not lift NARROWING_FILTER_KEYS")
    pieces.append(keys.group(0))
    sig = re.search(r"^let renderedFilterSignature = JSON\.stringify\(\n.*?\);\n", source, re.M | re.S)
    check(sig, "could not lift renderedFilterSignature")
    pieces.append(sig.group(0))
    seq = re.search(r"^let filterRequestSeq = 0;\n", source, re.M)
    check(seq, "could not lift filterRequestSeq")
    pieces.append(seq.group(0))
    return "\n".join(pieces)


def run_node(harness, functions):
    node = node_path()
    work = tempfile.mkdtemp(prefix="paging_js_")
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


# One fake world: a DOM just big enough for the loading code, a server that pages
# a list of cards the way /compose/posts/more does, an IntersectionObserver that
# can be told whether its target is in view, and a localStorage that can break.
WORLD = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const tick = () => new Promise(r => setTimeout(r, 0));
const settle = async () => { for (let i = 0; i < 40; i++) await tick(); };

function world(opts = {}) {
  const w = {
    total: opts.total ?? 250, shown: opts.shown ?? 20, filters: Object.assign({
      'filter-platform': '', 'filter-used': '', 'filter-queued': '', 'filter-image': '',
      'filter-source': '', 'filter-brief': '', 'filter-sort': '' }, opts.filters || {}),
    fetches: [], toasts: [], gates: [], failures: [], stored: opts.stored || {},
    observed: 0, disconnected: 0, observer: null, visible: !!opts.visible, listeners: [],
    containerHtml: '', appended: 0, snapshots: [],
  };
  const cardsHtml = (n, start) => Array.from({ length: n }, (_, i) =>
    `<div class="post-item" data-post-id="${start + i}"></div>`).join('');
  w.containerHtml = cardsHtml(w.shown, 0);

  const child = (cls) => ({ cls, dataset: {}, innerHTML: '', disabled: false });
  const makeWrap = (shown, total) => {
    const wrap = { className: 'load-more-container', dataset: { total: String(total) },
                   kids: { '.load-more-btn': child('more'), '.load-all-btn': child('all') },
                   querySelector(sel) { return this.kids[sel] || null; },
                   remove() { w.wrap = null; } };
    wrap.kids['.load-more-btn'].dataset.offset = String(shown);
    wrap.kids['.load-more-btn'].innerHTML = `Load more (${shown} of ${total}) ▼`;
    wrap.kids['.load-all-btn'].innerHTML = `Load all (${total - shown} more) ⏬`;
    Object.defineProperty(wrap, 'innerHTML', { set(html) {
      const off = /load-more-btn" data-offset="(\d+)"/.exec(html);
      this.kids['.load-more-btn'].dataset.offset = off ? off[1] : '0';
      this.html = html;
    }, get() { return this.html || ''; } });
    return wrap;
  };
  w.wrap = w.shown < w.total ? makeWrap(w.shown, w.total) : null;

  const container = {
    insertAdjacentHTML(_, html) { w.containerHtml += html; w.appended += (html.match(/class="post-item"/g) || []).length; },
    querySelectorAll(sel) { return sel === '.post-item' ? (w.containerHtml.match(/class="post-item"/g) || []) : []; },
    insertAdjacentElement(_, el) { w.wrap = el; },
    get innerHTML() { return w.containerHtml; },
    set innerHTML(html) { w.containerHtml = html; },
  };
  const body = { querySelector: (sel) => sel === '.load-more-container' ? w.wrap
                                       : sel === '.posts-container' ? container : null };
  const toggles = [{ checked: true }];
  const spans = { 'filter-results-count': { textContent: '' }, 'posts-count': { textContent: '' } };

  // The server: the same paging /compose/posts/more does, over ``total`` cards
  // (or, for another filter, the number in ``w.totals[platform]``).
  const serve = (params) => {
    const platform = params.get('platform') || '';
    const total = platform && w.totals ? w.totals[platform] : w.total;
    const offset = parseInt(params.get('offset') || '0', 10);
    const limit = Math.max(1, Math.min(parseInt(params.get('limit') || '20', 10) || 20, 100));
    const n = Math.max(0, Math.min(limit, total - offset));
    return { html: cardsHtml(n, offset + (platform ? 100000 : 0)), has_more: total > offset + n,
             next_offset: offset + n, total };
  };

  const ctx = {
    console, Promise, URLSearchParams, JSON, parseInt, Math,
    POSTS_TOTAL: w.total, selectModeActive: false,
    showToast: (m, t) => w.toasts.push([t, m]),
    initLoadedPosts: () => {}, clearSelectAllScope: () => {}, updateSelectedCount: () => {},
    localStorage: {
      getItem: (k) => { if (opts.storageThrows) throw new Error('blocked'); return k in w.stored ? w.stored[k] : null; },
      setItem: (k, v) => { if (opts.storageThrows) throw new Error('blocked'); w.stored[k] = v; },
    },
    document: {
      getElementById: (id) => id in w.filters ? { get value() { return w.filters[id]; } }
                            : id === 'posts-container' ? container : spans[id] || null,
      querySelector: (sel) => sel === '#saved-posts-container .load-more-container' ? w.wrap
                            : sel === '#saved-posts-container .social-copy-body' ? body : null,
      querySelectorAll: (sel) => sel === '.auto-load-toggle' ? toggles : [],
      createElement: () => makeWrap(0, 0),
      addEventListener: (ev, fn) => w.listeners.push([ev, fn]),
    },
    fetch: async (url) => {
      const params = new URLSearchParams(url.split('?')[1] || '');
      w.fetches.push({ url, offset: params.get('offset'), limit: params.get('limit'),
                       platform: params.get('platform'), more: w.wrap && w.wrap.kids['.load-more-btn'].innerHTML,
                       all: w.wrap && w.wrap.kids['.load-all-btn'].innerHTML });
      const gate = w.gates.shift();
      if (gate) await gate;
      const failure = w.failures.shift();
      if (failure === 'throw') throw new Error('offline');
      if (failure) return { ok: false, status: 500, json: async () => ({ error: failure }) };
      return { ok: true, status: 200, json: async () => serve(params) };
    },
  };
  ctx.document.querySelector.bind(ctx.document);
  if (!opts.noObserver) {
    ctx.IntersectionObserver = class {
      constructor(cb, o) { this.cb = cb; this.options = o; w.observer = this; this.target = null; }
      observe(t) { this.target = t; w.observed++; if (w.visible) setTimeout(() => this.cb([{ isIntersecting: true }]), 0); }
      disconnect() { this.target = null; w.disconnected++; }
    };
  }
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  const run = (code) => vm.runInContext(code, ctx);
  w.ctx = ctx; w.run = run; w.toggles = toggles; w.spans = spans;
  w.more = () => w.wrap && w.wrap.kids['.load-more-btn'];
  w.all = () => w.wrap && w.wrap.kids['.load-all-btn'];
  w.cards = () => (w.containerHtml.match(/class="post-item"/g) || []).length;
  w.hold = () => { let release; w.gates.push(new Promise(r => { release = r; })); return release; };
  // What the real filter bar does: change a select, then apply.
  w.setFilter = (id, value) => { w.filters[id] = value; };
  return w;
}
"""


# ---------------------------------------------------------------------------
# loadall
# ---------------------------------------------------------------------------

LOADALL_HARNESS = WORLD + r"""
const out = {};
(async () => {
  // 1. 250 cards, 20 shown: three batches of up to 100, progress on the button, then the row goes
  let w = world();
  await w.run('loadAllPosts()');
  out.full = { fetches: w.fetches.map(f => [f.offset, f.limit]), cards: w.cards(), wrap: !!w.wrap,
               progress: w.fetches.map(f => f.all), moreDisabled: w.fetches.map(f => f.more),
               toasts: [...w.toasts], loading: w.run('postsLoading'), running: w.run('loadAllRunning') };

  // 2. under a filter: every batch carries it and pages inside it
  w = world({ total: 250, filters: { 'filter-platform': 'threads' } });
  w.totals = { threads: 130 };
  w.run(`renderedFilterSignature = JSON.stringify(currentPostFilters())`);
  await w.run('loadAllPosts()');
  out.filtered = { fetches: w.fetches.map(f => [f.offset, f.limit, f.platform]), cards: w.cards(),
                   toasts: [...w.toasts] };

  // 3. stop: a second click during the first batch lets it land and then ends the run
  w = world();
  let release = w.hold();
  const running = w.run('loadAllPosts()');
  await settle();
  const during = { all: w.all().innerHTML, moreDisabled: w.more().disabled, allDisabled: w.all().disabled };
  w.run('loadAllPosts()');
  await settle();
  const stopping = w.all().innerHTML;
  release();
  await running;
  await settle();
  out.stop = { during, stopping, fetches: w.fetches.length, cards: w.cards(), wrap: !!w.wrap,
               more: w.more() && w.more().innerHTML, all: w.all() && w.all().innerHTML,
               toasts: [...w.toasts], running: w.run('loadAllRunning'), stopFlag: w.run('loadAllStop') };

  // ...and the next Load all starts clean rather than inheriting the stop
  await w.run('loadAllPosts()');
  out.afterStop = { cards: w.cards(), wrap: !!w.wrap, toasts: w.toasts.length };

  // 4. a failed batch part-way: what loaded stays, the error shows, the row recovers
  for (const mode of ['server error', 'throw']) {
    w = world();
    w.failures = [null, mode === 'throw' ? 'throw' : 'Server exploded'];
    await w.run('loadAllPosts()');
    out[mode] = { fetches: w.fetches.length, cards: w.cards(), wrap: !!w.wrap,
                  more: w.more() && w.more().innerHTML, all: w.all() && w.all().innerHTML,
                  moreDisabled: w.more() && w.more().disabled, allDisabled: w.all() && w.all().disabled,
                  toasts: [...w.toasts], paused: w.run('autoLoadPaused'), running: w.run('loadAllRunning') };
    // a retry works (the click un-pauses)
    await w.run('loadAllPosts()');
    out[mode].retry = { cards: w.cards(), wrap: !!w.wrap };
  }

  // 5. the filters change while a batch is out: that batch is dropped, nothing more is requested
  w = world({ filters: { 'filter-platform': 'threads' } });
  w.run(`renderedFilterSignature = JSON.stringify(currentPostFilters())`);
  release = w.hold();
  const stale = w.run('loadAllPosts()');
  await settle();
  w.setFilter('filter-platform', 'linkedin');          // the user picks another platform
  release();
  await stale;
  await settle();
  out.staleFilter = { fetches: w.fetches.length, cards: w.cards(), appended: w.appended,
                      toasts: [...w.toasts], wrap: !!w.wrap, more: w.more().innerHTML,
                      offset: w.more().dataset.offset };

  // 6. a filter change that is still rendering (rendered signature differs): no request at all
  w = world();
  w.setFilter('filter-platform', 'threads');            // changed, not yet rendered
  await w.run('loadAllPosts()');
  out.unrendered = { fetches: w.fetches.length, cards: w.cards(), more: w.more().innerHTML };

  // 7. the real filter render during Load all: ends it, replaces the list, and nothing stale lands
  w = world();
  w.totals = { threads: 130 };
  release = w.hold();
  const all = w.run('loadAllPosts()');
  await settle();
  w.setFilter('filter-platform', 'threads');
  const applying = w.run('applyPostFilters()');
  await settle();
  release();
  await all;
  await applying;
  await settle();
  out.filterDuringAll = { fetches: w.fetches.map(f => [f.offset, f.limit, f.platform]), cards: w.cards(),
                          more: w.more() && w.more().innerHTML, all: w.all() && w.all().innerHTML,
                          running: w.run('loadAllRunning'), toasts: [...w.toasts],
                          stopFlag: w.run('loadAllStop') };

  // 8. one load at a time
  w = world();
  release = w.hold();
  const first = w.run("loadMorePosts({})");
  await settle();
  w.run("loadMorePosts({})");
  w.run("loadAllPosts()");
  await settle();
  const concurrent = w.fetches.length;
  release();
  await first;
  await settle();
  out.singleFlight = { whileLoadingMore: concurrent, after: w.fetches.length, cards: w.cards() };

  w = world();
  release = w.hold();
  const running2 = w.run('loadAllPosts()');
  await settle();
  w.run("loadMorePosts({})");
  await settle();
  const duringAll = w.fetches.length;
  release();
  await running2;
  out.singleFlightAll = { whileLoadingAll: duringAll, total: w.fetches.length };

  // 9. nothing to load: no row, no request, no crash
  w = world({ total: 10, shown: 10 });
  await w.run('loadAllPosts()');
  await w.run('loadMorePosts({})');
  out.nothing = { fetches: w.fetches.length, toasts: [...w.toasts] };

  // 10. an exact multiple, and a single batch
  w = world({ total: 120, shown: 20 });
  await w.run('loadAllPosts()');
  out.exact = { fetches: w.fetches.map(f => [f.offset, f.limit]), cards: w.cards(), wrap: !!w.wrap };
  w = world({ total: 60, shown: 20 });
  await w.run('loadAllPosts()');
  out.single = { fetches: w.fetches.length, cards: w.cards(), toasts: [...w.toasts] };

  console.log(JSON.stringify(out));
})().catch(e => { console.error(e.stack); process.exit(1); });
"""


def section_loadall():
    html, script = compose_script()
    r = run_node(LOADALL_HARNESS, lift_region(script))

    f = r["full"]
    check(f["fetches"] == [["20", "100"], ["120", "100"], ["220", "100"]],
          f"Load all asks for 100 at a time from where the list ends: {f['fetches']}")
    check(f["cards"] == 250 and f["wrap"] is False, f"every card is loaded and the row goes: {f}")
    check(f["progress"] == ["⏳ Loading… 20 of 250 — click to stop", "⏳ Loading… 120 of 250 — click to stop",
                            "⏳ Loading… 220 of 250 — click to stop"],
          f"the button shows progress, and says how to stop: {f['progress']}")
    check(f["toasts"] == [["success", "Loaded all 250 posts"]], f"a finished Load all says so once: {f['toasts']}")
    check(f["loading"] is False and f["running"] is False, "Load all leaves nothing marked as running")

    g = r["filtered"]
    check(g["fetches"] == [["20", "100", "threads"], ["120", "100", "threads"]] and g["cards"] == 130
          and g["toasts"] == [["success", "Loaded all 130 posts"]],
          f"Load all pages inside the filter it was started under: {g}")

    s = r["stop"]
    check(s["during"]["moreDisabled"] is True and s["during"]["allDisabled"] is False,
          f"during Load all the same button must stay clickable to stop it: {s['during']}")
    check(s["stopping"] == "⏳ Stopping…", f"stopping says so: {s['stopping']}")
    check(s["fetches"] == 1 and s["cards"] == 120 and s["wrap"],
          f"a stopped Load all keeps what arrived and asks for nothing more: {s}")
    check(s["more"] == "Load more (120 of 250) ▼" and s["all"] == "Load all (130 more) ⏬",
          f"after a stop the row is idle with the new counts: {s['more']!r} {s['all']!r}")
    check(s["toasts"] == [] and s["running"] is False and s["stopFlag"] is False,
          f"a stop is not a success and leaves no flag behind: {s}")
    check(r["afterStop"]["cards"] == 250 and r["afterStop"]["wrap"] is False,
          f"the next Load all runs to the end: {r['afterStop']}")

    for mode in ("server error", "throw"):
        e = r[mode]
        check(e["fetches"] == 2 and e["cards"] == 120 and e["wrap"],
              f"{mode}: the first batch stays and the second is not retried at once: {e}")
        check(e["more"] == "Load more (120 of 250) ▼" and e["all"] == "Load all (130 more) ⏬"
              and e["moreDisabled"] is False and e["allDisabled"] is False,
              f"{mode}: the row recovers to idle so the user can try again: {e}")
        check(len(e["toasts"]) == 1 and e["toasts"][0][0] == "error"
              and e["toasts"][0][1].startswith("Failed to load more posts"),
              f"{mode}: the failure is shown, once: {e['toasts']}")
        check(e["paused"] is True and e["running"] is False, f"{mode}: scrolling pauses after a failure: {e}")
        check(e["retry"] == {"cards": 250, "wrap": False}, f"{mode}: a retry loads the rest: {e['retry']}")

    t = r["staleFilter"]
    check(t["fetches"] == 1 and t["cards"] == 20 and t["appended"] == 0 and t["toasts"] == [],
          f"a batch for filters that are gone must not be appended: {t}")
    check(str(t["offset"]) == "20" and t["more"] == "Load more (20 of 250) ▼", f"nothing about the row changed: {t}")
    u = r["unrendered"]
    check(u["fetches"] == 0 and u["cards"] == 20, f"no request while a filter change is still rendering: {u}")

    d = r["filterDuringAll"]
    check(d["running"] is False and d["stopFlag"] is False and d["toasts"] == [],
          f"a filter change ends Load all quietly: {d}")
    check(d["cards"] == 20 and d["more"] == "Load more (20 of 130) ▼" and d["all"] == "Load all (110 more) ⏬",
          f"the list is the new filter's first page, and the old batch did not land on it: {d}")
    check(d["fetches"] == [["20", "100", None], ["0", None, "threads"]],
          f"no batch is requested for the old list after the filter changed: {d['fetches']}")

    c = r["singleFlight"]
    check(c["whileLoadingMore"] == 1 and c["after"] == 1 and c["cards"] == 40,
          f"a click during a load must not start a second load: {c}")
    check(r["singleFlightAll"]["whileLoadingAll"] == 1,
          f"Load more during Load all must not start a second load: {r['singleFlightAll']}")
    check(r["nothing"] == {"fetches": 0, "toasts": []}, f"with nothing left, clicks do nothing: {r['nothing']}")
    check(r["exact"]["fetches"] == [["20", "100"]] and r["exact"]["cards"] == 120 and r["exact"]["wrap"] is False,
          f"an exact fit finishes in one request: {r['exact']}")
    check(r["single"]["fetches"] == 1 and r["single"]["cards"] == 60, f"a short list: {r['single']}")

    print("loadall: batches until the filtered set is exhausted, progress, stop, failure, stale filter, one at a time")
    print("PAGING_LOADALL_OK")


# ---------------------------------------------------------------------------
# scroll
# ---------------------------------------------------------------------------

SCROLL_HARNESS = WORLD + r"""
const out = {};
const intersect = (w, yes = true) => w.observer.cb([{ isIntersecting: yes }]);
(async () => {
  // 1. nearing the row loads one page of 20; leaving it loads nothing
  let w = world();
  await w.run('watchLoadMore()');
  intersect(w, false);
  await settle();
  const away = w.fetches.length;
  intersect(w);
  await settle();
  out.basic = { away, fetches: w.fetches.map(f => [f.offset, f.limit]), cards: w.cards(),
                more: w.more().innerHTML, margin: w.observer.options.rootMargin,
                target: w.observer.target === w.wrap };

  // 2. switched off: remembered, observer let go, intersections ignored
  w = world();
  await w.run('watchLoadMore()');
  w.run('setAutoLoad(false)');
  const off = { stored: w.stored['compose.autoLoad'], observing: !!w.observer.target,
                toggles: w.toggles.map(t => t.checked) };
  intersect(w);
  await settle();
  off.fetches = w.fetches.length;
  w.run('setAutoLoad(true)');
  off.onStored = w.stored['compose.autoLoad'];
  off.observingAgain = w.observer.target === w.wrap;
  off.toggles2 = w.toggles.map(t => t.checked);
  intersect(w);
  await settle();
  off.fetchesOn = w.fetches.length;
  out.off = off;

  // 3. the choice is applied when the page loads
  w = world({ stored: { 'compose.autoLoad': 'off' } });
  w.listeners.filter(l => l[0] === 'DOMContentLoaded').forEach(l => l[1]());
  out.initOff = { toggles: w.toggles.map(t => t.checked), observer: !!w.observer && !!w.observer.target };
  w = world({ stored: {} });
  w.listeners.filter(l => l[0] === 'DOMContentLoaded').forEach(l => l[1]());
  out.initOn = { toggles: w.toggles.map(t => t.checked), observing: w.observer.target === w.wrap };

  // 4. blocked storage: on by default, switching never throws
  w = world({ storageThrows: true });
  let threw = false;
  try { w.run('setAutoLoad(false)'); } catch (e) { threw = true; }
  out.storage = { threw, enabledAfterBlockedSet: w.run('autoLoadEnabled()'), toggles: w.toggles.map(t => t.checked) };

  // 5. no IntersectionObserver at all: nothing breaks, the buttons still work
  w = world({ noObserver: true });
  let noObs = true;
  try { w.run('watchLoadMore()'); } catch (e) { noObs = false; }
  await w.run('loadMorePosts({})');
  out.noObserver = { ok: noObs, cards: w.cards() };

  // 6. a second intersection while a load is out loads nothing more
  w = world();
  await w.run('watchLoadMore()');
  let release = w.hold();
  intersect(w);
  await settle();
  intersect(w);
  intersect(w);
  await settle();
  const overlapped = w.fetches.length;
  release();
  await settle();
  out.overlap = { overlapped, after: w.fetches.length, cards: w.cards() };

  // 7. while Load all runs, scrolling asks for nothing
  w = world();
  await w.run('watchLoadMore()');
  release = w.hold();
  const all = w.run('loadAllPosts()');
  await settle();
  intersect(w);
  await settle();
  const during = w.fetches.length;
  release();
  await all;
  out.duringAll = { during, final: w.fetches.length, cards: w.cards() };

  // 8. a row still in view after a load loads the next page, to the end, then stops
  w = world({ total: 100, shown: 20, visible: true });
  await w.run('watchLoadMore()');
  await settle();
  out.keepsGoing = { fetches: w.fetches.map(f => [f.offset, f.limit]), cards: w.cards(), wrap: !!w.wrap,
                     observed: w.observed, target: w.observer.target };
  const before = w.fetches.length;
  await settle();
  out.keepsGoing.extra = w.fetches.length - before;

  // 9. a failure under scroll: one request, one error, no retry loop; a click recovers
  w = world({ total: 100, shown: 20, visible: true });
  w.failures = ['Server exploded'];
  await w.run('watchLoadMore()');
  await settle();
  const failed = { fetches: w.fetches.length, toasts: w.toasts.length, paused: w.run('autoLoadPaused') };
  intersect(w);
  await settle();
  failed.afterMoreIntersections = w.fetches.length;
  await w.run("loadMorePosts({})");        // the user clicks
  await settle();
  failed.afterClick = { fetches: w.fetches.length, cards: w.cards(), paused: w.run('autoLoadPaused') };
  out.failure = failed;

  // 10. a stale scroll load (filters changed in flight) is dropped and does not pause scrolling
  w = world({ filters: { 'filter-platform': 'threads' } });
  w.run(`renderedFilterSignature = JSON.stringify(currentPostFilters())`);
  await w.run('watchLoadMore()');
  release = w.hold();
  intersect(w);
  await settle();
  w.setFilter('filter-platform', 'linkedin');
  release();
  await settle();
  out.stale = { cards: w.cards(), appended: w.appended, toasts: [...w.toasts], paused: w.run('autoLoadPaused') };

  // 11. nothing left: the row and its observer are gone
  w = world({ total: 30, shown: 20 });
  await w.run('watchLoadMore()');
  intersect(w);
  await settle();
  out.end = { wrap: !!w.wrap, observing: !!w.observer.target, cards: w.cards(), fetches: w.fetches.length };

  console.log(JSON.stringify(out));
})().catch(e => { console.error(e.stack); process.exit(1); });
"""


def section_scroll():
    html, script = compose_script()
    r = run_node(SCROLL_HARNESS, lift_region(script))

    b = r["basic"]
    check(b["away"] == 0, "a row that is not near the viewport loads nothing")
    check(b["fetches"] == [["20", "20"]] and b["cards"] == 40 and b["more"] == "Load more (40 of 250) ▼",
          f"nearing the row loads one page of 20: {b}")
    check(b["target"] is True and b["margin"] == "600px 0px",
          f"the row itself is watched, with room to load before it is reached: {b}")

    o = r["off"]
    check(o["stored"] == "off" and o["observing"] is False and o["toggles"] == [False],
          f"switching off is remembered, stops watching and unticks every switch: {o}")
    check(o["fetches"] == 0, f"with scrolling off, reaching the row loads nothing: {o}")
    check(o["onStored"] == "on" and o["observingAgain"] is True and o["toggles2"] == [True] and o["fetchesOn"] == 1,
          f"switching back on is remembered and resumes: {o}")
    check(r["initOff"] == {"toggles": [False], "observer": False},
          f"a stored 'off' is applied when the page loads: {r['initOff']}")
    check(r["initOn"] == {"toggles": [True], "observing": True},
          f"scrolling is on by default and watching from the start: {r['initOn']}")
    check(r["storage"] == {"threw": False, "enabledAfterBlockedSet": True, "toggles": [False]},
          f"blocked storage must not break the switch or turn scrolling off for good: {r['storage']}")
    check(r["noObserver"] == {"ok": True, "cards": 40}, f"no IntersectionObserver: {r['noObserver']}")

    v = r["overlap"]
    check(v["overlapped"] == 1 and v["after"] == 1 and v["cards"] == 40,
          f"intersections during a load must not start another: {v}")
    d = r["duringAll"]
    check(d["during"] == 1 and d["cards"] == 250, f"scrolling is ignored while Load all runs: {d}")

    k = r["keepsGoing"]
    check(k["fetches"] == [["20", "20"], ["40", "20"], ["60", "20"], ["80", "20"]] and k["cards"] == 100,
          f"a row still in view keeps loading, a page at a time: {k}")
    check(k["wrap"] is False and k["target"] is None and k["extra"] == 0,
          f"at the end the row and watcher are gone and nothing more is asked: {k}")

    f = r["failure"]
    check(f["fetches"] == 1 and f["toasts"] == 1 and f["paused"] is True and f["afterMoreIntersections"] == 1,
          f"a failed scroll load shows one error and stops retrying on its own: {f}")
    check(f["afterClick"]["cards"] > 20 and f["afterClick"]["paused"] is False,
          f"a click recovers from a failed scroll load: {f['afterClick']}")

    s = r["stale"]
    check(s["cards"] == 20 and s["appended"] == 0 and s["toasts"] == [] and s["paused"] is False,
          f"a stale scroll load is dropped without pausing scrolling: {s}")
    e = r["end"]
    check(e["wrap"] is False and e["observing"] is False and e["cards"] == 30 and e["fetches"] == 1,
          f"the last page removes the row and its watcher: {e}")

    print("scroll: one page per approach, off switch remembered, no overlap, runs to the end, failure pauses")
    print("PAGING_SCROLL_OK")


# ---------------------------------------------------------------------------
# wiring: the rendered row and the JS-drawn row are the same
# ---------------------------------------------------------------------------

class Tags(HTMLParser):
    """The tags of a fragment as (tag, attributes, text), with attributes normalised."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        norm = {k: (" ".join(sorted(v.split())) if k == "class" else (v or "")) for k, v in attrs}
        item = {"tag": tag, "attrs": norm, "text": []}
        self.items.append(item)
        self.stack.append(item)

    def handle_endtag(self, tag):
        while self.stack:
            if self.stack.pop()["tag"] == tag:
                break

    def handle_data(self, data):
        for item in self.stack:
            item["text"].append(data)


def tags(fragment):
    parser = Tags()
    parser.feed(fragment)
    return [{"tag": i["tag"], "attrs": i["attrs"], "text": " ".join("".join(i["text"]).split())}
            for i in parser.items]


WIRING_HARNESS = WORLD + r"""
(async () => {
  const w = world();
  const html = w.run('loadMoreControlsHtml(20, 160)');
  // syncLoadMore builds the row when a filter change leaves more cards than are shown
  const w2 = world({ total: 160, shown: 160 });
  w2.wrap = null;
  w2.run('syncLoadMore(20, 160)');
  console.log(JSON.stringify({ html, wrapClass: w2.wrap && w2.wrap.className,
                               wrapTotal: w2.wrap && w2.wrap.dataset.total,
                               built: w2.wrap && w2.wrap.html, observed: w2.observed }));
})();
"""


def section_wiring():
    html, script = compose_script(threads=140, linkedin=20)

    match = re.search(r'<div class="text-center mt-3 pt-3 border-top load-more-container[^"]*"[^>]*>.*?</label>\s*</div>',
                      html, re.S)
    check(match, "the rendered page has no Load more row")
    row = match.group(0)
    rendered = tags(row)
    wrap, rest = rendered[0], rendered[1:]
    check(wrap["attrs"].get("data-total") == "160", f"the row carries the total for the script: {wrap['attrs']}")

    names = [(i["tag"], i["attrs"].get("class", "")) for i in rest if i["tag"] in ("button", "label", "input")]
    check(any("load-more-btn" in c for _, c in names), "the row has a Load more button")
    check(any("load-all-btn" in c for _, c in names), "the row has a Load all button")
    check(any("auto-load-toggle" in c for _, c in names), "the row has the scroll-loading switch")
    order = [c for t, c in names if t == "button"]
    check("load-more-btn" in order[0] and "load-all-btn" in order[1],
          f"Load all sits right next to Load more: {order}")
    by_class = {a["attrs"].get("class", ""): a for a in rest}
    more = next(i for i in rest if "load-more-btn" in i["attrs"].get("class", ""))
    every = next(i for i in rest if "load-all-btn" in i["attrs"].get("class", ""))
    check(more["text"] == "Load more (20 of 160) ▼" and more["attrs"]["data-offset"] == "20",
          f"Load more shows the counts: {more}")
    check(every["text"] == "Load all (140 more) ⏬" and "every post" in every["attrs"].get("title", ""),
          f"Load all shows how many remain: {every}")
    check(more["attrs"]["onclick"] == "loadMorePosts(this)" and every["attrs"]["onclick"] == "loadAllPosts(this)",
          "the buttons call the page's loaders")
    toggle = next(i for i in rest if "auto-load-toggle" in i["attrs"].get("class", ""))
    check("checked" in toggle["attrs"] and toggle["attrs"]["onchange"] == "setAutoLoad(this.checked)",
          f"the switch starts on and is wired to setAutoLoad: {toggle}")

    # the script draws the same row after a filter change
    r = run_node(WIRING_HARNESS, lift_region(script))
    drawn = tags(r["html"])
    check(drawn == rest,
          f"the Jinja row and loadMoreControlsHtml disagree:\n  jinja : {json.dumps(rest, ensure_ascii=False)[:900]}\n"
          f"  script: {json.dumps(drawn, ensure_ascii=False)[:900]}")
    check(sorted(r["wrapClass"].split()) == sorted(wrap["attrs"]["class"].split()),
          f"syncLoadMore builds the row with the classes the page does: {r['wrapClass']!r} vs {wrap['attrs']['class']!r}")
    check(str(r["wrapTotal"]) == "160", f"syncLoadMore records the total: {r['wrapTotal']}")
    check(r["observed"] >= 1, "syncLoadMore starts watching the row it builds")

    # the new loaders are declared once each, and the old entry point is still there
    for name in ("loadMorePosts", "loadAllPosts", "setAutoLoad", "watchLoadMore", "loadPostsBatch",
                 "syncLoadMore"):
        count = len(re.findall(rf"^(?:async\s+)?function {name}\(", script, re.M))
        check(count == 1, f"{name} must be declared exactly once on the Compose page, found {count}")

    # a page with everything on one screen has no row at all
    database, web, client = seeded_app(threads=5, linkedin=5)
    small = page_html(client)
    check("load-more-container" not in re.sub(r"<script.*?</script>", "", small, flags=re.S),
          "a list that fits on one page shows no Load more row")

    print("wiring: Load more, Load all and the scroll switch are in the page, and the script redraws them identically")
    print("PAGING_WIRING_OK")


SECTIONS = {"route": section_route, "wiring": section_wiring, "loadall": section_loadall,
            "scroll": section_scroll}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_compose_paging.py {'|'.join(SECTIONS)}")
        sys.exit(2)
    SECTIONS[name]()
