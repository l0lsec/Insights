"""Gates for Find & Replace results on Compose: going to the post, and acting on it.

    python scripts/check_find_results.py route     FIND_RESULTS_ROUTE_OK
    python scripts/check_find_results.py wiring    FIND_RESULTS_WIRING_OK
    python scripts/check_find_results.py jump      FIND_JUMP_OK
    python scripts/check_find_results.py actions   FIND_ACTIONS_OK
    python scripts/check_find_results.py server    FIND_ACTIONS_SERVER_OK

Each result in the Find & Replace modal stands for one card. "Go to post" closes
the modal, pages the list until that card is on it — however far down — and
scrolls to it; the ⋯ menu copies, marks used, posts now, schedules, queues or
deletes the whole card without leaving the results.

``route`` proves the search tells the page where each card sits in the list it is
showing, by walking /compose/posts/more and finding the card exactly there. The
browser code is run, not read: ``wiring`` draws a result with the page's own
function, ``jump`` and ``actions`` lift the page's functions and run them under
node against a fake DOM and a fake server. ``server`` sends the requests the
actions send to the real routes and reads the database back.

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
from check_compose_paging import lift_region  # noqa: E402


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------

def seeded_app():
    """160 cards: every third goes to LinkedIn and Threads (one card, two rows),
    the rest to one platform each; every fifth says "flight"."""
    directory, database, web, publisher, client = isolated_app()
    install_fake_clients(publisher, web)
    web._maybe_attach_link_image = lambda *a, **k: None
    connect(database, "linkedin", "li-work", "Work")
    connect(database, "threads", "th-main", "Main")
    for i in range(160):
        word = "flight" if i % 5 == 0 else "walk"
        # Lengths differ so a sort by length really reorders the list.
        content = f"Card {i:03d} about the {word}" + (" and more" * (i % 7))
        platforms = ["linkedin", "threads"] if i % 3 == 0 else (["threads"] if i % 2 else ["linkedin"])
        for platform in platforms:
            database.add_standalone_post("manual", "gate", platform, content, db_path=database.DB_PATH)
    return database, web, client


def card_rows_in(html):
    """[(card id, [row ids])] for each card in a rendered fragment, in order."""
    return [(int(cid), [int(x) for x in ids.split(",") if x])
            for cid, ids in re.findall(r'class="post-item" data-post-id="(\d+)"\s+data-post-ids="([\d,]*)"', html)]


def list_order(client, **filters):
    """Every card of the filtered list, in the order the page pages it."""
    cards, offset = [], 0
    while True:
        data = client.get("/compose/posts/more", query_string=dict(filters, offset=offset, limit=100)).get_json()
        cards.extend(card_rows_in(data["html"]))
        offset = data["next_offset"]
        if not data["has_more"]:
            return cards


def search(client, find, **params):
    response = client.get("/compose/posts/search", query_string=dict(params, find=find))
    check(response.status_code == 200, f"search {find!r} {params} returned {response.status_code}")
    data = response.get_json()
    check(data["success"], f"search failed: {data}")
    return data


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

def section_route():
    database, web, client = seeded_app()

    cases = [
        ({}, "no filter"),
        ({"platform": "threads"}, "platform filter"),
        ({"sort": "longest"}, "longest first"),
        ({"sort": "oldest", "platform": "linkedin"}, "oldest first, LinkedIn only"),
    ]
    checked = 0
    for filters, label in cases:
        order = list_order(client, **filters)
        data = search(client, "flight", **filters)
        check(data["posts"], f"{label}: the search found nothing to check against")
        for post in data["posts"]:
            position = post.get("position")
            check(isinstance(position, int) and 0 <= position < len(order),
                  f"{label}: result {post['id']} has no usable position: {position!r}")
            card_id, rows = order[position]
            check(post["id"] in rows,
                  f"{label}: the list has card {card_id} {rows} at {position}, not result {post['id']}")
            card = post.get("card") or {}
            check(card.get("id") == card_id,
                  f"{label}: result {post['id']} names card {card.get('id')}, the page draws {card_id}")
            check(sorted(card.get("post_ids") or []) == sorted(rows),
                  f"{label}: the result's card rows {card.get('post_ids')} are not the card's {rows}")
            targets = card.get("targets") or []
            check(sorted(t["post_id"] for t in targets) == sorted(rows)
                  and all(t.get("platform") and t.get("account_id") for t in targets),
                  f"{label}: every row needs a target with its platform and account: {targets}")
            check(card.get("used") is False, f"{label}: nothing is used yet: {card}")
            checked += 1
        positions = [p["position"] for p in data["posts"]]
        check(positions == sorted(positions), f"{label}: results come in list order: {positions}")

    # Under a platform filter the result narrows what Replace touches, but its
    # actions (delete, used, publish) still reach the whole card, like the card's own.
    data = search(client, "Card 000 ", platform="threads")
    check(len(data["posts"]) == 1, f"one card matches 'Card 000 ': {data['posts']}")
    post = data["posts"][0]
    check(len(post["post_ids"]) == 1 and len(post["card"]["post_ids"]) == 2,
          f"platform filter: replace rows {post['post_ids']}, card rows {post['card']['post_ids']}")

    # The used state is the card's: every row used reads used, one row does not.
    rows = post["card"]["post_ids"]
    database.mark_standalone_post_used(rows[0], True, db_path=database.DB_PATH)
    check(search(client, "Card 000 ")["posts"][0]["card"]["used"] is False, "one used row is not a used card")
    database.mark_standalone_post_used(rows[1], True, db_path=database.DB_PATH)
    check(search(client, "Card 000 ")["posts"][0]["card"]["used"] is True, "every row used reads as used")

    # A capped search still positions what it shows.
    data = search(client, "Card", limit=7)
    order = list_order(client)
    check(len(data["posts"]) == 7 and data["truncated"], f"the cap holds: {len(data['posts'])}")
    check([p["position"] for p in data["posts"]] == list(range(7))
          and all(p["id"] in order[p["position"]][1] for p in data["posts"]),
          "a capped search's results still sit where the list has them")

    print(f"route: {checked} results across {len(cases)} filter/sort cases sit exactly where the list pages them")
    print("FIND_RESULTS_ROUTE_OK")


# ---------------------------------------------------------------------------
# node plumbing
# ---------------------------------------------------------------------------

_SCRIPT = None


def page_script():
    """The rendered Compose page's inline scripts, once per run."""
    global _SCRIPT
    if _SCRIPT is None:
        database, web, client = seeded_app()
        response = client.get("/compose")
        check(response.status_code == 200, f"/compose returned {response.status_code}")
        html = response.get_data(as_text=True)
        _SCRIPT = (html, "\n".join(inline_scripts(html)))
    return _SCRIPT


def lift(source, names):
    pieces = []
    for name in names:
        found = re.findall(rf"^(?:async\s+)?function {name}\(", source, re.M)
        check(len(found) == 1, f"{name} is declared {len(found)} times on the page, expected once")
        match = re.search(rf"^(?:async\s+)?function {name}\(.*?^\}}\n", source, re.M | re.S)
        check(match, f"could not lift {name}")
        pieces.append(match.group(0))
    return "\n".join(pieces)


def platform_tables(source):
    """The page's PLATFORM_META block and the label/icon/colour tables built from it."""
    match = re.search(r"^const PLATFORM_META = .*?^\}\);\n", source, re.M | re.S)
    check(match, "could not lift the PLATFORM_META tables")
    return match.group(0)


def run_node(harness, functions):
    node = node_path()
    work = tempfile.mkdtemp(prefix="find_results_js_")
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


# ---------------------------------------------------------------------------
# wiring: the result markup, drawn by the page's own function
# ---------------------------------------------------------------------------

WIRING_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const ctx = {
  console, JSON, Math,
  escapeHtml: (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                                .replace(/"/g, '&quot;').replace(/'/g, '&#39;'),
};
vm.createContext(ctx);
vm.runInContext(source, ctx);
const base = { id: '12', content: 'Fly <away>\nnow', platform: 'linkedin', platforms: ['linkedin', 'threads'],
               index: '117', matchCount: 2, highlightedContent: 'Fly', previewContent: null, used: false };
const out = {
  many: ctx.searchResultCardHtml(Object.assign({}, base, { targets: [
    { post_id: 12, platform: 'linkedin', account_id: 3, account_label: 'Work <b>' },
    { post_id: 13, platform: 'linkedin', account_id: 4, account_label: 'Studio' },
    { post_id: 14, platform: 'threads', account_id: 5, account_label: null } ] })),
  one: ctx.searchResultCardHtml(Object.assign({}, base, { id: '20', used: true, platforms: ['threads'],
    targets: [{ post_id: 20, platform: 'threads', account_id: 5, account_label: null }] })),
  none: ctx.searchResultCardHtml(Object.assign({}, base, { id: '30', targets: [] })),
};
console.log(JSON.stringify(out));
"""


class Markup(HTMLParser):
    """Buttons (class, onclick, text), and every attribute seen, from a fragment."""

    def __init__(self):
        super().__init__()
        self.buttons, self.attrs, self._open = [], [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.attrs.append((tag, attrs))
        if tag == "button":
            self._open = {"class": attrs.get("class", ""), "onclick": attrs.get("onclick", ""),
                          "attrs": attrs, "text": ""}
            self.buttons.append(self._open)

    def handle_endtag(self, tag):
        if tag == "button":
            self._open = None

    def handle_data(self, data):
        if self._open is not None:
            self._open["text"] += data


def parse(fragment):
    markup = Markup()
    markup.feed(fragment)
    for button in markup.buttons:
        button["text"] = " ".join(button["text"].split())
    return markup


def handler(onclick):
    match = re.match(r"\s*(\w+)\(", onclick or "")
    return match.group(1) if match else None


def section_wiring():
    html, source = page_script()
    functions = platform_tables(source) + "\n" + lift(source, ["searchResultCardHtml", "searchResultMenuHtml"])
    r = run_node(WIRING_HARNESS, functions)
    declared = set(re.findall(r"^(?:async\s+)?function (\w+)\(", source, re.M))

    many = parse(r["many"])
    by_class = lambda cls: [b for b in many.buttons if cls in b["class"].split()]  # noqa: E731
    goto = by_class("search-goto-btn")
    badge = by_class("search-goto-badge")
    check(len(goto) == 1 and goto[0]["onclick"] == "jumpToSearchResult('12')" and "Go to post" in goto[0]["text"],
          f"the result has a Go to post button for its own id: {goto}")
    check(len(badge) == 1 and badge[0]["onclick"] == "jumpToSearchResult('12')" and "Post 117" in badge[0]["text"],
          f"the Post N badge jumps too: {badge}")
    check(len(by_class("search-edit-btn")) == 1, "the inline Edit button is still there")

    expect = {
        "search-copy-item": ["copySearchResult('12')"],
        "search-used-item": ["toggleSearchResultUsed('12')"],
        "search-postnow-item": ["postNowFromSearchResult('12', 0)", "postNowFromSearchResult('12', 1)",
                                "postNowFromSearchResult('12', 2)"],
        "search-postall-item": ["postAllFromSearchResult('12')"],
        "search-schedule-item": ["scheduleFromSearchResult('12', 0)", "scheduleFromSearchResult('12', 1)",
                                 "scheduleFromSearchResult('12', 2)"],
        "search-queueall-item": ["queueAllFromSearchResult('12')"],
        "search-delete-item": ["deleteSearchResult('12')"],
    }
    for cls, calls in expect.items():
        got = [b["onclick"] for b in by_class(cls)]
        check(got == calls, f"{cls}: expected {calls}, got {got}")
    texts = {b["onclick"]: b["text"] for b in many.buttons}
    check("Work <b>" in texts["postNowFromSearchResult('12', 0)"] and "<b>" not in r["many"].split("Work")[1][:12],
          f"an account label is shown as text, escaped: {texts['postNowFromSearchResult(' + chr(39) + '12' + chr(39) + ', 0)']!r}")
    check("Threads" in texts["postNowFromSearchResult('12', 2)"], "a target with no account label falls back to the platform")
    check("Post to all 3 accounts" in texts["postAllFromSearchResult('12')"]
          and "Queue all 3 accounts" in texts["queueAllFromSearchResult('12')"],
          "the all-accounts items name how many accounts")
    check(texts["toggleSearchResultUsed('12')"].endswith("Mark as used"), "an unused card offers Mark as used")

    toggle = by_class("search-actions-toggle")
    check(len(toggle) == 1 and toggle[0]["attrs"].get("data-bs-toggle") == "dropdown",
          f"the ⋯ button opens a dropdown: {toggle}")
    config = json.loads(toggle[0]["attrs"].get("data-bs-popper-config") or "{}")
    check(config.get("strategy") == "fixed",
          f"the menu is positioned against the viewport so the results' scroll box cannot clip it: {config}")

    every = [handler(b["onclick"]) for b in many.buttons if b["onclick"]]
    missing = sorted({name for name in every if name not in declared})
    check(not missing, f"result buttons call functions the page does not declare: {missing}")

    one = parse(r["one"])
    one_classes = [c for b in one.buttons for c in b["class"].split()]
    check("search-postall-item" not in one_classes, "a one-account card has no 'all accounts' publish")
    queue_one = [b for b in one.buttons if "search-queueall-item" in b["class"].split()]
    check(len(queue_one) == 1 and "Add to queue" in queue_one[0]["text"], f"a one-account card still queues: {queue_one}")
    check("search-used-badge" in r["one"] and any(b["text"].endswith("Mark as not used") for b in one.buttons),
          "a used card shows Used and offers to undo it")

    none = parse(r["none"])
    none_classes = [c for b in none.buttons for c in b["class"].split()]
    check("search-postnow-item" not in none_classes and "search-delete-item" in none_classes
          and "search-goto-btn" in none_classes,
          "without targets the menu drops publishing but keeps Go to post and Delete")

    # The page uses this function to draw results, and has the way back.
    do_search = lift(source, ["doSearch"])
    check("matchingPosts.map(searchResultCardHtml)" in do_search, "doSearch draws its results with searchResultCardHtml")
    pill = re.search(r'<div class="search-back-pill[^"]*" id="search-back-pill".*?</div>', html, re.S)
    check(pill, "the page has the Back to results control")
    pill_calls = {handler(c) for c in re.findall(r'onclick="([^"]+)"', pill.group(0))}
    check(pill_calls == {"reopenSearchResults", "hideBackToSearchResults"} and pill_calls <= declared,
          f"Back to results reopens the modal and can be dismissed: {pill_calls}")
    check("Go to post</strong> jumps to it" in html, "the modal's hint mentions Go to post")
    open_modal = lift(source, ["openSearchReplaceModal"])
    check("hideBackToSearchResults()" in open_modal and "getOrCreateInstance" in open_modal,
          "a fresh Find & Replace drops the old way back and reuses the one modal instance")

    print(f"wiring: Go to post, the Post N badge and {sum(len(v) for v in expect.values())} menu items "
          f"drawn by the page, bound to declared functions")
    print("FIND_RESULTS_WIRING_OK")


# ---------------------------------------------------------------------------
# jump and actions: a fake page
# ---------------------------------------------------------------------------

# A fake of just enough page: a list of cards (each with its rows), the Load
# more row, a server that pages ``total`` cards the way /compose/posts/more
# does, the Find & Replace modal, toasts, and a record of every request.
WORLD = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const tick = () => new Promise(r => setTimeout(r, 0));
const settle = async () => { for (let i = 0; i < 60; i++) await tick(); };

// Card n of the list: primary row 1000+n, its twin 5000+n.
const cardIds = (n) => [1000 + n, 5000 + n];

function world(opts = {}) {
  const w = { total: opts.total ?? 300, fetches: [], toasts: [], gates: [], failures: [], alerts: [],
              confirms: [], confirmAnswer: opts.confirm ?? true, revealed: [], cards: [], usedUI: [],
              refreshed: [], usedRefreshed: [], searches: [], scheduleOpened: [], clipboard: [],
              modalShown: opts.modalShown ?? true, modalHides: 0, modalShows: 0, scheduleListeners: [],
              replies: opts.replies || {}, wrap: null, badge: { textContent: opts.badge || '300 saved posts' } };

  const makeCard = (n) => {
    const ids = opts.cardRows ? opts.cardRows(n) : cardIds(n);
    const card = { n, dataset: { postId: String(ids[0]), postIds: ids.join(',') }, removed: false,
                   classList: { set: new Set(), add(c) { this.set.add(c); }, remove(c) { this.set.delete(c); },
                                contains(c) { return this.set.has(c); } },
                   offsetWidth: 100, querySelector() { return null; },
                   scrollIntoView(o) { w.revealed.push({ n, o }); },
                   remove() { this.removed = true; w.cards = w.cards.filter(c => c !== this); } };
    return card;
  };
  const shown = opts.shown ?? 20;
  for (let i = 0; i < Math.min(shown, w.total); i++) w.cards.push(makeCard(i));

  const child = () => ({ dataset: {}, innerHTML: '', disabled: false });
  const makeWrap = (shownN, total) => {
    const wrap = { dataset: { total: String(total) }, kids: { '.load-more-btn': child(), '.load-all-btn': child() },
                   querySelector(sel) { return this.kids[sel] || null; }, remove() { w.wrap = null; } };
    wrap.kids['.load-more-btn'].dataset.offset = String(shownN);
    return wrap;
  };
  if (shown < w.total) w.wrap = makeWrap(shown, w.total);

  const container = {
    insertAdjacentHTML(_, html) {
      for (const m of html.matchAll(/data-n="(\d+)"/g)) w.cards.push(makeCard(parseInt(m[1], 10)));
    },
  };
  const filters = { 'filter-platform': '', 'filter-used': '', 'filter-queued': '', 'filter-image': '',
                    'filter-source': '', 'filter-brief': '', 'filter-sort': '' };

  const serve = (params) => {
    const offset = parseInt(params.get('offset') || '0', 10);
    const limit = Math.max(1, Math.min(parseInt(params.get('limit') || '20', 10) || 20, 100));
    const n = Math.max(0, Math.min(limit, w.total - offset));
    const html = Array.from({ length: n }, (_, i) => `<div class="post-item" data-n="${offset + i}"></div>`).join('');
    return { html, has_more: w.total > offset + n, next_offset: offset + n, total: w.total };
  };

  const elements = {
    'searchReplaceModal': {
      classList: { contains: (c) => c === 'show' && w.modalShown },
      listeners: [],
      addEventListener(ev, fn) { this.listeners.push([ev, fn]); },
    },
    'scheduleModal': { addEventListener(ev, fn, o) { w.scheduleListeners.push([ev, fn, o]); } },
    'search-back-pill': { classList: { set: new Set(['d-none']), add(c) { this.set.add(c); },
                                       remove(c) { this.set.delete(c); } } },
    'search-find-input': { value: opts.find ?? 'flight' },
    'posts-container': container,
    'posts-count': w.badge,
  };
  const modal = {
    hide() {
      w.modalHides++;
      if (opts.modalNeverHides) return;
      w.modalShown = false;
      setTimeout(() => elements.searchReplaceModal.listeners
        .filter(([ev]) => ev === 'hidden.bs.modal').forEach(([, fn]) => fn()), 5);
    },
    show() { w.modalShows++; w.modalShown = true; },
  };

  const ctx = {
    console, Promise, URLSearchParams, JSON, parseInt, Math, Number, Set, Object, Array, String, Date,
    setTimeout, clearTimeout,
    selectModeActive: false, linkedinConnected: true, threadsConnected: true, facebookConnected: false,
    twitterConnected: false, instagramConnected: false,
    showToast: (m, t) => w.toasts.push([t, m]),
    alert: (m) => w.alerts.push(m),
    confirm: (m) => { w.confirms.push(m); return w.confirmAnswer; },
    initLoadedPosts: () => {}, updateSelectedCount: () => {}, clearSelectAllScope: () => {},
    setCardUsedUI: (card, used) => w.usedUI.push([card ? card.n : null, used]),
    refreshCard: async (card) => { w.refreshed.push(card ? card.n : null); },
    refreshCardUsedState: async (card) => { w.usedRefreshed.push(card ? card.n : null); },
    doSearch: async (preserve) => { w.searches.push(preserve); },
    openScheduleModal: (id, type, platform) => w.scheduleOpened.push([id, type, platform]),
    navigator: { clipboard: { writeText: async (t) => { w.clipboard.push(t); } } },
    FormData: class {
      constructor() { this.entries = []; }
      append(k, v) { this.entries.push([k, String(v)]); }
      get(k) { const e = this.entries.find(x => x[0] === k); return e ? e[1] : null; }
    },
    bootstrap: { Modal: { getInstance: () => modal, getOrCreateInstance: () => modal } },
    localStorage: { getItem: () => null, setItem: () => {} },
    IntersectionObserver: class { observe() {} disconnect() {} },
    document: {
      getElementById: (id) => id in filters ? { value: filters[id] } : elements[id] || null,
      querySelector: (sel) => {
        if (sel === '#saved-posts-container .load-more-container') return w.wrap;
        const m = /^\.post-item\[data-post-id="(\d+)"\]$/.exec(sel);
        if (m) return w.cards.find(c => c.dataset.postId === m[1]) || null;
        return null;
      },
      querySelectorAll: (sel) => sel === '.post-item' ? [...w.cards] : [],
      addEventListener: () => {},
      createElement: () => makeWrap(0, 0),
    },
    fetch: async (url, init) => {
      const [path, query] = url.split('?');
      const params = new URLSearchParams(query || '');
      const body = init && init.body;
      w.fetches.push({ url, path, offset: params.get('offset'), limit: params.get('limit'),
                       method: init ? init.method : 'GET',
                       post_ids: body && body.get ? body.get('post_ids') : null,
                       all: body && body.get ? body.get('all') : null,
                       account_id: body && body.get ? body.get('account_id') : null });
      const gate = w.gates.shift();
      if (gate) await gate;
      const failure = w.failures.shift();
      if (failure === 'throw') throw new Error('offline');
      if (failure) return { ok: false, status: 500, json: async () => ({ error: failure }) };
      if (path === '/compose/posts/more') return { ok: true, status: 200, json: async () => serve(params) };
      const reply = Object.keys(w.replies).find(k => path.endsWith(k));
      const data = reply ? w.replies[reply] : { success: true };
      return { ok: data.__status ? data.__status < 400 : true, status: data.__status || 200, json: async () => data };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  w.ctx = ctx;
  w.run = (code) => vm.runInContext(code, ctx);
  w.hold = () => { let release; w.gates.push(new Promise(r => { release = r; })); return release; };
  w.more = () => w.wrap && w.wrap.kids['.load-more-btn'];
  w.pill = () => !elements['search-back-pill'].classList.set.has('d-none');
  w.setResults = (posts) => { ctx.searchReplacePostData = posts; w.run('searchReplacePostData = globalThis.searchReplacePostData'); };
  return w;
}

// A result as doSearch keeps it.
function result(n, extra = {}) {
  const ids = cardIds(n);
  return Object.assign({ id: String(ids[1]), postIds: String(ids[1]), content: `Card ${n} flight`,
    platform: 'threads', platforms: ['threads'], index: String(n + 1), position: n,
    cardId: String(ids[0]), cardPostIds: ids, used: false,
    targets: [{ post_id: ids[0], platform: 'linkedin', account_id: 3, account_label: 'Work' },
              { post_id: ids[1], platform: 'threads', account_id: 5, account_label: null }] }, extra);
}
"""


def jump_functions(source):
    names = ["searchResultPost", "loadedCardFor", "cardForPostId", "waitForPostsIdle", "ensureCardLoaded",
             "revealCard", "hideSearchModal", "jumpToSearchResult", "showBackToSearchResults",
             "hideBackToSearchResults", "reopenSearchResults"]
    return (lift_region(source) + "\n" + lift(source, names)
            + "\nvar searchReplacePostData = [];\n")


JUMP_HARNESS = WORLD + r"""
const out = {};
const jump = async (w, n, extra) => {
  w.setResults([result(n, extra)]);
  await w.run(`jumpToSearchResult('${cardIds(n)[1]}')`);
  await settle();
};
const summary = (w) => ({
  fetches: w.fetches.map(f => [f.offset, f.limit]), cards: w.cards.length, revealed: w.revealed.map(r => [r.n, r.o.block]),
  highlighted: w.cards.filter(c => c.classList.contains('jump-highlight')).map(c => c.n),
  toasts: [...w.toasts], pill: w.pill(), modalHides: w.modalHides, modalShown: w.modalShown,
  loading: w.run('postsLoading'), more: w.more() && w.more().innerHTML, offset: w.more() && String(w.more().dataset.offset),
});
(async () => {
  // 1. deep: card 250 of 300 with 20 shown — batches of 100 until it lands, then stop
  let w = world();
  await jump(w, 250);
  out.deep = summary(w);

  // 2. near: card 45 — one request sized to reach it exactly
  w = world();
  await jump(w, 45);
  out.near = summary(w);

  // 3. already on the page: no request at all
  w = world();
  await jump(w, 5);
  out.loaded = summary(w);

  // 4. found by its twin row: the result names row 5000+n, the page card is keyed by 1000+n
  w = world();
  await jump(w, 7, { cardPostIds: [cardIds(7)[1]] });
  out.twin = summary(w);

  // 5. moved since the search: the hint says 39, the card is now at 40
  w = world();
  w.setResults([result(40, { position: 39 })]);
  await w.run(`jumpToSearchResult('${cardIds(40)[1]}')`);
  await settle();
  out.moved = summary(w);

  // 6. not in the list: every page loads, then it says so, once, and stops
  w = world({ total: 60 });
  await jump(w, 999, { position: null });
  out.missing = summary(w);

  // 7. a load already in flight (a scroll load): wait for it, then go on — never two at once
  w = world();
  const release = w.hold();
  const scroll = w.run("loadMorePosts('scroll')");
  await settle();
  w.setResults([result(150)]);
  const going = w.run(`jumpToSearchResult('${cardIds(150)[1]}')`);
  await settle();
  const during = w.fetches.length;
  release();
  await scroll; await going; await settle();
  out.inflight = Object.assign(summary(w), { during });

  // 8. a load fails part-way: it stops and says to try again, not "not in the list"
  w = world();
  w.failures = [null, 'Server exploded'];
  await jump(w, 250);
  out.failed = summary(w);

  // 9. the modal never reports hidden (mid-transition): the jump still goes ahead
  w = world({ modalNeverHides: true });
  await jump(w, 5);
  out.stuckModal = summary(w);

  // 10. Back to results: reopens the one modal and re-runs the search keeping state
  w = world();
  await jump(w, 5);
  const before = w.pill();
  w.run('reopenSearchResults()');
  await settle();
  out.back = { before, after: w.pill(), shows: w.modalShows, searches: [...w.searches] };

  console.log(JSON.stringify(out));
})().catch(e => { console.error(e.stack); process.exit(1); });
"""


def section_jump():
    html, source = page_script()
    r = run_node(JUMP_HARNESS, jump_functions(source))

    d = r["deep"]
    check(d["fetches"] == [["20", "100"], ["120", "100"], ["220", "31"]],
          f"a deep card loads 100 at a time, then just enough to reach it, and stops: {d['fetches']}")
    check(d["revealed"] == [[250, "center"]] and d["highlighted"] == [250],
          f"the card is scrolled to the middle and highlighted: {d}")
    check(d["modalHides"] == 1 and d["modalShown"] is False and d["pill"] is True,
          f"the modal closes and the way back shows: {d}")
    check(d["loading"] is False and d["offset"] == "251" and d["more"] == "Load more (251 of 300) ▼",
          f"the list is left idle, counting what is now shown: {d}")
    check([t for t in d["toasts"] if t[0] != "info"] == [], f"a successful jump shows no warning: {d['toasts']}")

    n = r["near"]
    check(n["fetches"] == [["20", "26"]] and n["cards"] == 46 and n["revealed"] == [[45, "center"]],
          f"a near card takes one request sized to reach it: {n}")
    check(n["more"] == "Load more (46 of 300) ▼", f"the Load more row counts what is now shown: {n['more']!r}")

    a = r["loaded"]
    check(a["fetches"] == [] and a["revealed"] == [[5, "center"]] and a["toasts"] == [],
          f"a card already on the page is revealed with no request: {a}")
    t = r["twin"]
    check(t["fetches"] == [] and t["revealed"] == [[7, "center"]], f"the card is found through any of its rows: {t}")

    m = r["moved"]
    check(m["fetches"] == [["20", "20"], ["40", "20"]] and m["revealed"] == [[40, "center"]],
          f"a card that moved past the hint is still found, a batch later: {m}")

    x = r["missing"]
    check(x["fetches"] == [["20", "20"], ["40", "20"]] and x["revealed"] == [] and x["loading"] is False,
          f"a card not in the list loads the list once and stops: {x}")
    warnings = [t for t in x["toasts"] if t[0] == "warning"]
    check(len(warnings) == 1 and "isn't in the list under the current filters" in warnings[0][1],
          f"it says the filters don't cover it, once: {x['toasts']}")

    i = r["inflight"]
    check(i["during"] == 1, f"no second request while a load is in flight: {i['during']}")
    check(i["fetches"][0] == ["20", "20"] and i["fetches"][1:] == [["40", "100"], ["140", "20"]]
          and i["revealed"] == [[150, "center"]],
          f"after the in-flight load lands the jump carries on from there: {i['fetches']}")

    f = r["failed"]
    check(f["fetches"] == [["20", "100"], ["120", "100"]] and f["revealed"] == [] and f["loading"] is False,
          f"a failed load stops the jump: {f}")
    fw = [t[1] for t in f["toasts"] if t[0] in ("warning", "error")]
    check(any(s.startswith("Failed to load more posts") for s in fw)
          and any("try Go to post again" in s for s in fw)
          and not any("isn't in the list" in s for s in fw),
          f"a failure is not reported as 'not in the list': {f['toasts']}")

    s = r["stuckModal"]
    check(s["revealed"] == [[5, "center"]], f"a modal that never reports hidden doesn't block the jump: {s}")

    b = r["back"]
    check(b["before"] is True and b["after"] is False and b["shows"] == 1 and b["searches"] == [True],
          f"Back to results reopens the modal and re-runs the search keeping state: {b}")

    print("jump: deep, near, loaded, twin, moved, missing, in-flight, failed and stuck-modal cases behave")
    print("FIND_JUMP_OK")


ACTIONS_HARNESS = WORLD + r"""
const out = {};
const posts = (w) => w.fetches.filter(f => f.method === 'POST').map(f => [f.path, f.post_ids, f.all, f.account_id]);
(async () => {
  // delete, card on the page (card 5)
  let w = world();
  w.setResults([result(5)]);
  await w.run(`deleteSearchResult('${cardIds(5)[1]}')`);
  await settle();
  out.deleteLoaded = { posts: posts(w), confirms: [...w.confirms], removed: !w.cards.some(c => c.n === 5),
                       badge: w.badge.textContent, more: w.more().innerHTML, offset: String(w.more().dataset.offset),
                       searches: [...w.searches], toasts: [...w.toasts] };

  // delete, card not loaded (card 200): the total shrinks, the offset does not
  w = world();
  w.setResults([result(200)]);
  await w.run(`deleteSearchResult('${cardIds(200)[1]}')`);
  await settle();
  out.deleteUnloaded = { posts: posts(w), cards: w.cards.length, badge: w.badge.textContent,
                         more: w.more().innerHTML, offset: String(w.more().dataset.offset) };

  // delete under a filter: the badge reads "N of M posts" and both numbers drop
  w = world({ badge: '120 of 300 posts' });
  w.setResults([result(5)]);
  await w.run(`deleteSearchResult('${cardIds(5)[1]}')`);
  await settle();
  out.deleteFilteredBadge = { badge: w.badge.textContent };

  // delete, cancelled
  w = world({ confirm: false });
  w.setResults([result(5)]);
  await w.run(`deleteSearchResult('${cardIds(5)[1]}')`);
  await settle();
  out.deleteCancelled = { posts: posts(w), cards: w.cards.length, searches: w.searches.length };

  // delete, server refuses: nothing removed, the error is shown
  w = world({ replies: { '/delete': { __status: 404, error: 'Post not found' } } });
  w.setResults([result(5)]);
  await w.run(`deleteSearchResult('${cardIds(5)[1]}')`);
  await settle();
  out.deleteRefused = { cards: w.cards.length, badge: w.badge.textContent, toasts: [...w.toasts],
                        searches: w.searches.length };

  // used toggle
  w = world({ replies: { '/toggle-used': { success: true, used: true } } });
  w.setResults([result(5)]);
  await w.run(`toggleSearchResultUsed('${cardIds(5)[1]}')`);
  await settle();
  out.used = { posts: posts(w), usedUI: [...w.usedUI], searches: [...w.searches], toasts: [...w.toasts] };

  // post now, one account (the Threads one, index 1), card not loaded
  w = world({ replies: { '/threads': { success: true, permalink: 'https://threads.net/x' } } });
  w.setResults([result(200)]);
  await w.run(`postNowFromSearchResult('${cardIds(200)[1]}', 1)`);
  await settle();
  out.postNow = { posts: posts(w), confirms: [...w.confirms], toasts: [...w.toasts], searches: [...w.searches] };

  // post now, cancelled
  w = world({ confirm: false });
  w.setResults([result(5)]);
  await w.run(`postNowFromSearchResult('${cardIds(5)[1]}', 0)`);
  await settle();
  out.postNowCancelled = { posts: posts(w) };

  // post to all, card on the page
  w = world({ replies: { '/publish': { success: true, published: 2, message: 'Published to 2 accounts',
    results: [{ success: true, platform: 'linkedin', post_urn: 'urn:li:share:1', account_id: 3 },
              { success: true, platform: 'threads', permalink: 'https://threads.net/y', account_id: 5 }] } } });
  w.setResults([result(5)]);
  await w.run(`postAllFromSearchResult('${cardIds(5)[1]}')`);
  await settle();
  out.postAll = { posts: posts(w), confirms: [...w.confirms], toasts: [...w.toasts],
                  usedRefreshed: [...w.usedRefreshed], searches: [...w.searches] };

  // queue all, card on the page, then not
  w = world({ replies: { '/queue': { success: true, queued: [{}, {}], message: 'Queued 2' } } });
  w.setResults([result(5), result(200)]);
  await w.run(`queueAllFromSearchResult('${cardIds(5)[1]}')`);
  await w.run(`queueAllFromSearchResult('${cardIds(200)[1]}')`);
  await settle();
  out.queueAll = { posts: posts(w), refreshed: [...w.refreshed], toasts: [...w.toasts] };

  // queue all, nothing went in: it must not say Queued
  w = world({ replies: { '/queue': { success: false, queued: [], error: 'Nothing to queue', __status: 400 } } });
  w.setResults([result(5)]);
  await w.run(`queueAllFromSearchResult('${cardIds(5)[1]}')`);
  await settle();
  out.queueNone = { toasts: [...w.toasts] };

  // schedule: steps out of the results, opens the schedule modal for that row, comes back after
  w = world();
  w.setResults([result(5)]);
  await w.run(`scheduleFromSearchResult('${cardIds(5)[1]}', 1)`);
  await settle();
  const listener = w.scheduleListeners.find(([ev, , o]) => ev === 'hidden.bs.modal' && o && o.once);
  const opened = [...w.scheduleOpened];
  const hides = w.modalHides;
  if (listener) listener[1]();
  await settle();
  out.schedule = { opened, hides, listener: !!listener, shows: w.modalShows, searches: [...w.searches] };

  // schedule on a platform that isn't connected: says so and stays in the results
  w = world();
  w.setResults([result(5, { targets: [{ post_id: 1005, platform: 'twitter', account_id: 9, account_label: null }] })]);
  await w.run(`scheduleFromSearchResult('${cardIds(5)[1]}', 0)`);
  await settle();
  out.scheduleOff = { alerts: [...w.alerts], opened: w.scheduleOpened.length, hides: w.modalHides };

  // copy
  w = world();
  w.setResults([result(5)]);
  await w.run(`copySearchResult('${cardIds(5)[1]}')`);
  await settle();
  out.copy = { clipboard: [...w.clipboard], toasts: [...w.toasts] };

  // the card's own buttons still go through the same requests (refactored here)
  w = world();
  const card = w.cards.find(c => c.n === 5);
  card.dataset.platforms = JSON.stringify({ linkedin: 1005, threads: 5005 });
  card.closest = () => card;
  w.ctx.__card = card;
  await w.run(`deletePost(1005, { closest: () => __card })`);
  await settle();
  out.cardDelete = { posts: posts(w), confirms: [...w.confirms], removed: card.removed,
                     offset: String(w.more().dataset.offset), badge: w.badge.textContent };
  w = world({ replies: { '/toggle-used': { success: true, used: false } } });
  const card2 = w.cards.find(c => c.n === 6);
  w.ctx.__card = card2;
  await w.run(`toggleUsed(1006, { closest: () => __card })`);
  await settle();
  out.cardUsed = { posts: posts(w), usedUI: [...w.usedUI] };

  console.log(JSON.stringify(out));
})().catch(e => { console.error(e.stack); process.exit(1); });
"""


def actions_functions(source):
    names = ["searchResultPost", "loadedCardFor", "cardForPostId", "hideSearchModal",
             "showBackToSearchResults", "hideBackToSearchResults", "reopenSearchResults",
             "copySearchResult", "toggleSearchResultUsed", "postNowFromSearchResult", "postAllFromSearchResult",
             "queueAllFromSearchResult", "scheduleFromSearchResult", "deleteSearchResult",
             "requestToggleUsed", "toggleUsed", "deleteConfirmText", "requestDeleteCard", "noteCardDeleted",
             "deletePost", "postNowConfirmText", "publishAllConfirmText", "publishCardToAll", "queueCardToAll",
             "publishCardPlatform", "markChipPosted", "postedUrlFrom", "platformConnected",
             "adjustPostCountBadge", "loadMoreControls", "setLoadMoreLabels", "cardPostIds", "cardPlatforms"]
    return (platform_tables(source) + "\n" + lift(source, names)
            + "\nvar searchReplacePostData = [];\nvar openSearchEdits = {};\n")


def section_actions():
    html, source = page_script()
    r = run_node(ACTIONS_HARNESS, actions_functions(source))
    whole = "1005,5005"

    d = r["deleteLoaded"]
    check(d["posts"] == [["/compose/post/1005/delete", whole, None, None]],
          f"delete sends every row of the card to the card's delete route: {d['posts']}")
    check(d["confirms"] == ["Delete this post? It goes to LinkedIn, Threads — all of them will be deleted."],
          f"delete asks first, naming every platform: {d['confirms']}")
    check(d["removed"] and d["badge"] == "299 saved posts", f"the page card goes and the count drops: {d}")
    check(d["offset"] == "19" and d["more"] == "Load more (19 of 299) ▼",
          f"a deleted loaded card moves Load more back one so nothing is skipped: {d}")
    check(d["searches"] == [True] and ["success", "Post deleted"] in d["toasts"],
          f"the results re-run, keeping state, and it says so: {d}")

    u = r["deleteUnloaded"]
    check(u["posts"] == [["/compose/post/1200/delete", "1200,5200", None, None]] and u["cards"] == 20,
          f"a card that isn't loaded is deleted by its rows and nothing on the page goes: {u}")
    check(u["offset"] == "20" and u["more"] == "Load more (20 of 299) ▼" and u["badge"] == "299 saved posts",
          f"only the total shrinks: {u}")

    check(r["deleteFilteredBadge"]["badge"] == "119 of 299 posts",
          f"a filtered count badge drops both numbers, not read as one big number: {r['deleteFilteredBadge']}")

    c = r["deleteCancelled"]
    check(c["posts"] == [] and c["cards"] == 20 and c["searches"] == 0, f"a cancelled delete does nothing: {c}")
    f = r["deleteRefused"]
    check(f["cards"] == 20 and f["badge"] == "300 saved posts" and f["searches"] == 0
          and ["error", "Could not delete: Post not found"] in f["toasts"],
          f"a refused delete changes nothing and says why: {f}")

    s = r["used"]
    check(s["posts"] == [["/compose/post/1005/toggle-used", whole, None, None]]
          and s["usedUI"] == [[5, True]] and s["searches"] == [True] and ["success", "Marked as used"] in s["toasts"],
          f"used toggles the whole card, the page card and the results follow: {s}")

    p = r["postNow"]
    check(p["posts"] == [["/compose/post/5200/threads", None, None, "5"]],
          f"post now publishes that account's row to its platform, naming the account: {p['posts']}")
    check(p["confirms"] and p["confirms"][0] == "Post to Threads now?"
          and ["success", "Posted to Threads!"] in p["toasts"] and p["searches"] == [True],
          f"post now asks, reports and refreshes: {p}")
    check(r["postNowCancelled"]["posts"] == [], "a cancelled post now sends nothing")

    a = r["postAll"]
    check(a["posts"] == [["/compose/post/1005/publish", whole, None, None]],
          f"post to all publishes every row in one request: {a['posts']}")
    check(a["confirms"] == ["Publish this post to Work, Threads now? This cannot be undone."]
          and ["success", "Published to 2 accounts"] in a["toasts"] and a["usedRefreshed"] == [5]
          and a["searches"] == [True],
          f"post to all asks, reports, and refreshes the page card and results: {a}")

    q = r["queueAll"]
    check(q["posts"] == [["/compose/post/1005/queue", whole, "1", None], ["/compose/post/1200/queue", "1200,5200", "1", None]],
          f"queue all queues every row of the card: {q['posts']}")
    check(q["refreshed"] == [5], f"a loaded card is redrawn after queueing, an unloaded one is left: {q['refreshed']}")
    check(["error", "Nothing to queue"] in r["queueNone"]["toasts"]
          and not any("Queued" in m for _, m in r["queueNone"]["toasts"]),
          f"nothing queued never says Queued: {r['queueNone']}")

    h = r["schedule"]
    check(h["opened"] == [[5005, "standalone", "threads"]] and h["hides"] == 1 and h["listener"],
          f"schedule steps out of the results and opens the schedule modal for that row: {h}")
    check(h["shows"] == 1 and h["searches"] == [True], f"closing the schedule modal comes back to the results: {h}")
    o = r["scheduleOff"]
    check(o["alerts"] == ["Please connect X first"] and o["opened"] == 0 and o["hides"] == 0,
          f"an unconnected platform says so and stays in the results: {o}")

    check(r["copy"]["clipboard"] == ["Card 5 flight"], f"copy puts the post's text on the clipboard: {r['copy']}")

    cd = r["cardDelete"]
    check(cd["posts"] == [["/compose/post/1005/delete", whole, None, None]] and cd["removed"]
          and cd["offset"] == "19" and cd["badge"] == "299 saved posts"
          and cd["confirms"] == ["Delete this post? It goes to LinkedIn, Threads — all of them will be deleted."],
          f"the card's own delete still sends every row and now keeps Load more in step: {cd}")
    cu = r["cardUsed"]
    check(cu["posts"] == [["/compose/post/1006/toggle-used", "1006,5006", None, None]] and cu["usedUI"] == [[6, False]],
          f"the card's own used toggle still sends every row and redraws: {cu}")

    print("actions: delete, used, post now, post to all, queue all, schedule and copy send the whole card "
          "and keep the page and results in step")
    print("FIND_ACTIONS_OK")


# ---------------------------------------------------------------------------
# server: the actions' requests against the real routes
# ---------------------------------------------------------------------------

def section_server():
    database, web, client = seeded_app()
    db = database.DB_PATH
    # Two slots a day, every day, so queue-all has somewhere to put each row.
    for hhmm in ("09:00", "18:00"):
        database.add_time_slot(-1, hhmm, True, None, db_path=db)

    def row(pid):
        return database.get_standalone_post(pid, db_path=db)

    def result_for(text):
        data = search(client, text)
        check(len(data["posts"]) == 1, f"{text!r} should match one card: {len(data['posts'])}")
        return data["posts"][0]

    # used: the result's request flips every row of the card, and back
    post = result_for("Card 003 ")
    ids = post["card"]["post_ids"]
    check(len(ids) == 2, f"card 3 goes to two platforms: {ids}")
    for expected in (True, False):
        response = client.post(f"/compose/post/{post['card']['id']}/toggle-used",
                               data={"post_ids": ",".join(map(str, ids))})
        check(response.status_code == 200 and response.get_json()["used"] is expected,
              f"toggle-used should return used={expected}: {response.get_json()}")
        check(all(bool(row(i)["used"]) is expected for i in ids), f"every row of the card is used={expected}")

    # queue all: every row of the card gets its own pending entry
    post = result_for("Card 006 ")
    ids = post["card"]["post_ids"]
    response = client.post(f"/compose/post/{post['card']['id']}/queue",
                           data={"all": "1", "post_ids": ",".join(map(str, ids))})
    data = response.get_json()
    check(response.status_code == 200 and len(data.get("queued") or []) == 2, f"queue all queued both rows: {data}")
    pending = database.get_pending_schedules_for_standalone_posts(ids, db_path=db)
    check(sorted(pending) == sorted(ids), f"each row has a pending schedule: {pending}")
    check(search(client, "Card 006 ")["posts"][0]["card"]["queued"] and True,
          "the result reports the card as queued afterwards")

    # a one-platform card queues through the same request
    post = result_for("Card 001 ")
    check(len(post["card"]["post_ids"]) == 1, f"card 1 goes to one platform: {post['card']}")
    response = client.post(f"/compose/post/{post['card']['id']}/queue",
                           data={"all": "1", "post_ids": str(post["card"]["post_ids"][0])})
    check(response.status_code == 200 and len(response.get_json().get("queued") or []) == 1,
          f"Add to queue on a one-platform card queues it: {response.get_json()}")

    # delete under a platform filter: the result's replace rows are one, but the
    # delete takes the whole card, as the card's own button does
    data = search(client, "Card 009 ", platform="threads")
    post = data["posts"][0]
    ids = post["card"]["post_ids"]
    check(len(post["post_ids"]) == 1 and len(ids) == 2, f"filtered result, whole card: {post}")
    response = client.post(f"/compose/post/{post['card']['id']}/delete", data={"post_ids": ",".join(map(str, ids))})
    check(response.status_code == 200 and sorted(response.get_json()["deleted_ids"]) == sorted(ids),
          f"delete removed every row: {response.get_json()}")
    check(all(row(i) is None for i in ids), "no row of the card is left")
    check(search(client, "Card 009 ")["matched_posts"] == 0, "the deleted card no longer turns up in a search")
    check(search(client, "Card 012 ")["matched_posts"] == 1, "its neighbours are untouched")

    print("server: used, queue all and delete from a result reach every row of the card on the real routes")
    print("FIND_ACTIONS_SERVER_OK")


SECTIONS = {
    "route": section_route,
    "wiring": section_wiring,
    "jump": section_jump,
    "actions": section_actions,
    "server": section_server,
}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SECTIONS:
        print(f"usage: check_find_results.py {{{'|'.join(SECTIONS)}}}")
        sys.exit(2)
    SECTIONS[sys.argv[1]]()
