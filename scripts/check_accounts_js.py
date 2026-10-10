"""Gates that run the accounts and cross-posting JavaScript, not just read it.

    python scripts/check_accounts_js.py compose    COMPOSE_ACCOUNTS_JS_OK
    python scripts/check_accounts_js.py accounts   ACCOUNTS_PAGE_JS_OK
    python scripts/check_accounts_js.py syntax     ACCOUNTS_JS_SYNTAX_OK

The accounts UI gate renders the real pages and checks that the controls and
function names are present. That is how a "Queue all accounts" button shipped
broken: the markup was right, the handler was named, and a duplicate declaration
made the handler throw the moment it ran. Nothing that only reads the page can
see that.

These gates lift the real functions out of the rendered pages and run them under
node, with only the boundary stubbed (fetch, FormData, confirm, the DOM). What
they assert is what the code sends and what it shows the user, including every
failure path, because a control that only works when everything goes right is
the ordinary way for a UI to look wired and not be.

Each runs on a throwaway database with fake platform clients, so none of them
touches a real insights.db or a real account.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import (  # noqa: E402
    isolated_app, install_fake_clients, connect, check,
)


def node_path():
    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    return node


def inline_scripts(html):
    """The inline scripts of a rendered page, as source strings."""
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


def rendered_compose():
    directory, database, web, publisher, client = isolated_app()
    install_fake_clients(publisher, web)
    web._maybe_attach_link_image = lambda *a, **k: None
    work = connect(database, "linkedin", "li-work", "Work")
    studio = connect(database, "linkedin", "li-studio", "Studio")
    client.post("/compose/post/create", data={
        "targets": [f"linkedin:{work}", f"linkedin:{studio}"], "content": "Script gate card"})
    return "\n".join(inline_scripts(client.get("/compose").get_data(as_text=True)))


def rendered_accounts():
    directory, database, web, publisher, client = isolated_app()
    install_fake_clients(publisher, web)
    connect(database, "linkedin", "li-work", "Work")
    connect(database, "linkedin", "li-studio", "Studio")
    page = client.get("/accounts")
    check(page.status_code == 200, f"the accounts page did not render ({page.status_code})")
    scripts = inline_scripts(page.get_data(as_text=True))
    check(scripts, "the accounts page rendered no inline scripts")
    return scripts


def lift(source, names):
    """The source of the named top-level functions, taken from the page as is."""
    lifted = []
    for name in names:
        found = re.findall(rf"^(?:async\s+)?function {name}\(", source, re.M)
        check(len(found) == 1, f"{name} is declared {len(found)} times on the page, expected once")
        match = re.search(rf"^(?:async\s+)?function {name}\(.*?^\}}\n", source, re.M | re.S)
        check(match, f"could not lift {name}'s source out of the page")
        lifted.append(match.group(0))
    return "\n".join(lifted)


def run_harness(harness, functions_source):
    node = node_path()
    work = tempfile.mkdtemp(prefix="accounts_js_")
    try:
        fn_path = os.path.join(work, "functions.js")
        harness_path = os.path.join(work, "harness.js")
        with open(fn_path, "w", encoding="utf-8") as handle:
            handle.write(functions_source)
        with open(harness_path, "w", encoding="utf-8") as handle:
            handle.write(harness)
        run = subprocess.run([node, harness_path, fn_path], capture_output=True,
                             text=True, timeout=60)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check(run.returncode == 0, f"the harness crashed:\n{run.stderr.strip()[:800]}")
    return json.loads(run.stdout.strip().splitlines()[-1])


def fields(call):
    """A recorded request's form fields as a dict of lists."""
    out = {}
    for key, value in call["fields"]:
        out.setdefault(key, []).append(value)
    return out


# ---------------------------------------------------------------------------
# compose: the cross-posting controls on the Compose page
# ---------------------------------------------------------------------------

COMPOSE_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

class FakeFormData {
  constructor() { this.entries = []; }
  append(k, v) { this.entries.push([k, String(v)]); }
}

// One page's worth of world. Only the boundary is stubbed; every function under
// test is the page's own source, so a call from one to another is a real call.
function world(opts = {}) {
  const log = { fetch: [], alerts: [], toasts: [], confirms: [], selectors: [],
                replaced: [], cardRefreshes: 0, usedRefreshes: 0 };
  const chips = opts.chips || {};
  const postItem = {
    dataset: opts.dataset || {},
    querySelector(sel) { log.selectors.push(sel); return chips[sel] || null; },
  };
  const ctx = {
    FormData: FakeFormData,
    PLATFORM_LABELS: { linkedin: 'LinkedIn', threads: 'Threads', twitter: 'X',
                       facebook: 'Facebook', instagram: 'Instagram' },
    confirm: (m) => { log.confirms.push(m); return opts.confirm !== false; },
    alert: (m) => log.alerts.push(m),
    showToast: (m, kind) => log.toasts.push([m, kind]),
    platformConnected: () => opts.connected !== false,
    refreshCard: async () => { log.cardRefreshes++; },
    refreshCardUsedState: () => { log.usedRefreshes++; },
    replaceCard: (item, html) => log.replaced.push(html),
    document: {
      createElement: () => ({}),
      querySelectorAll: (sel) => (opts.all || {})[sel] || [],
      querySelector: () => null,
      getElementById: () => null,
    },
    fetch: async (url, init) => {
      log.fetch.push({ url, method: init.method, fields: init.body.entries });
      if (opts.fetchThrows) throw new Error('offline');
      const reply = (opts.replies || []).shift() || { status: 200, body: {} };
      return { ok: reply.status >= 200 && reply.status < 300, status: reply.status,
               json: async () => reply.body };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  const button = {
    innerHTML: 'ORIGINAL', disabled: false,
    closest: (sel) => sel === '.post-item' ? postItem
                     : sel === '.platform-chip' ? (opts.chip || null) : null,
  };
  return { ctx, log, postItem, button };
}

function chipEl() {
  return { appended: [], querySelector: () => null, appendChild(c) { this.appended.push(c); } };
}
const TARGETS = JSON.stringify([
  { post_id: 11, platform: 'linkedin', account_id: 1, account_label: 'Work' },
  { post_id: 12, platform: 'linkedin', account_id: 2, account_label: 'Studio' },
  { post_id: 13, platform: 'threads', account_id: 3, account_label: null },
]);
const CARD = { postId: '11', postIds: '11,12,13', postIndex: '4', targets: TARGETS };
const sel = (platform, account) =>
  `.platform-chip[data-platform="${platform}"]` + (account ? `[data-account-id="${account}"]` : '');

const out = {};
(async () => {
  // --- readers of card and picker state ---------------------------------
  let w = world();
  out.cardTargets = {
    ok: w.ctx.cardTargets({ dataset: { targets: '[{"platform":"linkedin","account_id":1}]' } }),
    missing: w.ctx.cardTargets({ dataset: {} }),
    garbage: w.ctx.cardTargets({ dataset: { targets: '{not json' } }),
    nothing: w.ctx.cardTargets(null),
  };

  w = world({ all: {
    '#platform-checkboxes input:checked': [{ value: 'linkedin' }, { value: 'threads' }],
    '.account-target-checkbox:checked': [
      { value: 'linkedin:3', dataset: { platform: 'linkedin' } },
      { value: 'linkedin:4', dataset: { platform: 'linkedin' } },
      { value: 'twitter:9', dataset: { platform: 'twitter' } },
    ] } });
  out.selectedAccountTargets = w.ctx.selectedAccountTargets();

  w = world({ all: { '#new-post-platforms .platform-chip.is-on': [
    { dataset: { platform: 'linkedin', accountId: '3' } },
    { dataset: { platform: 'linkedin', accountId: '4' } },
    { dataset: { platform: 'threads', accountId: '' } },
  ] } });
  out.newPost = { targets: w.ctx.newPostTargets(), platforms: w.ctx.newPostPlatforms() };

  // --- publish one chip, as the account it stands for --------------------
  let chip = chipEl();
  w = world({ chips: { [sel('linkedin', 4)]: chip },
              replies: [{ status: 200, body: { success: true, permalink: 'https://li.test/4', post_urn: 'https://li.test/4' } }] });
  let returned = await w.ctx.publishCardPlatform(w.postItem, 11, 'linkedin', 4);
  out.publishAs = { returned, fetch: w.log.fetch, selectors: w.log.selectors, linked: chip.appended.length };

  chip = chipEl();
  w = world({ chips: { [sel('threads')]: chip },
              replies: [{ status: 200, body: { success: true, permalink: 'https://th.test/1' } }] });
  returned = await w.ctx.publishCardPlatform(w.postItem, 12, 'threads');
  out.publishDefault = { returned, fetch: w.log.fetch, selectors: w.log.selectors, linked: chip.appended.length };

  // The chip for that account is missing: the link must still land somewhere.
  chip = chipEl();
  w = world({ chips: { [sel('linkedin')]: chip },
              replies: [{ status: 200, body: { success: true, permalink: 'https://li.test/9', post_urn: 'https://li.test/9' } }] });
  await w.ctx.publishCardPlatform(w.postItem, 11, 'linkedin', 9);
  out.publishFallback = { selectors: w.log.selectors, linked: chip.appended.length };

  w = world({ replies: [{ status: 400, body: { success: false, account_label: 'Studio', error: 'over the limit' } }] });
  returned = await w.ctx.publishCardPlatform(w.postItem, 11, 'linkedin', 2);
  out.publishFails = { returned, alerts: w.log.alerts };

  w = world({ connected: false });
  returned = await w.ctx.publishCardPlatform(w.postItem, 11, 'linkedin', 2);
  out.publishDisconnected = { returned, alerts: w.log.alerts, fetches: w.log.fetch.length };

  // --- publish the whole card --------------------------------------------
  const chipWork = chipEl(), chipStudio = chipEl(), chipThreads = chipEl();
  w = world({ dataset: CARD,
              chips: { [sel('linkedin', 1)]: chipWork, [sel('linkedin', 2)]: chipStudio, [sel('threads', 3)]: chipThreads },
              replies: [{ status: 200, body: { success: false, partial: true, published: 2, message: 'Posted to 2 of 3 targets',
                results: [
                  { platform: 'linkedin', account_id: 1, account_label: 'Work', success: true, permalink: 'https://li.test/1', post_urn: 'https://li.test/1' },
                  { platform: 'linkedin', account_id: 2, account_label: 'Studio', success: true, permalink: 'https://li.test/2', post_urn: 'https://li.test/2' },
                  { platform: 'threads', account_id: 3, account_label: 'Threads', success: false, error: 'rate limit' },
                ] } }] });
  await w.ctx.postNowAllFromCard(w.button);
  out.publishAll = {
    confirm: w.log.confirms, fetch: w.log.fetch, alerts: w.log.alerts, toasts: w.log.toasts,
    linkedWork: chipWork.appended.length, linkedStudio: chipStudio.appended.length,
    linkedThreads: chipThreads.appended.length,
    button: { html: w.button.innerHTML, disabled: w.button.disabled }, usedRefreshes: w.log.usedRefreshes,
  };

  w = world({ dataset: CARD, confirm: false });
  await w.ctx.postNowAllFromCard(w.button);
  out.publishAllCancelled = { fetches: w.log.fetch.length, html: w.button.innerHTML, disabled: w.button.disabled };

  w = world({ dataset: { postId: '11', postIds: '11' } });
  await w.ctx.postNowAllFromCard(w.button);
  out.publishAllNoTargets = { confirms: w.log.confirms.length, fetches: w.log.fetch.length };

  w = world({ dataset: CARD, fetchThrows: true });
  await w.ctx.postNowAllFromCard(w.button);
  out.publishAllOffline = { toasts: w.log.toasts, html: w.button.innerHTML, disabled: w.button.disabled };

  w = world({ dataset: CARD, replies: [{ status: 500, body: { error: 'boom' } }] });
  await w.ctx.postNowAllFromCard(w.button);
  out.publishAllServerError = { toasts: w.log.toasts, html: w.button.innerHTML, disabled: w.button.disabled };

  w = world({ dataset: CARD, replies: [{ status: 200, body: { success: false, published: 0, message: 'Could not post to any target',
    results: [{ platform: 'linkedin', account_id: 1, account_label: 'Work', success: false, error: 'expired' }] } }] });
  await w.ctx.postNowAllFromCard(w.button);
  out.publishAllNone = { toasts: w.log.toasts, alerts: w.log.alerts };

  // --- queue the whole card ----------------------------------------------
  w = world({ dataset: CARD, replies: [{ status: 200, body: { success: true, partial: true, message: 'Queued 2 of 3 targets',
    queued: [{ post_id: 11 }, { post_id: 12 }], skipped: [{ target: 'Threads', error: 'No available time slots for Threads' }] } }] });
  await w.ctx.queueAllFromCard(w.button);
  out.queueAll = { fetch: w.log.fetch, alerts: w.log.alerts, toasts: w.log.toasts,
                   cardRefreshes: w.log.cardRefreshes, html: w.button.innerHTML, disabled: w.button.disabled };

  w = world({ dataset: CARD, replies: [{ status: 200, body: { success: true, partial: false, message: 'Queued for 3 targets',
    queued: [{ post_id: 11 }, { post_id: 12 }, { post_id: 13 }], skipped: [] } }] });
  await w.ctx.queueAllFromCard(w.button);
  out.queueAllClean = { toasts: w.log.toasts, alerts: w.log.alerts };

  // Nothing could be queued: the server answers 400 with an empty list. The
  // user must be told nothing was queued, not that something was.
  w = world({ dataset: CARD, replies: [{ status: 400, body: { success: false, queued: [],
    skipped: [{ target: 'Studio', error: 'No available time slots for Studio' }],
    error: 'No available time slots for Studio' } }] });
  await w.ctx.queueAllFromCard(w.button);
  out.queueNone = { toasts: w.log.toasts, alerts: w.log.alerts, html: w.button.innerHTML, disabled: w.button.disabled };

  w = world({ dataset: CARD, replies: [{ status: 404, body: { error: 'Post not found' } }] });
  await w.ctx.queueAllFromCard(w.button);
  out.queueMissing = { toasts: w.log.toasts, html: w.button.innerHTML, disabled: w.button.disabled };

  w = world({ dataset: CARD, fetchThrows: true });
  await w.ctx.queueAllFromCard(w.button);
  out.queueOffline = { toasts: w.log.toasts, html: w.button.innerHTML, disabled: w.button.disabled };

  // --- tick and untick a chip --------------------------------------------
  const tickChip = (on) => ({ dataset: { accountId: '4', platformPostId: '12' },
                              classList: { contains: () => on } });
  w = world({ dataset: CARD, chip: tickChip(false),
              replies: [{ status: 200, body: { success: true, html: '<div>card</div>', account_label: 'Studio', post_id: 99 } }] });
  await w.ctx.togglePostPlatform(w.button, 'linkedin');
  out.tickAdd = { fetch: w.log.fetch, replaced: w.log.replaced, toasts: w.log.toasts };

  w = world({ dataset: CARD, chip: tickChip(true),
              replies: [{ status: 200, body: { success: true, html: '', account_label: 'Studio' } }] });
  await w.ctx.togglePostPlatform(w.button, 'linkedin');
  out.tickRemove = { fetch: w.log.fetch, toasts: w.log.toasts };

  w = world({ dataset: CARD, chip: { dataset: { accountId: '' }, classList: { contains: () => false } },
              replies: [{ status: 200, body: { success: true, html: '<div/>' } }] });
  await w.ctx.togglePostPlatform(w.button, 'threads');
  out.tickNoAccount = { fetch: w.log.fetch };

  w = world({ dataset: CARD, chip: tickChip(true),
              replies: [{ status: 409, body: { needs_confirm: true, queued: true, scheduled_for: 'Friday' } },
                        { status: 200, body: { success: true, html: '', account_label: 'Studio' } }] });
  await w.ctx.togglePostPlatform(w.button, 'linkedin');
  out.tickConfirmed = { fetch: w.log.fetch, confirms: w.log.confirms };

  w = world({ dataset: CARD, chip: tickChip(true), confirm: false,
              replies: [{ status: 409, body: { needs_confirm: true, queued: true } }] });
  await w.ctx.togglePostPlatform(w.button, 'linkedin');
  out.tickDeclined = { fetches: w.log.fetch.length, disabled: w.button.disabled, replaced: w.log.replaced.length };

  w = world({ dataset: CARD, chip: tickChip(false), replies: [{ status: 400, body: { error: 'That LinkedIn account is not connected' } }] });
  await w.ctx.togglePostPlatform(w.button, 'linkedin');
  out.tickRefused = { toasts: w.log.toasts, disabled: w.button.disabled, replaced: w.log.replaced.length };

  console.log(JSON.stringify(out));
})();
"""

COMPOSE_FUNCTIONS = (
    "cardTargets", "cardPostIds", "selectedAccountTargets", "newPostTargets",
    "newPostPlatforms", "publishCardPlatform", "postNowAllFromCard",
    "queueAllFromCard", "markChipPosted", "postedUrlFrom", "togglePostPlatform",
    # the requests the card's publish-all and queue-all share with Find & Replace
    "publishAllConfirmText", "publishCardToAll", "queueCardToAll",
)


def section_compose():
    source = lift(rendered_compose(), COMPOSE_FUNCTIONS)
    r = run_harness(COMPOSE_HARNESS, source)

    # --- readers ------------------------------------------------------------
    t = r["cardTargets"]
    check(t["ok"] == [{"platform": "linkedin", "account_id": 1}],
          f"cardTargets did not read the card's targets: {t['ok']}")
    check(t["missing"] == [] and t["garbage"] == [] and t["nothing"] == [],
          f"cardTargets must return [] for a missing, malformed or absent card: {t}")

    check(r["selectedAccountTargets"] == ["linkedin:3", "linkedin:4"],
          f"the generator must send only accounts on ticked platforms, as platform:id: "
          f"{r['selectedAccountTargets']}")
    check(r["newPost"]["targets"] == ["linkedin:3", "linkedin:4", "threads"],
          f"the new-post composer must send platform:id per account and a bare platform "
          f"for the rest: {r['newPost']['targets']}")
    check(r["newPost"]["platforms"] == ["linkedin", "threads"],
          f"two accounts on one platform must count as one platform: {r['newPost']['platforms']}")

    # --- publish one chip -----------------------------------------------------
    p = r["publishAs"]
    check(p["returned"] is True, "publishing a chip that succeeded did not report success")
    check(p["fetch"][0]["url"] == "/compose/post/11/linkedin"
          and fields(p["fetch"][0]).get("account_id") == ["4"],
          f"publishing a chip must name the account it stands for: {p['fetch']}")
    check(p["selectors"][0] == '.platform-chip[data-platform="linkedin"][data-account-id="4"]'
          and p["linked"] == 1,
          f"the posted link must land on the chip for that account: {p}")

    p = r["publishDefault"]
    check("account_id" not in fields(p["fetch"][0]),
          f"a chip with no account must not invent one: {p['fetch']}")
    check(p["selectors"][0] == '.platform-chip[data-platform="threads"]' and p["linked"] == 1,
          f"a chip with no account should be found by platform alone: {p}")

    check(r["publishFallback"]["linked"] == 1
          and len(r["publishFallback"]["selectors"]) == 2,
          f"when no chip matches the account the link must fall back to the platform's chip: "
          f"{r['publishFallback']}")

    p = r["publishFails"]
    check(p["returned"] is False and p["alerts"] == ["Studio: over the limit"],
          f"a failed publish must name the account and the reason: {p}")
    p = r["publishDisconnected"]
    check(p["returned"] is False and p["fetches"] == 0 and "connect LinkedIn" in p["alerts"][0],
          f"a disconnected platform must be refused before any request: {p}")

    # --- publish the whole card -------------------------------------------------
    a = r["publishAll"]
    check("Work, Studio, Threads" in a["confirm"][0],
          f"the confirmation must list every account it will post to: {a['confirm']}")
    check(len(a["fetch"]) == 1 and a["fetch"][0]["url"] == "/compose/post/11/publish"
          and fields(a["fetch"][0]).get("post_ids") == ["11,12,13"],
          f"publish-all must be one request naming every row on the card: {a['fetch']}")
    check(a["linkedWork"] == 1 and a["linkedStudio"] == 1 and a["linkedThreads"] == 0,
          f"only the accounts that posted should get a link, each on its own chip: {a}")
    check(a["alerts"] == ["Threads: rate limit"],
          f"a failed target must be reported by account and reason: {a['alerts']}")
    check(a["toasts"] == [["Posted to 2 of 3 targets", "warning"]],
          f"a partial publish must say so, as a warning: {a['toasts']}")
    check(a["button"] == {"html": "ORIGINAL", "disabled": False} and a["usedRefreshes"] == 1,
          f"the button must come back and the used state refresh: {a}")

    c = r["publishAllCancelled"]
    check(c["fetches"] == 0 and c["html"] == "ORIGINAL" and c["disabled"] is False,
          f"declining the confirmation must do nothing at all: {c}")
    check(r["publishAllNoTargets"] == {"confirms": 0, "fetches": 0},
          f"a card with no targets must not prompt or publish: {r['publishAllNoTargets']}")
    for name in ("publishAllOffline", "publishAllServerError"):
        e = r[name]
        check(e["toasts"] and e["toasts"][0][1] == "error"
              and e["toasts"][0][0].startswith("Could not publish"),
              f"{name}: the user must be told the publish failed: {e['toasts']}")
        check(e["html"] == "ORIGINAL" and e["disabled"] is False,
              f"{name}: the button was left stuck in its busy state: {e}")
    n = r["publishAllNone"]
    check(n["toasts"] == [["Could not post to any target", "error"]] and n["alerts"] == ["Work: expired"],
          f"a publish that reached nothing must read as an error: {n}")

    # --- queue the whole card -----------------------------------------------------
    q = r["queueAll"]
    check(q["fetch"][0]["url"] == "/compose/post/11/queue"
          and fields(q["fetch"][0]).get("all") == ["1"]
          and fields(q["fetch"][0]).get("post_ids") == ["11,12,13"],
          f"queue-all must ask the server for a card-wide queue naming every row: {q['fetch']}")
    check(q["alerts"] == ["Threads: No available time slots for Threads"],
          f"a target that could not be queued must be reported: {q['alerts']}")
    check(q["toasts"] == [["Queued 2 of 3 targets", "warning"]],
          f"a partial queue must say so, as a warning: {q['toasts']}")
    check(q["cardRefreshes"] == 1 and q["html"] == "ORIGINAL" and q["disabled"] is False,
          f"the card must redraw and the button come back: {q}")
    check(r["queueAllClean"]["toasts"] == [["Queued for 3 targets", "success"]]
          and r["queueAllClean"]["alerts"] == [],
          f"a full queue must read as success with nothing to apologise for: {r['queueAllClean']}")

    n = r["queueNone"]
    check(n["toasts"] == [["No available time slots for Studio", "error"]],
          f"when nothing could be queued the user must be told so, as an error, "
          f"not that something was queued: {n['toasts']}")
    check(n["alerts"] == ["Studio: No available time slots for Studio"]
          and n["html"] == "ORIGINAL" and n["disabled"] is False,
          f"the reason must be shown and the button restored: {n}")
    for name in ("queueMissing", "queueOffline"):
        e = r[name]
        check(e["toasts"] and e["toasts"][0][1] == "error"
              and e["toasts"][0][0].startswith("Could not queue"),
              f"{name}: the user must be told the queue failed: {e['toasts']}")
        check(e["html"] == "ORIGINAL" and e["disabled"] is False,
              f"{name}: the button was left stuck in its busy state: {e}")

    # --- tick and untick ------------------------------------------------------------
    a = r["tickAdd"]
    f = fields(a["fetch"][0])
    check(a["fetch"][0]["url"] == "/compose/post/11/platform"
          and f.get("platform") == ["linkedin"] and f.get("account_id") == ["4"]
          and f.get("action") == ["add"] and f.get("post_ids") == ["11,12,13"],
          f"ticking a chip must name its platform AND account, or the wrong chip lights up: {a['fetch']}")
    check(a["replaced"] == ["<div>card</div>"] and a["toasts"] == [["Added to Studio", "success"]],
          f"a ticked chip must redraw the card and name the account: {a}")

    a = r["tickRemove"]
    f = fields(a["fetch"][0])
    check(a["fetch"][0]["url"] == "/compose/post/12/platform" and f.get("action") == ["remove"]
          and f.get("account_id") == ["4"],
          f"unticking must go to the row that chip owns, naming its account: {a['fetch']}")
    check(a["toasts"] == [["Removed from Studio", "success"]],
          f"an unticked chip must name the account: {a['toasts']}")

    check("account_id" not in fields(r["tickNoAccount"]["fetch"][0]),
          "a chip with no account must not send one")

    a = r["tickConfirmed"]
    check(len(a["fetch"]) == 2 and fields(a["fetch"][1]).get("force") == ["1"]
          and fields(a["fetch"][1]).get("account_id") == ["4"] and len(a["confirms"]) == 1,
          f"removing a queued target must confirm, then resend as forced for the same account: {a}")
    a = r["tickDeclined"]
    check(a["fetches"] == 1 and a["disabled"] is False and a["replaced"] == 0,
          f"declining the confirmation must leave the card alone: {a}")
    a = r["tickRefused"]
    check(a["toasts"] == [["That LinkedIn account is not connected", "error"]]
          and a["disabled"] is False and a["replaced"] == 0,
          f"a refused tick must say why and leave the chip usable: {a}")

    print("Compose cross-posting: every control sends the account it stands for, reports "
          "each target honestly, and comes back after a failure")
    print("COMPOSE_ACCOUNTS_JS_OK")


# ---------------------------------------------------------------------------
# accounts: the accounts page's own controls
# ---------------------------------------------------------------------------

ACCOUNTS_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

class FakeFormData {
  constructor() { this.entries = []; }
  append(k, v) { this.entries.push([k, String(v)]); }
}
function el(extra) {
  return Object.assign({ children: [], hidden: false, disabled: false, removed: false,
    textContent: '', className: '', innerHTML: '',
    remove() { this.removed = true; },
    appendChild(c) { this.children.push(c); } }, extra || {});
}

function world(opts = {}) {
  const log = { fetch: [], prompts: [], confirms: [], reloads: 0, timers: [] };
  const flash = el();
  const ctx = {
    FormData: FakeFormData,
    document: { getElementById: (id) => id === 'accounts-flash' ? flash : null,
                createElement: () => el() },
    window: { location: { reload: () => { log.reloads++; } } },
    setTimeout: (fn, ms) => { log.timers.push(ms); fn(); },
    prompt: (m, current) => { log.prompts.push(current); return opts.promptReply; },
    confirm: (m) => { log.confirms.push(m); return opts.confirm !== false; },
    fetch: async (url, init) => {
      log.fetch.push({ url, method: init.method, fields: init.body.entries });
      const reply = opts.reply || { status: 200, body: {} };
      return { ok: reply.status >= 200 && reply.status < 300, status: reply.status,
               json: async () => { if (reply.notJson) throw new Error('not json'); return reply.body; } };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  return { ctx, log, flash };
}

// A platform card with two accounts, the first of which is the default.
function card() {
  const oldBadge = el({ className: 'badge default-badge', textContent: 'Default' });
  const nameBox = el();
  const label = el({ textContent: 'Work' });
  const thisBtn = el();
  const otherBtn = el({ hidden: true });
  const row = el();
  row.querySelector = (sel) => sel === '.account-name' ? nameBox : sel === '.account-label-text' ? label : null;
  const platform = el();
  platform.querySelectorAll = (sel) => sel === '.default-badge' ? [oldBadge]
                                     : sel === '.set-default-btn' ? [thisBtn, otherBtn] : [];
  thisBtn.closest = (sel) => sel === '.account-platform' ? platform : sel === '.account-row' ? row : null;
  return { thisBtn, otherBtn, oldBadge, nameBox, label, row, platform };
}

const out = {};
(async () => {
  // --- make default --------------------------------------------------------
  let w = world({ reply: { status: 200, body: { success: true, message: 'Studio is now the default LinkedIn account' } } });
  let c = card();
  await w.ctx.makeAccountDefault(7, c.thisBtn);
  out.defaultOk = { fetch: w.log.fetch, flash: w.flash.innerHTML, oldRemoved: c.oldBadge.removed,
    newBadge: c.nameBox.children.map(b => [b.textContent, b.className]), thisHidden: c.thisBtn.hidden,
    otherHidden: c.otherBtn.hidden };

  w = world({ reply: { status: 404, body: { error: 'Account not found' } } });
  c = card();
  await w.ctx.makeAccountDefault(7, c.thisBtn);
  out.defaultFails = { flash: w.flash.innerHTML, oldRemoved: c.oldBadge.removed,
    badges: c.nameBox.children.length, disabled: c.thisBtn.disabled };

  // --- rename ----------------------------------------------------------------
  w = world({ promptReply: 'Studio', reply: { status: 200, body: { success: true, label: 'Studio' } } });
  c = card();
  await w.ctx.renameAccount(7, c.thisBtn);
  out.renameOk = { fetch: w.log.fetch, prompted: w.log.prompts, label: c.label.textContent,
    flash: w.flash.innerHTML, disabled: c.thisBtn.disabled };

  w = world({ promptReply: null });
  c = card();
  await w.ctx.renameAccount(7, c.thisBtn);
  out.renameCancelled = { fetches: w.log.fetch.length, label: c.label.textContent };

  w = world({ promptReply: 'x'.repeat(80), reply: { status: 400, body: { error: 'Keep the name under 60 characters' } } });
  c = card();
  await w.ctx.renameAccount(7, c.thisBtn);
  out.renameRefused = { flash: w.flash.innerHTML, label: c.label.textContent, disabled: c.thisBtn.disabled };

  // --- disconnect ------------------------------------------------------------
  const disconnect = async (reply, opts = {}) => {
    const wd = world(Object.assign({ reply }, opts));
    const cd = card();
    await wd.ctx.disconnectAccount(7, cd.thisBtn);
    return { fetch: wd.log.fetch, confirms: wd.log.confirms, flash: wd.flash.innerHTML,
             reloads: wd.log.reloads, disabled: cd.thisBtn.disabled };
  };
  out.disconnectQueued = await disconnect({ status: 200, body: { success: true, message: 'Disconnected Work', unqueued: 2 } });
  out.disconnectOne = await disconnect({ status: 200, body: { success: true, message: 'Disconnected Work', unqueued: 1 } });
  out.disconnectPlain = await disconnect({ status: 200, body: { success: true, message: 'Disconnected Work', unqueued: 0 } });
  out.disconnectCancelled = await disconnect({ status: 200, body: {} }, { confirm: false });
  out.disconnectFails = await disconnect({ status: 404, body: { error: 'Account not found' } });
  out.disconnectNotJson = await disconnect({ status: 500, notJson: true, body: {} });

  console.log(JSON.stringify(out));
})();
"""

ACCOUNTS_FUNCTIONS = (
    "accountsFlash", "accountsPost", "makeAccountDefault", "renameAccount",
    "disconnectAccount",
)


def section_accounts():
    source = lift("\n".join(rendered_accounts()), ACCOUNTS_FUNCTIONS)
    r = run_harness(ACCOUNTS_HARNESS, source)

    # --- make default ---------------------------------------------------------------
    d = r["defaultOk"]
    check(d["fetch"][0]["url"] == "/accounts/7/default" and d["fetch"][0]["method"] == "POST",
          f"make-default must POST to that account's endpoint: {d['fetch']}")
    check(d["oldRemoved"] and d["newBadge"] == [["Default", "badge bg-success default-badge"]],
          f"the Default badge must move to the new account, not duplicate: {d}")
    check(d["thisHidden"] is True and d["otherHidden"] is False,
          f"the new default's button must hide and the others' show: {d}")
    check("alert-success" in d["flash"] and "now the default LinkedIn account" in d["flash"],
          f"the user must be told it worked: {d['flash']}")

    d = r["defaultFails"]
    check("alert-danger" in d["flash"] and "Account not found" in d["flash"],
          f"a refused make-default must say why: {d['flash']}")
    check(not d["oldRemoved"] and d["badges"] == 0 and d["disabled"] is False,
          f"a refused make-default must leave the page as it was and the button usable: {d}")

    # --- rename ------------------------------------------------------------------------
    d = r["renameOk"]
    fld = fields(d["fetch"][0])
    check(d["fetch"][0]["url"] == "/accounts/7/label" and fld.get("label") == ["Studio"],
          f"rename must POST the new name to that account: {d['fetch']}")
    check(d["prompted"] == ["Work"], f"rename must offer the current name: {d['prompted']}")
    check(d["label"] == "Studio" and "Renamed" in d["flash"] and d["disabled"] is False,
          f"rename must show the saved name and free the button: {d}")
    d = r["renameCancelled"]
    check(d["fetches"] == 0 and d["label"] == "Work",
          f"cancelling the prompt must change nothing: {d}")
    d = r["renameRefused"]
    check("alert-danger" in d["flash"] and "under 60 characters" in d["flash"]
          and d["label"] == "Work" and d["disabled"] is False,
          f"a refused rename must say why and keep the old name: {d}")

    # --- disconnect ---------------------------------------------------------------------
    d = r["disconnectQueued"]
    check(d["fetch"][0]["url"] == "/accounts/7/disconnect" and d["fetch"][0]["method"] == "POST",
          f"disconnect must POST to that account's endpoint: {d['fetch']}")
    check("Work" in d["confirms"][0] and "Other accounts on this platform keep working" in d["confirms"][0],
          f"the confirmation must name the account and say the others are safe: {d['confirms']}")
    check("Disconnected Work. 2 queued posts removed." in d["flash"] and d["reloads"] == 1,
          f"disconnect must say how many queued posts went and reload: {d}")
    check("1 queued post removed" in r["disconnectOne"]["flash"]
          and "1 queued posts" not in r["disconnectOne"]["flash"],
          f"one queued post is singular: {r['disconnectOne']['flash']}")
    check("queued post" not in r["disconnectPlain"]["flash"]
          and "Disconnected Work" in r["disconnectPlain"]["flash"],
          f"with nothing queued the message must not mention the queue: {r['disconnectPlain']['flash']}")
    d = r["disconnectCancelled"]
    check(d["fetch"] == [] and d["reloads"] == 0,
          f"declining the confirmation must do nothing at all: {d}")
    for name, expect in (("disconnectFails", "Account not found"),
                         ("disconnectNotJson", "Request failed (500)")):
        d = r[name]
        check("alert-danger" in d["flash"] and expect in d["flash"]
              and d["disabled"] is False and d["reloads"] == 0,
              f"{name}: a failed disconnect must say why, not reload, and free the button: {d}")

    print("accounts page: make default, rename and disconnect each send the right request "
          "and leave the page correct on every failure")
    print("ACCOUNTS_PAGE_JS_OK")


# ---------------------------------------------------------------------------
# syntax: the accounts page's scripts parse
# ---------------------------------------------------------------------------

def section_syntax():
    node = node_path()
    scripts = rendered_accounts()
    work = tempfile.mkdtemp(prefix="accounts_syntax_")
    try:
        for index, body in enumerate(scripts):
            path = os.path.join(work, f"script_{index}.js")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body)
            run = subprocess.run([node, "--check", path], capture_output=True, text=True)
            check(run.returncode == 0,
                  f"inline script {index} on the accounts page does not parse:\n{run.stderr.strip()[:500]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"all {len(scripts)} inline scripts on the accounts page parse")
    print("ACCOUNTS_JS_SYNTAX_OK")


SECTIONS = {"compose": section_compose, "accounts": section_accounts, "syntax": section_syntax}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_accounts_js.py {'|'.join(SECTIONS)}")
        sys.exit(2)
    SECTIONS[name]()
