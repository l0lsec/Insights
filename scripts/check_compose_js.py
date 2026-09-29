"""Gates for the Compose page's one big script.

    python scripts/check_compose_js.py duplicates   COMPOSE_JS_NO_DUPES_OK
    python scripts/check_compose_js.py refreshcard  REFRESH_CARD_OK

The page's inline scripts share one global scope, so two top-level declarations
of the same function name are not an error: the later one silently replaces the
earlier, and every caller written for the first gets the second. That is how
``refreshCard(postItem)`` ended up throwing for "Queue all accounts". ``duplicates``
fails on any such pair; ``refreshcard`` runs the function itself in node against
a stubbed ``fetch`` and DOM, so what it does is checked and not just how it is
declared.

Both run on a throwaway database with fake platform clients, so neither touches
a real insights.db or a real account.
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
from _accounts_gate import (  # noqa: E402
    isolated_app, install_fake_clients, connect, check,
)


def rendered_scripts():
    """The inline scripts of the real /compose page, as source strings."""
    directory, database, web, publisher, client = isolated_app()
    install_fake_clients(publisher, web)
    web._maybe_attach_link_image = lambda *a, **k: None
    connect(database, "linkedin", "li-work", "Work")
    database.add_standalone_post(
        "manual", "gate", "linkedin", "Script gate card", db_path=database.DB_PATH)
    html = client.get("/compose").get_data(as_text=True)
    blocks = []
    for match in re.finditer(r"<script([^>]*)>(.*?)</script>", html, re.S | re.I):
        attrs, body = match.group(1), match.group(2)
        if "src=" in attrs or not body.strip():
            continue
        kind = re.search(r'type="([^"]+)"', attrs)
        if kind and "javascript" not in kind.group(1) and kind.group(1) != "module":
            continue
        blocks.append(body)
    check(blocks, "the page rendered no inline scripts")
    return blocks


def section_duplicates():
    blocks = rendered_scripts()
    # A declaration at column 0 is top-level; nested helpers are indented.
    names = [name for body in blocks
             for name in re.findall(r"^(?:async\s+)?function\s+(\w+)\s*\(", body, re.M)]
    check(len(names) > 100, f"only {len(names)} top-level functions found: the scan is not reading the page")
    dupes = {n: c for n, c in collections.Counter(names).items() if c > 1}
    check(not dupes, f"declared more than once, so the later one silently wins: {dupes}")
    print(f"{len(names)} top-level functions, each declared once")
    print("COMPOSE_JS_NO_DUPES_OK")


# What refreshCard is run against. Written to a file and run by node, with the
# function's own source lifted from the page so the test cannot drift from it.
HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

function item(dataset) {
  return { dataset, removed: false, remove() { this.removed = true; } };
}

async function run(args, opts) {
  const calls = { fetch: [], replace: [], badge: [] };
  const ctx = {
    FormData: class {
      constructor() { this.entries = []; }
      append(key, value) { this.entries.push([key, String(value)]); }
      get(key) { const hit = this.entries.find(e => e[0] === key); return hit ? hit[1] : null; }
    },
    cardPostIds: (card) => (card.dataset.postIds || '').split(',').filter(Boolean).map(Number),
    replaceCard: (card, html) => calls.replace.push(html),
    adjustPostCountBadge: (delta) => calls.badge.push(delta),
    fetch: async (url, init) => {
      calls.fetch.push({ url, post_ids: init.body.get('post_ids'), index: init.body.get('display_index') });
      if (opts.reject) throw new Error('offline');
      return { ok: opts.ok !== false, json: async () => opts.body };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  let threw = null;
  try { await ctx.refreshCard(...args(ctx)); } catch (e) { threw = e.message; }
  return { calls, threw };
}

const results = {};
(async () => {
  // 1. The one-argument form: redraw from the ids the page already holds.
  let card = item({ postId: '4', postIds: '4,5,6', postIndex: '12' });
  results.oneArg = await run(() => [card], { body: { success: true, html: '<div>card</div>' } });
  results.oneArg.removed = card.removed;

  // 2. The two-argument form: redraw from just the rows that survived a delete.
  card = item({ postId: '4', postIds: '4,5,6' });
  results.remaining = await run(() => [card, [5, 6]], { body: { success: true, html: '<div>left</div>' } });
  results.remaining.removed = card.removed;

  // 3. An explicitly empty list means the card is gone: no request, card removed.
  card = item({ postId: '4', postIds: '4,5,6' });
  results.empty = await run(() => [card, []], { body: { success: true, html: '' } });
  results.empty.removed = card.removed;

  // 4. A card the page holds no ids for still redraws from its own id, and is
  //    NOT taken for an emptied card.
  card = item({ postId: '9' });
  results.noIds = await run(() => [card], { body: { success: true, html: '<div>x</div>' } });
  results.noIds.removed = card.removed;

  // 5. A network failure or a non-JSON reply leaves the card alone and does not throw.
  card = item({ postId: '4', postIds: '4,5' });
  results.offline = await run(() => [card], { reject: true });
  results.offline.removed = card.removed;
  card = item({ postId: '4', postIds: '4,5' });
  results.serverError = await run(() => [card], { ok: false, body: { error: 'nope' } });
  results.serverError.removed = card.removed;

  // 6. Nothing to redraw.
  results.noCard = await run(() => [null], { body: { success: true, html: 'x' } });

  console.log(JSON.stringify(results));
})();
"""


def section_refreshcard():
    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    source = "\n".join(rendered_scripts())

    declared = re.findall(r"^async function refreshCard\(", source, re.M)
    check(len(declared) == 1, f"refreshCard is declared {len(declared)} times, expected exactly once")
    match = re.search(r"^async function refreshCard\(.*?^\}\n", source, re.M | re.S)
    check(match, "could not lift refreshCard's source out of the page")

    work = tempfile.mkdtemp(prefix="refresh_card_")
    try:
        fn_path = os.path.join(work, "refresh_card.js")
        harness_path = os.path.join(work, "harness.js")
        with open(fn_path, "w", encoding="utf-8") as handle:
            handle.write(match.group(0))
        with open(harness_path, "w", encoding="utf-8") as handle:
            handle.write(HARNESS)
        run = subprocess.run([node, harness_path, fn_path], capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check(run.returncode == 0, f"the harness crashed:\n{run.stderr.strip()[:600]}")
    results = json.loads(run.stdout.strip().splitlines()[-1])

    one = results["oneArg"]
    check(one["threw"] is None, f"refreshCard(card) threw: {one['threw']}")
    check(len(one["calls"]["fetch"]) == 1, "refreshCard(card) did not ask the server for the card")
    fetch = one["calls"]["fetch"][0]
    check(fetch["url"] == "/compose/post/4/card" and fetch["post_ids"] == "4,5,6" and fetch["index"] == "12",
          f"refreshCard(card) asked for the wrong card: {fetch}")
    check(one["calls"]["replace"] == ["<div>card</div>"] and not one["removed"],
          "refreshCard(card) did not swap in the redrawn card")

    left = results["remaining"]
    check(left["threw"] is None, f"refreshCard(card, ids) threw: {left['threw']}")
    fetch = left["calls"]["fetch"][0]
    check(fetch["url"] == "/compose/post/5/card" and fetch["post_ids"] == "5,6",
          f"refreshCard(card, ids) should redraw from the surviving rows only: {fetch}")
    check(left["calls"]["replace"] == ["<div>left</div>"], "refreshCard(card, ids) did not swap in the card")

    empty = results["empty"]
    check(empty["threw"] is None and empty["removed"] and empty["calls"]["badge"] == [-1]
          and not empty["calls"]["fetch"],
          f"an emptied card should be removed with no request and the count lowered: {empty}")

    no_ids = results["noIds"]
    check(no_ids["threw"] is None and not no_ids["removed"]
          and no_ids["calls"]["fetch"] and no_ids["calls"]["fetch"][0]["url"] == "/compose/post/9/card",
          f"a card with no recorded ids must redraw from its own id, not be removed: {no_ids}")

    for name in ("offline", "serverError"):
        failed = results[name]
        check(failed["threw"] is None and not failed["removed"] and not failed["calls"]["replace"],
              f"a failed redraw ({name}) must leave the card alone without throwing: {failed}")

    check(results["noCard"]["threw"] is None and not results["noCard"]["calls"]["fetch"],
          "refreshCard(null) should do nothing")

    print("refreshCard redraws from ids the page holds or from surviving rows, removes an emptied card, "
          "and survives a failed request")
    print("REFRESH_CARD_OK")


SECTIONS = {"duplicates": section_duplicates, "refreshcard": section_refreshcard}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_compose_js.py {'|'.join(SECTIONS)}")
        sys.exit(2)
    SECTIONS[name]()
