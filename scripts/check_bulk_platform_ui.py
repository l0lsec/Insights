"""Gates for the Compose page side of bulk "Add platform".

    python scripts/check_bulk_platform_ui.py wiring   BULK_PLATFORM_UI_OK
    python scripts/check_bulk_platform_ui.py syntax   COMPOSE_JS_SYNTAX_OK

``wiring`` renders the real /compose page and checks the controls are present
AND bound to the endpoint that does the work, not inert markup. ``syntax`` runs
every inline script through ``node --check``: the Compose page is one very large
script, and a single stray brace silently takes every button on it down.

Both run on a throwaway database with fake platform clients, so neither touches
a real insights.db or a real account.
"""

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

# Every function the feature adds. Each must be declared exactly once: a second
# declaration of the same name silently replaces the first in a classic script.
FEATURE_FUNCTIONS = (
    "bulkPlatformCount", "bulkPlatformChips", "bulkPlatformChosen", "bulkPlatformPresence",
    "showBulkPlatformModal", "toggleBulkPlatformTarget", "updateBulkPlatformSummary",
    "setBulkPlatformBusy", "postBulkPlatform", "newBulkPlatformTotals",
    "mergeBulkPlatformTotals", "applyBulkPlatformCards", "applyBulkPlatform",
    "renderBulkPlatformResult",
)


def rig(with_posts=True):
    directory, database, web, publisher, client = isolated_app()
    install_fake_clients(publisher, web)
    web._maybe_attach_link_image = lambda *a, **k: None
    ids = {}
    ids["work"] = connect(database, "linkedin", "li-work", "Work")
    ids["studio"] = connect(database, "linkedin", "li-studio", "Studio")
    ids["threads"] = connect(database, "threads", "th-1", "brandco")
    # Twitter is deliberately left unconnected to prove the modal says so.
    if with_posts:
        for i in range(2):
            database.add_standalone_post(
                "manual", "gate", "linkedin", f"Wiring card {i}", db_path=database.DB_PATH)
    return database, web, client, ids


def region(html, start_marker, end_marker):
    start = html.index(start_marker)
    return html[start:html.index(end_marker, start)]


def function_body(html, name):
    """The source of ``function name(...) {...}`` up to the next top-level function."""
    match = re.search(rf"(?:async\s+)?function\s+{name}\s*\(", html)
    check(match, f"the page does not define {name}()")
    rest = html[match.start():]
    nxt = re.search(r"\n(?:async\s+)?function\s+\w+\s*\(", rest[1:])
    return rest[: nxt.start() + 1] if nxt else rest


def section_wiring():
    database, web, client, ids = rig()
    response = client.get("/compose")
    check(response.status_code == 200, f"/compose did not render ({response.status_code})")
    html = response.get_data(as_text=True)

    # --- the toolbar button exists, starts hidden, and opens the modal -------
    button = re.search(r'<button[^>]*id="add-platform-selected-btn"[^>]*>', html)
    check(button, "the toolbar has no Add Platform button")
    check("d-none" in button.group(0), "the Add Platform button is visible before anything is selected")
    check("onclick=\"showBulkPlatformModal('add')\"" in button.group(0),
          "the Add Platform button does not open the modal in add mode")
    check('id="add-platform-count"' in html, "the button has no count to update")

    remove_button = re.search(r'<button[^>]*id="remove-platform-selected-btn"[^>]*>', html)
    check(remove_button, "the toolbar has no Remove Platform button")
    check("d-none" in remove_button.group(0), "the Remove Platform button is visible before anything is selected")
    check("onclick=\"showBulkPlatformModal('remove')\"" in remove_button.group(0),
          "the Remove Platform button does not open the modal in remove mode")
    check('id="remove-platform-count"' in html, "the Remove Platform button has no count to update")

    # --- the modal: one chip per publish target, queue option, apply button --
    check('id="bulkPlatformModal"' in html, "the page has no bulk platform modal")
    chips_html = region(html, 'id="bulk-platform-targets"', 'id="bulk-platform-queue"')
    composer_html = region(html, 'id="new-post-platforms"', "<textarea")
    modal_chips = re.findall(r'<span class="platform-chip[^"]*"[^>]*>', chips_html)
    composer_chips = re.findall(r'<span class="platform-chip[^"]*"[^>]*>', composer_html)
    check(len(modal_chips) == len(composer_chips) and len(modal_chips) >= 6,
          f"the modal offers {len(modal_chips)} targets, the composer {len(composer_chips)}: "
          "a connected account is missing from one of them")
    labels = re.findall(r'data-label="([^"]*)"', chips_html)
    check("Work" in labels and "Studio" in labels,
          f"both LinkedIn accounts should be offered by name, got {labels}")
    account_ids = re.findall(r'data-account-id="(\d*)"', chips_html)
    check(str(ids["work"]) in account_ids and str(ids["studio"]) in account_ids
          and len(set(a for a in account_ids if a)) == len([a for a in account_ids if a]),
          f"the modal's chips do not carry distinct account ids: {account_ids}")
    twitter = next((c for c in modal_chips if 'data-platform="twitter"' in c), "")
    check('data-connected="false"' in twitter, "an unconnected platform is not marked as such")
    check('data-connected="true"' in next(c for c in modal_chips if 'data-platform="threads"' in c),
          "a connected platform is marked unconnected")
    check(re.search(r'id="bulk-platform-queue"[^>]*checked', html),
          "the queue option should start ticked: queueing is what was asked for")
    modal_tag = re.search(r'<div[^>]*id="bulkPlatformModal"[^>]*>', html).group(0)
    check('data-mode="add"' in modal_tag, "the modal does not start in add mode")
    add_only = region(html, 'class="form-check mb-2 bulk-only-add"', 'id="bulk-platform-force"')
    check('id="bulk-platform-queue"' in add_only, "the queue option is not confined to add mode")
    check('id="bulk-platform-force"' in region(html, 'class="form-check mb-2 bulk-only-remove"', 'id="bulk-platform-summary"'),
          "the queued-or-published option is not confined to remove mode")
    check(not re.search(r'id="bulk-platform-force"[^>]*checked', html),
          "removing queued or published posts must be opt-in, not ticked by default")
    for rule in ('#bulkPlatformModal[data-mode="add"] .bulk-only-remove',
                 '#bulkPlatformModal[data-mode="remove"] .bulk-only-add'):
        check(rule in html, f"the stylesheet never hides the other mode's controls ({rule})")
    apply_button = re.search(r'<button[^>]*id="bulk-platform-apply-btn"[^>]*>', html)
    check(apply_button and 'onclick="applyBulkPlatform()"' in apply_button.group(0),
          "the modal's apply button is not bound to applyBulkPlatform()")
    check("disabled" in apply_button.group(0), "apply should be disabled until a platform is ticked")
    check('onclick="toggleBulkPlatformTarget(this)"' in chips_html,
          "the modal's chips are not bound to toggleBulkPlatformTarget()")

    # --- the handlers talk to the real endpoint, and send what it reads ------
    check("/compose/posts/bulk-add-platform" in html, "no handler calls the bulk endpoint")
    apply_src = function_body(html, "applyBulkPlatform")
    for token in ("post_ids:", "filters: selectAllScope.filters", "targets: targets",
                  "queue: queue", "force: force", "render: true", "BULK_PLATFORM_BATCH"):
        check(token in apply_src, f"applyBulkPlatform() never sends `{token}`")
    check("/compose/posts/bulk-remove-platform" in html, "no handler calls the bulk remove endpoint")
    check("This cannot be undone" in apply_src,
          "an across-pages removal is not confirmed before it runs")
    check("selectAllScope" in function_body(html, "bulkPlatformCount"),
          "the count ignores an across-pages selection")

    # --- select mode shows and hides the button with the other bulk buttons ---
    toggle_src = function_body(html, "toggleSelectMode")
    check("add-platform-selected-btn" in toggle_src and "addPlatformBtn.classList.add('d-none')" in toggle_src,
          "leaving select mode does not hide the Add Platform button")
    check("remove-platform-selected-btn" in toggle_src and "removePlatformBtn.classList.add('d-none')" in toggle_src,
          "leaving select mode does not hide the Remove Platform button")
    count_src = function_body(html, "updateSelectedCount")
    check("addPlatformBtn.classList.remove('d-none')" in count_src,
          "a selection does not reveal the Add Platform button")
    check("removePlatformBtn.classList.remove('d-none')" in count_src
          and count_src.count("removePlatformBtn.classList.add('d-none')") >= 1,
          "the Remove Platform button is not shown and hidden with the selection")
    check(count_src.count("addPlatformBtn.classList.add('d-none')") >= 1,
          "an empty selection does not hide the Add Platform button")
    remove_at = count_src.index("addPlatformBtn.classList.remove('d-none')")
    scoped_at = count_src.index("if (scoped) {")
    check(remove_at < scoped_at,
          "the Add Platform button is shown only for one selection kind: it must work for both")

    # --- every element the feature's code reaches for exists on the page ------
    # A getElementById that returns null throws on the next property read, which
    # takes the whole modal down, and only when that branch runs.
    bulk_js = html[html.index("// ============ Bulk: add or remove a platform"):
                   html.index("// Bulk image operations")]
    wanted = sorted(set(re.findall(r"getElementById\('([^']+)'\)", bulk_js)))
    check(len(wanted) >= 10, f"only {len(wanted)} element ids found in the feature's code: the scan is broken")
    absent = [i for i in wanted if f'id="{i}"' not in html]
    check(not absent, f"the feature's code looks up elements the page does not have: {absent}")

    # --- one declaration per function, none shadowing an existing one ---------
    for name in FEATURE_FUNCTIONS:
        found = len(re.findall(rf"function\s+{name}\s*\(", html))
        check(found == 1, f"{name}() is declared {found} times")

    # --- the empty page still has the modal its parse-time listener needs -----
    empty_db, empty_web, empty_client, _ = rig(with_posts=False)
    empty = empty_client.get("/compose").get_data(as_text=True)
    check('id="bulkPlatformModal"' in empty,
          "with no saved posts the modal is missing, and the page script would throw at parse time")

    print(f"Add/Remove Platform: buttons, modal with {len(modal_chips)} account-aware chips, "
          f"mode-specific options and handlers are wired to the bulk add and remove endpoints")
    print("BULK_PLATFORM_UI_OK")


def section_syntax():
    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    database, web, client, ids = rig()
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
    check(len(blocks) >= 2, f"expected the page's inline scripts, found {len(blocks)}")
    largest = max(len(b) for b in blocks)
    check(largest > 100_000, f"the main script looks truncated ({largest} chars)")

    work = tempfile.mkdtemp(prefix="compose_js_")
    try:
        for index, body in enumerate(blocks):
            path = os.path.join(work, f"block{index}.js")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body)
            result = subprocess.run([node, "--check", path], capture_output=True, text=True)
            check(result.returncode == 0,
                  f"inline script #{index} does not parse:\n{result.stderr.strip()[:600]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"all {len(blocks)} inline scripts parse (largest {largest} chars)")
    print("COMPOSE_JS_SYNTAX_OK")


SECTIONS = {"wiring": section_wiring, "syntax": section_syntax}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_bulk_platform_ui.py {'|'.join(SECTIONS)}")
        sys.exit(2)
    SECTIONS[name]()
