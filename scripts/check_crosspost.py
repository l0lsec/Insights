"""G3: one piece of copy reaches every platform and every account it names.

The two halves of the feature meet here. A card is created for six targets that
span five platforms and two accounts on one of them, published in a single
request, and each fake client is asked which token it was handed. That is the
only way to prove the copy did not all go out through one account.

The gate also covers the awkward half of a fan-out: one target failing must not
stop the others, and the reply has to say which ones landed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import (  # noqa: E402
    isolated_app, install_fake_clients, connect, check, RecordingClient,
)

directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
clients = install_fake_clients(publisher)

work = connect(database, "linkedin", "li-work", "Work")
studio = connect(database, "linkedin", "li-studio", "Studio")
threads = connect(database, "threads", "th-1", "brandco")
twitter = connect(database, "twitter", "tw-1", "brandco")
facebook = connect(database, "facebook", "fb-1", "Brand Page")
instagram = connect(database, "instagram", "ig-1", "brandco")

# --- one card, six targets across five platforms --------------------------
COPY = "The same announcement, everywhere it needs to go."
created = client.post("/compose/post/create", data={
    "targets": [f"linkedin:{work}", f"linkedin:{studio}", f"threads:{threads}",
                f"twitter:{twitter}", f"facebook:{facebook}", f"instagram:{instagram}"],
    "content": COPY,
    "image_url": "https://example.test/pic.jpg",
}).get_json()
check(created and created.get("success"), f"creating the card failed: {created}")
post_ids = created["post_ids"]
check(len(post_ids) == 6, f"expected 6 rows for 6 targets, got {len(post_ids)}")

# The rows are one card, not six: same copy, one row per target.
rows = [database.get_standalone_post(pid, db_path=P) for pid in post_ids]
check(all(r["content"] == COPY for r in rows), "the rows do not share their copy")
targets = {(r["platform"], r["account_id"]) for r in rows}
check(len(targets) == 6, f"rows collapsed onto shared targets: {targets}")
check(("linkedin", work) in targets and ("linkedin", studio) in targets,
      "both LinkedIn accounts did not get their own row")

# --- the card renders as one card with six chips ---------------------------
page = client.get("/compose").get_data(as_text=True)
check(page.count(f'data-post-ids="{",".join(str(i) for i in post_ids)}"') == 1
      or "data-targets" in page,
      "the card did not render as a single card over its rows")
check("Work" in page and "Studio" in page,
      "the two LinkedIn accounts are not named on the card")

# --- publish the whole card in one request ---------------------------------
result = client.post(f"/compose/post/{post_ids[0]}/publish", data={
    "post_ids": ",".join(str(i) for i in post_ids),
}).get_json()
check(result and result.get("success"), f"the fan-out reported failure: {result}")
check(result["published"] == 6,
      f"expected 6 targets published, got {result['published']}")

# --- each target published with its own credentials ------------------------
li_tokens = [c["token"] for c in clients["linkedin"].calls if c["kind"] != "refresh"]
check(len(li_tokens) == 2, f"LinkedIn published {len(li_tokens)} times, expected 2")
check(set(li_tokens) == {"token-linkedin-li-work", "token-linkedin-li-studio"},
      f"the two LinkedIn accounts did not publish with their own tokens: {li_tokens}")

for platform, expected in (("threads", "token-threads-th-1"),
                           ("twitter", "token-twitter-tw-1"),
                           ("facebook", "token-facebook-fb-1"),
                           ("instagram", "token-instagram-ig-1")):
    tokens = [c["token"] for c in clients[platform].calls if c["kind"] != "refresh"]
    check(tokens == [expected],
          f"{platform} published with {tokens}, expected [{expected}]")

# --- publishing recorded the account it went out as ------------------------
history = database.list_scheduled_posts(status="posted", db_path=P)
check(len(history) == 6, f"expected 6 history rows, got {len(history)}")
history_targets = {(h["platform"], h["account_id"]) for h in history}
check(history_targets == targets,
      f"history does not record the accounts that published: {history_targets}")
check(all(database.get_standalone_post(pid, db_path=P)["used"] for pid in post_ids),
      "published rows were not marked used")

# --- one failing target does not stop the rest -----------------------------
directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
clients = install_fake_clients(
    publisher,
    twitter=RecordingClient("twitter", fail=True, error="over the rate limit"),
)
work = connect(database, "linkedin", "li-work", "Work")
studio = connect(database, "linkedin", "li-studio", "Studio")
twitter = connect(database, "twitter", "tw-1", "brandco")

created = client.post("/compose/post/create", data={
    "targets": [f"linkedin:{work}", f"linkedin:{studio}", f"twitter:{twitter}"],
    "content": "Partial fan-out",
}).get_json()
ids = created["post_ids"]
result = client.post(f"/compose/post/{ids[0]}/publish", data={
    "post_ids": ",".join(str(i) for i in ids),
}).get_json()

check(not result["success"], "a fan-out with a failing target reported full success")
check(result["partial"], "a partly successful fan-out was not reported as partial")
check(result["published"] == 2,
      f"expected 2 of 3 targets published, got {result['published']}")
failed = [r for r in result["results"] if not r["success"]]
check(len(failed) == 1 and failed[0]["platform"] == "twitter",
      f"the wrong target failed: {failed}")
check("rate limit" in (failed[0]["error"] or ""),
      f"the platform's reason was not reported: {failed[0]['error']}")
check(len([c for c in clients["linkedin"].calls if c["kind"] != "refresh"]) == 2,
      "a failing target stopped the other targets from publishing")

# --- a card can be published to a subset of its targets ---------------------
directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
clients = install_fake_clients(publisher)
work = connect(database, "linkedin", "li-work", "Work")
studio = connect(database, "linkedin", "li-studio", "Studio")
created = client.post("/compose/post/create", data={
    "targets": [f"linkedin:{work}", f"linkedin:{studio}"],
    "content": "Only one of these",
}).get_json()
ids = created["post_ids"]
result = client.post(f"/compose/post/{ids[0]}/publish", data={
    "post_ids": ",".join(str(i) for i in ids),
    "target_ids": str(ids[1]),
}).get_json()
check(result["attempted"] == 1,
      f"narrowing the fan-out published {result['attempted']} targets, expected 1")
published = [c["token"] for c in clients["linkedin"].calls if c["kind"] != "refresh"]
check(published == ["token-linkedin-li-studio"],
      f"the narrowed fan-out published the wrong account: {published}")

# --- an article's social post publishes to a named account too -------------
# Article posts go out through a different set of routes; they must route by
# account the same way, or a second account would work in the composer and
# silently publish as the first from the Articles page.
directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
clients = install_fake_clients(publisher)
work = connect(database, "linkedin", "li-work", "Work")
studio = connect(database, "linkedin", "li-studio", "Studio")
article_id = database.add_article(None, "A topic", "brief", "Body of the article", db_path=P)
social_id = database.add_social_post(article_id, "linkedin", "From an article", db_path=P)
response = client.post(f"/linkedin/post/{social_id}",
                       data={"account_id": str(studio)}).get_json()
check(response and response.get("success"),
      f"publishing an article's social post failed: {response}")
tokens = [c["token"] for c in clients["linkedin"].calls if c["kind"] != "refresh"]
check(tokens == ["token-linkedin-li-studio"],
      f"the article route published as the wrong account: {tokens}")

print("cross-posting: one card reached 5 platforms and 2 accounts on one of them, "
      "each with its own token; partial failures and subsets both report honestly")
print("CROSSPOST_OK")
