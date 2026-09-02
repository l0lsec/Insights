"""G4: the queue carries accounts, and the scheduler honours them.

A queued post is published minutes or days after it was queued, by a background
thread with no request context, so it cannot ask the composer which account was
meant. The account has to be on the queue row itself. This gate queues one card
to several accounts and checks that each entry publishes with its own token,
that a failure retries rather than dying, and that an unrecoverable one fails
straight away instead of burning the platform's remaining slots.
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

# Slots every day, so queueing always finds somewhere to land.
for hour in ("09:00", "12:00", "15:00", "18:00"):
    database.add_time_slot(-1, hour, True,
                           ["linkedin", "threads", "twitter", "facebook", "instagram"],
                           db_path=P)

work = connect(database, "linkedin", "li-work", "Work")
studio = connect(database, "linkedin", "li-studio", "Studio")
threads = connect(database, "threads", "th-1", "brandco")

created = client.post("/compose/post/create", data={
    "targets": [f"linkedin:{work}", f"linkedin:{studio}", f"threads:{threads}"],
    "content": "Queued for everyone",
}).get_json()
ids = created["post_ids"]

# --- queueing the whole card takes one slot per target ---------------------
queued = client.post(f"/compose/post/{ids[0]}/queue", data={
    "all": "1", "post_ids": ",".join(str(i) for i in ids),
}).get_json()
check(queued and queued.get("success"), f"queueing the card failed: {queued}")
check(len(queued["queued"]) == 3,
      f"expected 3 queue entries, got {len(queued['queued'])}")

pending = database.list_scheduled_posts(status="pending", db_path=P)
check(len(pending) == 3, f"expected 3 pending entries, got {len(pending)}")
entry_targets = {(e["platform"], e["account_id"]) for e in pending}
check(entry_targets == {("linkedin", work), ("linkedin", studio), ("threads", threads)},
      f"queue entries did not record their accounts: {entry_targets}")

# The two LinkedIn entries take different slots rather than stacking.
li_slots = sorted(e["scheduled_for"] for e in pending if e["platform"] == "linkedin")
check(li_slots[0] != li_slots[1],
      f"two LinkedIn accounts were queued into the same slot: {li_slots}")

# --- the worker's publish path uses each entry's own account ---------------
for entry in pending:
    result = web._publish_scheduled_entry(database.get_scheduled_post(entry["id"], db_path=P))
    check(result["success"], f"publishing queue entry {entry['id']} failed: {result}")

li_tokens = sorted(c["token"] for c in clients["linkedin"].calls if c["kind"] != "refresh")
check(li_tokens == ["token-linkedin-li-studio", "token-linkedin-li-work"],
      f"the queue published LinkedIn with {li_tokens}")
th_tokens = [c["token"] for c in clients["threads"].calls if c["kind"] != "refresh"]
check(th_tokens == ["token-threads-th-1"],
      f"the queue published Threads with {th_tokens}")
check(not database.list_scheduled_posts(status="pending", db_path=P),
      "published entries are still pending")

# --- post-now from the schedule page uses the entry's account -------------
directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
clients = install_fake_clients(publisher)
work = connect(database, "linkedin", "li-work", "Work")
studio = connect(database, "linkedin", "li-studio", "Studio")
post_id = database.add_standalone_post(
    source_type="manual", source_content="gate", platform="linkedin",
    content="Post me now", account_id=studio, db_path=P,
)
scheduled_id = database.add_scheduled_post(
    scheduled_for="2099-01-01T09:00:00", post_type="standalone",
    standalone_post_id=post_id, platform="linkedin", account_id=studio, db_path=P,
)
response = client.post(f"/schedule/{scheduled_id}/post-now").get_json()
check(response and response.get("success"), f"post-now failed: {response}")
tokens = [c["token"] for c in clients["linkedin"].calls if c["kind"] != "refresh"]
check(tokens == ["token-linkedin-li-studio"],
      f"post-now published as the wrong account: {tokens}")

# --- a queue row with no account still publishes as the default -----------
directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
clients = install_fake_clients(publisher)
work = connect(database, "linkedin", "li-work", "Work")
legacy_post = database.add_standalone_post(
    source_type="manual", source_content="gate", platform="linkedin",
    content="Written before accounts existed", db_path=P,
)
import sqlite3  # noqa: E402
with sqlite3.connect(P) as conn:
    conn.execute("UPDATE standalone_posts SET account_id = NULL WHERE id = ?", (legacy_post,))
    conn.commit()
legacy_entry = database.add_scheduled_post(
    scheduled_for="2099-01-01T09:00:00", post_type="standalone",
    standalone_post_id=legacy_post, platform="linkedin", db_path=P,
)
with sqlite3.connect(P) as conn:
    conn.execute("UPDATE scheduled_posts SET account_id = NULL WHERE id = ?", (legacy_entry,))
    conn.commit()
result = web._publish_scheduled_entry(database.get_scheduled_post(legacy_entry, db_path=P))
check(result["success"], f"a queue row with no account failed to publish: {result}")
tokens = [c["token"] for c in clients["linkedin"].calls if c["kind"] != "refresh"]
check(tokens == ["token-linkedin-li-work"],
      f"a queue row with no account did not use the default account: {tokens}")

# --- an unrecoverable failure is marked failed, not retried forever -------
directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
install_fake_clients(publisher)
orphan_post = database.add_standalone_post(
    source_type="manual", source_content="gate", platform="linkedin",
    content="Nothing is connected", db_path=P,
)
orphan_entry = database.add_scheduled_post(
    scheduled_for="2099-01-01T09:00:00", post_type="standalone",
    standalone_post_id=orphan_post, platform="linkedin", db_path=P,
)
result = web._publish_scheduled_entry(database.get_scheduled_post(orphan_entry, db_path=P))
check(not result["success"], "publishing with nothing connected reported success")
check(result["permanent"],
      "a disconnected platform was treated as a retryable failure")

# --- a platform rejecting the post is retryable ---------------------------
directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
install_fake_clients(
    publisher, linkedin=RecordingClient("linkedin", fail=True, error="try again later"),
)
connect(database, "linkedin", "li-work", "Work")
flaky_post = database.add_standalone_post(
    source_type="manual", source_content="gate", platform="linkedin",
    content="Rejected once", db_path=P,
)
flaky_entry = database.add_scheduled_post(
    scheduled_for="2099-01-01T09:00:00", post_type="standalone",
    standalone_post_id=flaky_post, platform="linkedin", db_path=P,
)
result = web._publish_scheduled_entry(database.get_scheduled_post(flaky_entry, db_path=P))
check(not result["success"], "a rejected post reported success")
check(not result["permanent"],
      "a platform rejecting a post was treated as permanent, so it would never retry")

print("scheduler: queue entries carry their account, publish with their own token, "
      "and separate retryable failures from permanent ones")
print("SCHEDULER_ACCOUNTS_OK")
