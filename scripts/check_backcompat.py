"""G6: an existing single-account install keeps working, unchanged.

Multi-account support is a change to the shape of every token and every post
row, so the risk is not that the new feature fails, it is that somebody's
working setup stops posting. This gate builds a database in the pre-accounts
shape, upgrades it, and checks that the login it was already using is still the
one it publishes with, that the migration can run twice, and that every caller
written before accounts existed still behaves the same way.
"""

import sqlite3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import (  # noqa: E402
    isolated_app, install_fake_clients, check, FUTURE,
)

directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
clients = install_fake_clients(publisher, web)


def legacy_state():
    """Put the database back into its pre-accounts shape.

    Tokens with no account, posts and queue rows with no account, and no
    accounts table content: exactly what an install that upgraded from an older
    version looks like the moment before the migration runs.
    """
    with sqlite3.connect(P) as conn:
        conn.execute("UPDATE linkedin_tokens SET account_id = NULL")
        conn.execute("UPDATE threads_tokens SET account_id = NULL")
        conn.execute("UPDATE standalone_posts SET account_id = NULL")
        conn.execute("UPDATE scheduled_posts SET account_id = NULL")
        conn.execute("DELETE FROM social_accounts")
        conn.commit()


# --- build an install as it looked before accounts existed ----------------
database.save_linkedin_token(
    access_token="legacy-li", expires_at=FUTURE, member_id="li-legacy",
    user_urn="urn:li:person:li-legacy", display_name="My LinkedIn",
    email="me@example.com", refresh_token="legacy-refresh", db_path=P,
)
database.save_threads_token(
    access_token="legacy-th", expires_at=FUTURE, user_id="th-legacy",
    username="me", display_name="My Threads", db_path=P,
)
post_id = database.add_standalone_post(
    source_type="manual", source_content="old", platform="linkedin",
    content="An old post", db_path=P,
)
queue_id = database.add_scheduled_post(
    scheduled_for="2099-01-01T09:00:00", post_type="standalone",
    standalone_post_id=post_id, platform="linkedin", db_path=P,
)
legacy_state()

with sqlite3.connect(P) as conn:
    check(conn.execute("SELECT COUNT(*) FROM social_accounts").fetchone()[0] == 0,
          "the legacy fixture still has accounts, so the migration is not being tested")

# --- upgrading promotes each existing login into its default account ------
database.init_db(P)

accounts = database.list_social_accounts(db_path=P)
check(len(accounts) == 2,
      f"the migration made {len(accounts)} accounts from 2 logins")
linkedin = database.get_default_social_account("linkedin", db_path=P)
check(linkedin is not None, "the LinkedIn login did not become an account")
check(linkedin["external_id"] == "li-legacy",
      f"the migrated account has the wrong identity: {linkedin['external_id']}")
check(linkedin["display_name"] == "My LinkedIn",
      "the migrated account lost the name the platform gave it")
check(bool(linkedin["is_default"]),
      "the only account on a platform is not its default")

# --- the token still belongs to that login, and reads the same way --------
token = database.get_linkedin_token(db_path=P)
check(token is not None and token["access_token"] == "legacy-li",
      "the existing token is no longer readable")
check(token["account_id"] == linkedin["id"],
      "the existing token was not linked to its account")
check(database.get_linkedin_token(linkedin["id"], db_path=P)["access_token"] == "legacy-li",
      "the token is not reachable by its account either")

# --- old posts and queue rows now name that account -----------------------
check(database.get_standalone_post(post_id, db_path=P)["account_id"] == linkedin["id"],
      "an existing saved post was not pointed at the migrated account")
check(database.get_scheduled_post(queue_id, db_path=P)["account_id"] == linkedin["id"],
      "an existing queue entry was not pointed at the migrated account")

# --- the migration is safe to run again -----------------------------------
database.init_db(P)
database.init_db(P)
check(len(database.list_social_accounts(db_path=P)) == 2,
      "running the migration again duplicated accounts")
check(database.get_linkedin_token(db_path=P)["access_token"] == "legacy-li",
      "running the migration again disturbed the token")

# --- callers written before accounts existed still behave the same --------
plain_post = database.add_standalone_post(
    source_type="manual", source_content="new", platform="linkedin",
    content="Written without naming an account", db_path=P,
)
check(database.get_standalone_post(plain_post, db_path=P)["account_id"] == linkedin["id"],
      "a post saved without an account did not land on the default account")

plain_queue = database.add_scheduled_post(
    scheduled_for="2099-01-01T10:00:00", post_type="standalone",
    standalone_post_id=plain_post, platform="linkedin", db_path=P,
)
check(database.get_scheduled_post(plain_queue, db_path=P)["account_id"] == linkedin["id"],
      "a queue entry made without an account did not land on the default account")

response = client.post(f"/compose/post/{plain_post}/linkedin").get_json()
check(response and response.get("success"),
      f"the per-platform post-now endpoint stopped working: {response}")
tokens = [c["token"] for c in clients["linkedin"].calls if c["kind"] != "refresh"]
check(tokens == ["legacy-li"],
      f"posting without naming an account used {tokens}, expected the existing login")

# --- a token that has never been migrated is still readable ---------------
# Half-upgraded databases exist in the wild: the app may read a token before
# init_db has linked it. That must not lock the user out of posting.
legacy_state()
check(database.get_linkedin_token(db_path=P) is not None,
      "an unmigrated token became unreadable")
check(database.get_linkedin_token(db_path=P)["access_token"] == "legacy-li",
      "an unmigrated token read back the wrong credentials")

# --- disconnecting a platform outright still empties it -------------------
database.init_db(P)
response = client.post("/linkedin/disconnect").get_json()
check(response and response.get("success"), f"the disconnect route failed: {response}")
check(database.get_linkedin_token(db_path=P) is None,
      "disconnecting the platform left a token behind")
check(database.count_social_accounts("linkedin", db_path=P) == 0,
      "disconnecting the platform left an account behind")
check(database.get_threads_token(db_path=P) is not None,
      "disconnecting LinkedIn took the Threads login with it")

print("backward compatibility: an existing single-account install migrates once, "
      "keeps its login and its token, and every pre-accounts caller behaves as before")
print("BACKCOMPAT_OK")
