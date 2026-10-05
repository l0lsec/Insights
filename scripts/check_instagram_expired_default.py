"""G8: an expired Instagram default never hides a working login.

Found 2026-10-05: connecting a second Instagram account (a brand account) while
the default account's token had expired left the schedule card reading "Token
expired. Please reconnect." The connect had worked; the status check only ever
looked at the default, and the new login joined as a non-default account. Posts
that named no account kept going to the dead login too.

This gate drives the real /instagram/status route and the real OAuth callback
against a throwaway database with a fake Instagram client, and checks:

  - a dead default plus a live second login reads as connected, names the live
    login, and warns about the dead one;
  - connecting a login while the default is dead makes the new login default;
  - a default that is merely close to expiry (still refreshable) keeps the
    default, and a fully live default is never displaced (negative controls);
  - with every login dead the card still says disconnected.
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import isolated_app, connect, check  # noqa: E402

directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH

PAST = "2026-09-18T22:40:47"
SOON = (datetime.utcnow() + timedelta(minutes=20)).isoformat(timespec="seconds")


class FakeInstagram:
    """Refresh always fails, as it does for a token past its expiry."""

    def __init__(self, new_login=None):
        self.new_login = new_login or {}

    def is_configured(self):
        return True

    def refresh_access_token(self, token):
        raise RuntimeError("Session has expired")

    def exchange_code_for_token(self, code):
        return {"access_token": "short", "user_id": self.new_login["user_id"]}

    def get_long_lived_token(self, short):
        return {"access_token": self.new_login["token"], "expires_in": 5184000}

    def get_user_profile(self, token):
        return {
            "id": self.new_login["user_id"],
            "user_id": self.new_login["ig_user_id"],
            "username": self.new_login["username"],
            "name": self.new_login["username"],
            "account_type": "BUSINESS",
            "profile_picture_url": "",
        }


def status():
    return client.get("/instagram/status").get_json()


def default_handle():
    return database.get_default_social_account("instagram", db_path=P)["handle"]


web.get_instagram_client = lambda: FakeInstagram()

# --- every login dead: still disconnected (the honest case) ----------------
connect(database, "instagram", "ig-creator", "creator")
database.update_instagram_token("token-dead", PAST, db_path=P)
s = status()
check(s["connected"] is False, f"a lone expired login read as connected: {s}")
check(len(s.get("accounts", [])) == 1, "the expired branch dropped the account list")

# --- dead default + live second login: connected, names the live one -------
connect(database, "instagram", "ig-brand", "brand")
check(default_handle() == "creator", "test setup: the creator login should still be default")
s = status()
check(s["connected"] is True, f"a live second login was reported as disconnected: {s}")
check(s.get("username") == "brand", f"status named the wrong login: {s.get('username')}")
check("creator" in (s.get("warning") or ""), "status did not warn about the expired default")
check(s.get("default_expired") is True, "status did not flag the expired default")

# --- the real callback: connecting over a dead default takes the default ---
database.set_default_social_account(
    database.find_social_account("instagram", "ig-creator", db_path=P)["id"], db_path=P)
web.get_instagram_client = lambda: FakeInstagram(
    {"user_id": "u-new", "ig_user_id": "ig-new", "username": "newbrand", "token": "tok-new"})
with client.session_transaction() as session:
    session["instagram_oauth_state"] = "st"
resp = client.get("/instagram/callback?code=c&state=st")
check(resp.status_code in (301, 302), f"callback did not redirect: {resp.status_code}")
check(default_handle() == "newbrand",
      f"connecting over an expired default left the default on @{default_handle()}")
check(status()["username"] == "newbrand", "status did not follow the new default")

# --- negative control: a default close to expiry (refreshable) is kept ----
creator_id = database.find_social_account("instagram", "ig-creator", db_path=P)["id"]
database.set_default_social_account(creator_id, db_path=P)
database.update_instagram_token("token-soon", SOON, account_id=creator_id, db_path=P)
check(web._promote_over_dead_instagram_default(
    database.find_social_account("instagram", "ig-brand", db_path=P)["id"]) is False,
    "a still-refreshable default was displaced")
check(default_handle() == "creator", "the refreshable default moved")

# --- negative control: a fully live default is never displaced ------------
web.get_instagram_client = lambda: FakeInstagram(
    {"user_id": "u-third", "ig_user_id": "ig-third", "username": "third", "token": "tok-3"})
database.update_instagram_token("token-live", "2099-01-01T00:00:00",
                                account_id=creator_id, db_path=P)
with client.session_transaction() as session:
    session["instagram_oauth_state"] = "st2"
client.get("/instagram/callback?code=c&state=st2")
check(default_handle() == "creator", "connecting a login displaced a live default")

print("IG_EXPIRED_DEFAULT_OK")
