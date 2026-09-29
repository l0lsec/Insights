"""G7: a login that arrives with no identity never damages an account that has one.

Some logins come back from OAuth with nothing to identify them: LinkedIn with
only the posting scope returns no profile, a Facebook login with no Pages has no
Page id to key on. The first version of multi-account support matched such a
login against the platform's default account, so connecting a second LinkedIn
overwrote the first one's token and blanked its member URN, which is the exact
failure the feature exists to prevent. Every gate before this one supplied an id
and so never went near that path.

This gate covers it at two levels. The storage level runs on all five platforms,
because the five save functions are five copies of the same shape. The callback
level drives the real OAuth routes with faked clients, because the storage fix
alone still left the redirect to the configure screen naming no account, which
made the screen write the second person's identity onto the first account.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import isolated_app, connect, check, FUTURE  # noqa: E402

PLATFORMS = ("linkedin", "threads", "twitter", "facebook", "instagram")


# ---------------------------------------------------------------------------
# Section 1: storage level, all five platforms
# ---------------------------------------------------------------------------

directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH

GET = {
    "linkedin": database.get_linkedin_token,
    "threads": database.get_threads_token,
    "twitter": database.get_twitter_token,
    "facebook": database.get_facebook_token,
    "instagram": database.get_instagram_token,
}


def save_without_identity(platform, token, new_login=False):
    """Save a login exactly as its callback would when the platform says nothing.

    ``new_login`` is what the callback passes when the run was started from the
    accounts screen, where every connect button means "add a login".
    """
    if platform == "linkedin":
        database.save_linkedin_token(token, FUTURE, "", "", "LinkedIn User (needs configuration)", "", new_login=new_login, db_path=P)
    elif platform == "threads":
        database.save_threads_token(token, FUTURE, "", "", new_login=new_login, db_path=P)
    elif platform == "twitter":
        database.save_twitter_token(token, "refresh", FUTURE, "", "", new_login=new_login, db_path=P)
    elif platform == "facebook":
        database.save_facebook_token(token, FUTURE, "", page_id=None, new_login=new_login, db_path=P)
    elif platform == "instagram":
        database.save_instagram_token(token, FUTURE, "", "", new_login=new_login, db_path=P)


# What the configure screen calls to give a placeholder its identity, and the
# identity that results. X has no configure screen: its identity only ever
# arrives over OAuth, so there is nothing to promote by hand.
def promote(platform, account_id, identity):
    if platform == "linkedin":
        return database.update_linkedin_member_urn(identity, display_name="Second", account_id=account_id, db_path=P)
    if platform == "threads":
        return database.update_threads_user_info(identity, username="second", display_name="Second", account_id=account_id, db_path=P)
    if platform == "instagram":
        return database.update_instagram_user_info("ig-u", username="second", display_name="Second", ig_user_id=identity, account_id=account_id, db_path=P)
    if platform == "facebook":
        return database.update_facebook_page_selection(identity, "Second Page", "page-token", account_id=account_id, db_path=P)
    raise ValueError(platform)


def accounts(platform):
    return database.list_social_accounts(platform, db_path=P)


def placeholders(platform):
    return [a for a in accounts(platform) if str(a["external_id"]).startswith("pending:")]


for platform in PLATFORMS:
    get = GET[platform]
    first = connect(database, platform, f"{platform}-1", "First")
    first_token = f"token-{platform}-{platform}-1"

    # --- an unidentified login is a new account, not the first one ---------
    save_without_identity(platform, "token-NOID")
    check(len(accounts(platform)) == 2,
          f"{platform}: a login with no identity did not become its own account "
          f"({len(accounts(platform))} accounts)")
    check(get(first, db_path=P)["access_token"] == first_token,
          f"{platform}: a login with no identity overwrote the first account's token")
    check(accounts(platform)[0]["id"] == first
          and accounts(platform)[0]["external_id"] == f"{platform}-1",
          f"{platform}: a login with no identity changed the first account's identity")
    check(bool(database.get_social_account(first, db_path=P)["is_default"]),
          f"{platform}: a login with no identity took the default from the first account")
    if platform == "linkedin":
        check(get(first, db_path=P)["user_urn"] == "urn:li:person:linkedin-1",
              "linkedin: a login with no profile blanked the first account's member URN")

    placeholder = placeholders(platform)
    check(len(placeholder) == 1, f"{platform}: expected one placeholder, got {len(placeholder)}")
    ph = placeholder[0]["id"]
    check(get(ph, db_path=P)["access_token"] == "token-NOID",
          f"{platform}: the placeholder does not hold the unidentified login's token")

    # --- a retry or re-authorisation reuses the placeholder ----------------
    save_without_identity(platform, "token-NOID2")
    check(len(accounts(platform)) == 2,
          f"{platform}: a second unidentified login multiplied accounts "
          f"({len(accounts(platform))})")
    check(get(ph, db_path=P)["access_token"] == "token-NOID2",
          f"{platform}: the placeholder did not take the newer login")
    check(get(first, db_path=P)["access_token"] == first_token,
          f"{platform}: a retry reached the first account")

    if platform == "twitter":
        continue

    # --- configuring the placeholder gives it its identity ------------------
    second_id = {"linkedin": "li-second", "threads": "th-second",
                 "instagram": "ig-second", "facebook": "fb-second"}[platform]
    check(promote(platform, ph, second_id),
          f"{platform}: configuring the placeholder was refused")
    check(database.get_social_account(ph, db_path=P)["external_id"] == second_id,
          f"{platform}: the placeholder did not take its identity")
    check(not placeholders(platform),
          f"{platform}: a configured placeholder is still listed as pending")
    check(get(first, db_path=P)["access_token"] == first_token,
          f"{platform}: configuring the second account touched the first")

    # --- re-authorising it gives a new placeholder, which merges back -------
    # An unidentified login that is really the second account again. Configuring
    # it with the identity the second account already has must fold it in, or
    # the user is stuck with an account that can never be configured.
    save_without_identity(platform, "token-NOID3")
    check(len(accounts(platform)) == 3,
          f"{platform}: a re-authorisation did not get a placeholder of its own")
    ph2 = placeholders(platform)[0]["id"]
    post = database.add_standalone_post(
        source_type="manual", source_content="gate", platform=platform,
        content="queued against the placeholder", account_id=ph2, db_path=P)
    check(promote(platform, ph2, second_id),
          f"{platform}: configuring a re-authorisation was refused")
    check(len(accounts(platform)) == 2,
          f"{platform}: the re-authorisation was not merged "
          f"({len(accounts(platform))} accounts)")
    check(not placeholders(platform),
          f"{platform}: a pending account was left behind by the merge")
    check(get(ph, db_path=P)["access_token"] == "token-NOID3",
          f"{platform}: the merge did not keep the newest credentials")
    check(database.get_standalone_post(post, db_path=P)["account_id"] == ph,
          f"{platform}: a post aimed at the merged placeholder was orphaned")
    check(get(first, db_path=P)["access_token"] == first_token,
          f"{platform}: the merge reached the first account")
    check(sum(1 for a in accounts(platform) if a["is_default"]) == 1,
          f"{platform}: the merge left the platform without exactly one default")

    # --- a real account is never moved onto another account's identity ------
    before = get(ph, db_path=P)
    refused = promote(platform, ph, f"{platform}-1")
    check(refused is False,
          f"{platform}: a real account was moved onto another account's identity")
    check(database.get_social_account(ph, db_path=P)["external_id"] == second_id,
          f"{platform}: a refused change still altered the account's identity")
    check(dict(get(ph, db_path=P)) == dict(before),
          f"{platform}: a refused change still altered the account's token")

# --- two logins added on purpose never share a placeholder -----------------
# Sharing is a fallback for callers that cannot say whether a login is new. The
# accounts screen can: every connect button there adds one. Two unidentified
# logins added from it must each survive, or the second silently destroys the
# first while the user is looking at a page that says "Connect another".
directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
for platform in PLATFORMS:
    get = GET[platform]
    real = connect(database, platform, f"{platform}-1", "First")
    real_token = f"token-{platform}-{platform}-1"

    save_without_identity(platform, "token-A", new_login=True)
    save_without_identity(platform, "token-B", new_login=True)
    pending = placeholders(platform)
    check(len(pending) == 2,
          f"{platform}: two logins added on purpose share {len(pending)} placeholder(s)")
    tokens = sorted(get(a["id"], db_path=P)["access_token"] for a in pending)
    check(tokens == ["token-A", "token-B"],
          f"{platform}: adding a second unidentified login destroyed the first: {tokens}")
    check(get(real, db_path=P)["access_token"] == real_token,
          f"{platform}: an added login reached the real account")

    # A caller that cannot say still gets the shared placeholder, and only ever
    # spends one of them. The real account and the other placeholder survive.
    save_without_identity(platform, "token-C")
    check(len(placeholders(platform)) == 2,
          f"{platform}: a login of unknown intent multiplied placeholders")
    after = sorted(get(a["id"], db_path=P)["access_token"] for a in placeholders(platform))
    check(after in (["token-A", "token-C"], ["token-B", "token-C"]),
          f"{platform}: a login of unknown intent did not reuse exactly one placeholder: {after}")
    check(get(real, db_path=P)["access_token"] == real_token,
          f"{platform}: a login of unknown intent reached the real account")

# --- a placeholder never holds the default against a real account ---------
directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH
save_without_identity("linkedin", "token-NOID")
placeholder_id = placeholders("linkedin")[0]["id"]
check(bool(database.get_social_account(placeholder_id, db_path=P)["is_default"]),
      "the only account on a platform should be its default, placeholder or not")
real = connect(database, "linkedin", "li-real", "Real")
check(database.resolve_account_id("linkedin", db_path=P) == real,
      "posts with no named account still go to the placeholder that cannot publish")
check(not database.get_social_account(placeholder_id, db_path=P)["is_default"],
      "a placeholder kept the default after a real account connected")

# ---------------------------------------------------------------------------
# Section 2: the real OAuth callbacks
# ---------------------------------------------------------------------------

directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH


class FakeLinkedIn:
    """Stands in for LinkedIn's OAuth endpoints; ``info`` is the profile reply."""

    def __init__(self, info):
        self.info = info

    def is_configured(self):
        return True

    def get_authorization_url(self, state=None):
        return "https://linkedin.example/authorize", "S"

    def exchange_code_for_token(self, code):
        return {"access_token": f"tok-{code}", "expires_in": 5184000}

    def get_user_info(self, token):
        return self.info


def linkedin_callback(info, code):
    web.get_linkedin_client = lambda: FakeLinkedIn(info)
    with client.session_transaction() as session:
        session["linkedin_oauth_state"] = "S"
    return client.get(f"/linkedin/callback?code={code}&state=S")


linkedin_callback({"sub": "m-first", "name": "First", "email": "f@x"}, "one")
second = linkedin_callback(None, "two")
check(second.status_code == 302,
      f"the LinkedIn callback for a login with no profile did not redirect ({second.status_code})")

placeholder_id = placeholders("linkedin")[0]["id"]
location = second.headers["Location"]
check(f"account_id={placeholder_id}" in location,
      f"the redirect to configure does not name the new account: {location}")

first_account = accounts("linkedin")[0]
check(first_account["external_id"] == "m-first"
      and database.get_linkedin_token(first_account["id"], db_path=P)["access_token"] == "tok-one",
      "the second LinkedIn callback damaged the first account")

# The user follows that redirect and types the second person's Member ID.
configured = client.post(location, data={"member_id": "m-second", "display_name": "Second"})
check(configured.status_code == 302,
      f"configuring the second LinkedIn account failed ({configured.status_code})")
check(database.get_social_account(first_account["id"], db_path=P)["external_id"] == "m-first",
      "configuring the second account rewrote the first account's identity")
check(database.get_linkedin_token(first_account["id"], db_path=P)["member_id"] == "m-first",
      "configuring the second account rewrote the first account's member id")
check(database.get_social_account(placeholder_id, db_path=P)["external_id"] == "m-second",
      "the second account did not receive the member id that was typed for it")
check(database.get_linkedin_token(placeholder_id, db_path=P)["access_token"] == "tok-two",
      "the second account lost its own token")

# Configuring the first account's Member ID onto the second is refused, with a reason.
refused = client.post(f"/linkedin/configure?account_id={placeholder_id}",
                      data={"member_id": "m-first", "display_name": "Clash"})
check(refused.status_code == 200 and b"already belongs to another" in refused.data,
      "a Member ID clash was not explained to the user")
check(database.get_social_account(placeholder_id, db_path=P)["external_id"] == "m-second",
      "a refused Member ID clash still changed the account")

# The accounts page's Connect button starts an OAuth run whose intent is "add a
# login". That has to survive the trip through the platform and back, so this
# goes through the real /auth route rather than setting the session by hand.
def linkedin_via_auth(info, code, from_accounts):
    web.get_linkedin_client = lambda: FakeLinkedIn(info)
    started = client.get("/linkedin/auth" + ("?return=accounts" if from_accounts else ""))
    check(started.status_code == 302, f"/linkedin/auth did not redirect ({started.status_code})")
    return client.get(f"/linkedin/callback?code={code}&state=S")


before = len(placeholders("linkedin"))
first_add = linkedin_via_auth(None, "added1", from_accounts=True)
second_add = linkedin_via_auth(None, "added2", from_accounts=True)
check(len(placeholders("linkedin")) == before + 2,
      f"two logins added from the accounts screen produced "
      f"{len(placeholders('linkedin')) - before} new placeholder(s), expected 2")
added = {database.get_linkedin_token(a["id"], db_path=P)["access_token"]: a["id"]
         for a in placeholders("linkedin")}
check("tok-added1" in added and "tok-added2" in added,
      f"the first added login was overwritten by the second: {sorted(added)}")
check(f"account_id={added['tok-added1']}" in first_add.headers["Location"]
      and f"account_id={added['tok-added2']}" in second_add.headers["Location"],
      "each added login was not sent to configure its own account")

# A run that did not start from the accounts screen keeps the old behaviour.
count = len(placeholders("linkedin"))
linkedin_via_auth(None, "legacy", from_accounts=False)
check(len(placeholders("linkedin")) == count,
      "a run not started from the accounts screen minted a placeholder of its own")
check(database.get_social_account(first_account["id"], db_path=P)["external_id"] == "m-first",
      "a legacy run damaged the first account")

# The intent is read once. A later run that never touched the accounts screen
# must not inherit it from an earlier one.
linkedin_via_auth(None, "added3", from_accounts=True)
count = len(placeholders("linkedin"))
web.get_linkedin_client = lambda: FakeLinkedIn(None)
with client.session_transaction() as session:
    session["linkedin_oauth_state"] = "S"
client.get("/linkedin/callback?code=stale&state=S")
check(len(placeholders("linkedin")) == count,
      "a callback inherited the add-a-login intent from an earlier, finished run")

# The accounts page offers a way to fix a login that still needs configuring.
linkedin_callback(None, "three")
page = client.get("/accounts").get_data(as_text=True)
pending = placeholders("linkedin")[0]["id"]
check(f"/linkedin/configure?account_id={pending}" in page,
      "the accounts page gives no way to configure a login that needs it")


class FakeFacebook:
    """Stands in for Facebook's OAuth and Graph endpoints."""

    def __init__(self, pages):
        self.pages = pages

    def exchange_code_for_token(self, code):
        return {"access_token": f"short-{code}"}

    def get_long_lived_token(self, token):
        return {"access_token": token.replace("short", "long"), "expires_in": 5184000}

    def get_user_profile(self, token):
        return {"id": "fb-user", "name": "A Person"}

    def get_user_pages(self, token):
        return self.pages

    def get_user_groups(self, token):
        return []


def facebook_callback(pages, code):
    web.get_facebook_client = lambda: FakeFacebook(pages)
    with client.session_transaction() as session:
        session["facebook_oauth_state"] = "S"
    return client.get(f"/facebook/callback?code={code}&state=S")


def page(page_id):
    return {"id": page_id, "name": f"Page {page_id}", "access_token": f"pt-{page_id}"}


directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH

facebook_callback([page("page-1")], "one")
first_fb = accounts("facebook")[0]
check(first_fb["external_id"] == "page-1", "the first Facebook login did not key on its Page")

facebook_callback([], "nopages")
check(database.get_facebook_token(first_fb["id"], db_path=P)["page_id"] == "page-1",
      "a Facebook login with no Pages overwrote the first account's Page")
check(len(accounts("facebook")) == 2,
      "a Facebook login with no Pages did not become its own account")

multi = facebook_callback([page("page-a"), page("page-b")], "multi")
check(multi.status_code == 302, "the multi-Page Facebook callback did not redirect")
new_account = database.find_social_account("facebook", "page-a", db_path=P)
check(new_account is not None, "the multi-Page login did not become an account")
check(f"account_id={new_account['id']}" in multi.headers["Location"],
      f"the Page picker does not name the account that just connected: "
      f"{multi.headers['Location']}")

# Choosing the other Page for that login must not touch the first account.
chosen = client.post(multi.headers["Location"], data={"page_id": "page-b"})
check(chosen.status_code == 302, f"choosing a Page failed ({chosen.status_code})")
check(database.get_facebook_token(first_fb["id"], db_path=P)["page_id"] == "page-1",
      "choosing a Page for the new login rewrote the first account's Page")
check(database.get_social_account(new_account["id"], db_path=P)["external_id"] == "page-b",
      "the new login's account did not move to the Page that was chosen")

# ---------------------------------------------------------------------------
# Section 3: the add-a-login intent reaches the save through every callback
# ---------------------------------------------------------------------------
# LinkedIn was driven above. The other four callbacks each had the same argument
# added by hand, so each is driven here through its real /auth and /callback
# routes, which is the only thing that notices one of them losing it.

class FakeOAuth:
    """Stands in for the Threads, Instagram, X and Facebook clients.

    They differ in which methods their callbacks call but all return plain
    dicts, and every reply here is the platform saying nothing about who logged
    in: no profile, no Pages.
    """

    def __init__(self, three=False):
        self.three = three

    def is_configured(self):
        return True

    def get_authorization_url(self, state=None):
        # X also returns the PKCE verifier its callback needs back.
        return ("https://oauth.example/authorize", "S", "verifier") if self.three \
            else ("https://oauth.example/authorize", "S")

    def exchange_code_for_token(self, code, verifier=None):
        return {"access_token": f"short-{code}", "refresh_token": "refresh",
                "expires_in": 7200}

    def get_long_lived_token(self, token):
        return {"access_token": token.replace("short", "long"), "expires_in": 5184000}

    def get_user_profile(self, token):
        return None

    def get_user_info(self, token):
        return None

    def get_user_pages(self, token):
        return []

    def get_user_groups(self, token):
        return []


directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH

for platform, factory, three in (
    ("threads", "get_threads_client", False),
    ("instagram", "get_instagram_client", False),
    ("twitter", "get_twitter_client", True),
    ("facebook", "get_facebook_client", False),
):
    setattr(web, factory, lambda three=three: FakeOAuth(three=three))

    def via_auth(code, from_accounts, platform=platform):
        started = client.get(f"/{platform}/auth" + ("?return=accounts" if from_accounts else ""))
        check(started.status_code == 302,
              f"/{platform}/auth did not redirect ({started.status_code})")
        return client.get(f"/{platform}/callback?code={code}&state=S")

    real = connect(database, platform, f"{platform}-1", "First")
    real_token = f"token-{platform}-{platform}-1"

    via_auth("a1", from_accounts=True)
    via_auth("a2", from_accounts=True)
    pending = placeholders(platform)
    check(len(pending) == 2,
          f"{platform}: two logins added from the accounts screen produced "
          f"{len(pending)} placeholder(s), so the callback dropped the intent")
    tokens = sorted(GET[platform](a["id"], db_path=P)["access_token"] for a in pending)
    check(tokens[0].endswith("a1") and tokens[1].endswith("a2"),
          f"{platform}: adding a second unidentified login destroyed the first: {tokens}")
    check(GET[platform](real, db_path=P)["access_token"] == real_token,
          f"{platform}: an added login reached the real account")

    via_auth("legacy", from_accounts=False)
    check(len(placeholders(platform)) == 2,
          f"{platform}: a run not started from the accounts screen minted a placeholder")

print("login without identity: an unidentified login never damages an identified "
      "account on any platform, and the callbacks send the user to the right account")
print("NO_IDENTITY_OK")
