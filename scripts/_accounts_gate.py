"""Shared harness for the multi-account and cross-posting gates.

Every gate here runs against a throwaway database and fake platform clients, so
a gate run needs no credentials, makes no network calls, and cannot touch the
real insights.db. That is what makes it safe to run on every change.

The fakes record what they were called with. A gate does not assert "publishing
worked" (nothing real was published); it asserts that the right token reached
the right platform for the right account, which is exactly what multi-account
support has to get right and exactly what a single-account codebase got wrong.
"""

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


class FakeResult(dict):
    """A client reply in the shape the real clients return."""


class RecordingClient:
    """A platform client that records calls instead of making them.

    ``calls`` holds one entry per publish attempt, each carrying the access
    token it was handed. A gate reads that to prove account B's post went out
    with account B's token rather than the default account's.
    """

    def __init__(self, name, fail=False, error="rejected by platform"):
        self.name = name
        self.calls = []
        self.fail = fail
        self.error = error

    # --- shared ---------------------------------------------------------
    def is_configured(self):
        return True

    def _record(self, kind, token, **extra):
        self.calls.append({"kind": kind, "token": token, **extra})
        if self.fail:
            return FakeResult(success=False, error=self.error)
        return FakeResult(
            success=True,
            permalink=f"https://{self.name}.test/{len(self.calls)}",
            post_urn=f"urn:{self.name}:{len(self.calls)}",
        )

    def refresh_access_token(self, token):
        self.calls.append({"kind": "refresh", "token": token})
        return {"access_token": f"refreshed-{token}", "expires_in": 5184000}

    # --- LinkedIn -------------------------------------------------------
    @staticmethod
    def extract_first_url(text):
        return None

    def create_smart_post(self, access_token, author_urn, text, article_title=None):
        return self._record("text", access_token, urn=author_urn, text=text)

    def create_image_post(self, access_token, author_urn=None, text=None, image_url=None):
        return self._record("image", access_token, text=text, image=image_url)

    # --- Threads --------------------------------------------------------
    def publish_text_post(self, access_token, text):
        return self._record("text", access_token, text=text)

    def publish_image_post(self, access_token, text=None, image_url=None, user_tags=None):
        return self._record("image", access_token, text=text, image=image_url)

    # --- X/Twitter ------------------------------------------------------
    def create_post(self, access_token, text):
        return self._record("text", access_token, text=text)

    # --- Facebook -------------------------------------------------------
    def publish_smart_post(self, page_access_token, page_id, text, image_url=None):
        return self._record("text", page_access_token, page=page_id, text=text)


def isolated_app():
    """A logged-in test client on a throwaway database.

    Returns ``(dir, database, web, publisher, client)``. The web app, the
    database module and the publisher are all pointed at the same temporary
    file, so a gate can seed rows directly and then drive the real routes.
    """
    directory = tempfile.mkdtemp(prefix="accounts_gate_")
    os.chdir(directory)

    import database
    database.DB_PATH = os.path.join(directory, "insights.db")
    database.init_db(database.DB_PATH)

    import insights_web as web
    import social_publisher as publisher
    web.database.DB_PATH = database.DB_PATH
    publisher.database.DB_PATH = database.DB_PATH

    from werkzeug.security import generate_password_hash
    uid = database.create_user("gate", generate_password_hash("p"),
                               db_path=database.DB_PATH)

    client = web.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = uid
    return directory, database, web, publisher, client


def install_fake_clients(publisher, web=None, **overrides):
    """Point the publisher at recording clients; returns them by platform.

    ``overrides`` takes a platform name mapped to a ready-made client, which is
    how a gate makes exactly one target fail while the rest succeed. Passing
    ``web`` also makes the app treat every platform as having credentials, which
    is what the accounts screen needs before it will offer to connect one.
    """
    clients = {}
    for platform in ("linkedin", "threads", "twitter", "facebook", "instagram"):
        clients[platform] = overrides.get(platform) or RecordingClient(platform)
    publisher.get_linkedin_client = lambda: clients["linkedin"]
    publisher.get_threads_client = lambda: clients["threads"]
    publisher.get_twitter_client = lambda: clients["twitter"]
    publisher.get_facebook_client = lambda: clients["facebook"]
    publisher.get_instagram_client = lambda: clients["instagram"]
    # Instagram's real publisher lives in the web app and reaches for media
    # helpers; the gates test account routing, not media validation.
    publisher.set_instagram_publisher(
        lambda token, content=None, image_url=None,
        standalone_post_id=None, social_post_id=None:
        clients["instagram"].publish_image_post(token, content, image_url)
    )
    if web is not None:
        web.PLATFORM_CLIENTS = {
            platform: (lambda c=client: c) for platform, client in clients.items()
        }
    return clients


FUTURE = "2099-01-01T00:00:00"


def connect(database, platform, external_id, label, db_path=None, token=None):
    """Connect one account on a platform with a token unique to it.

    The token is derived from the account so a gate can tell, from the token a
    fake client was handed, exactly which account published.
    """
    db_path = db_path or database.DB_PATH
    token = token or f"token-{platform}-{external_id}"
    if platform == "linkedin":
        database.save_linkedin_token(
            access_token=token, expires_at=FUTURE, member_id=external_id,
            user_urn=f"urn:li:person:{external_id}", display_name=label,
            email=f"{external_id}@example.com", refresh_token=f"refresh-{external_id}",
            db_path=db_path,
        )
    elif platform == "threads":
        database.save_threads_token(
            access_token=token, expires_at=FUTURE, user_id=external_id,
            username=label, display_name=label, db_path=db_path,
        )
    elif platform == "twitter":
        database.save_twitter_token(
            access_token=token, refresh_token=f"refresh-{external_id}",
            expires_at=FUTURE, user_id=external_id, username=label,
            display_name=label, db_path=db_path,
        )
    elif platform == "facebook":
        database.save_facebook_token(
            access_token=token, expires_at=FUTURE, user_id=external_id,
            user_name=label, page_id=external_id, page_name=label,
            page_access_token=token, db_path=db_path,
        )
    elif platform == "instagram":
        database.save_instagram_token(
            access_token=token, expires_at=FUTURE, user_id=external_id,
            username=label, ig_user_id=external_id, display_name=label,
            db_path=db_path,
        )
    else:
        raise ValueError(f"unknown platform {platform}")

    account = database.find_social_account(platform, external_id, db_path=db_path)
    assert account is not None, f"connecting {platform}:{external_id} made no account"
    return account["id"]


def fail(message):
    """End a gate with a readable reason and a non-zero exit."""
    print(f"FAIL: {message}")
    sys.exit(1)


def check(condition, message):
    """Assert a gate condition, failing the gate rather than raising."""
    if not condition:
        fail(message)
