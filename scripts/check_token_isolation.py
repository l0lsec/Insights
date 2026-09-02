"""G2: each account keeps its own credentials, on every platform.

This is the failure mode multi-account support exists to prevent. Before it,
every token table was read with LIMIT 1 and written with an upsert, so a second
login overwrote the first and a token refresh rewrote whatever row it found.
Every platform is checked, not just one, because the five token tables were
five separate copies of the same code and a fix to one proves nothing about the
other four.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import isolated_app, connect, check, FUTURE  # noqa: E402

directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH

GETTERS = {
    "linkedin": database.get_linkedin_token,
    "threads": database.get_threads_token,
    "twitter": database.get_twitter_token,
    "facebook": database.get_facebook_token,
    "instagram": database.get_instagram_token,
}
UPDATERS = {
    "linkedin": database.update_linkedin_token,
    "threads": database.update_threads_token,
    "twitter": database.update_twitter_token,
    "facebook": database.update_facebook_token,
    "instagram": database.update_instagram_token,
}
DELETERS = {
    "linkedin": database.delete_linkedin_token,
    "threads": database.delete_threads_token,
    "twitter": database.delete_twitter_token,
    "facebook": database.delete_facebook_token,
    "instagram": database.delete_instagram_token,
}

checked = 0
for platform in GETTERS:
    first = connect(database, platform, f"{platform}-1", "First")
    second = connect(database, platform, f"{platform}-2", "Second")
    get = GETTERS[platform]

    # Two logins, two tokens, neither overwriting the other.
    token_a = get(first, db_path=P)
    token_b = get(second, db_path=P)
    check(token_a is not None and token_b is not None,
          f"{platform}: an account has no token after connecting")
    check(token_a["access_token"] == f"token-{platform}-{platform}-1",
          f"{platform}: the first account's token was overwritten by the second")
    check(token_b["access_token"] == f"token-{platform}-{platform}-2",
          f"{platform}: the second account did not keep its own token")

    # A bare call means the default account, which is the first one connected.
    check(get(db_path=P)["access_token"] == token_a["access_token"],
          f"{platform}: an unnamed token read did not return the default account")

    # A refresh touches one account only.
    UPDATERS[platform](access_token="refreshed", expires_at=FUTURE,
                       account_id=second, db_path=P)
    check(get(second, db_path=P)["access_token"] == "refreshed",
          f"{platform}: refreshing the second account did not take")
    check(get(first, db_path=P)["access_token"] == token_a["access_token"],
          f"{platform}: refreshing one account rewrote the other's token")

    # A named account that does not exist is not connected, and must never
    # answer with someone else's credentials.
    check(get(99999, db_path=P) is None,
          f"{platform}: an unknown account id returned a token")

    # Disconnecting one account leaves the other posting.
    DELETERS[platform](second, db_path=P)
    check(get(second, db_path=P) is None,
          f"{platform}: the disconnected account still has a token")
    check(get(first, db_path=P) is not None,
          f"{platform}: disconnecting one account took the other's token too")

    # Disconnecting an id that is already gone must not wipe the platform.
    DELETERS[platform](99999, db_path=P)
    check(get(first, db_path=P) is not None,
          f"{platform}: disconnecting an unknown account wiped the platform")
    checked += 1

print(f"token isolation: {checked} platforms keep per-account credentials "
      "through connect, refresh and disconnect")
print("TOKEN_ISOLATION_OK")
