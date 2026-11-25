"""Gate: a LinkedIn connection that fails says what happened and how to fix it.

    python scripts/check_linkedin_connect_error.py     LINKEDIN_CONNECT_ERROR_OK

Drives the real /linkedin/callback route on a throwaway database. The refused
scope case uses the query string LinkedIn really sent on 2026-11-25 (an app
without "Sign In with LinkedIn using OpenID Connect" asked for openid), with its
HTML-escaped quotes. Checks that the page names the permission and the product,
gives both fixes with a LINKEDIN_SCOPES value that would work, offers Try again
back into the same flow, keeps nothing, and no longer shows the article error.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from _accounts_gate import isolated_app, check  # noqa: E402

EM, EN = chr(0x2014), chr(0x2013)
REAL_REFUSAL = ("/linkedin/callback?error=unauthorized_scope_error&error_description="
                "Scope+%26quot;openid%26quot;+is+not+authorized+for+your+application&state=s1")


def text(html):
    """The page's words, with tags dropped and entities read."""
    from html import unescape
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html)))


def page(client, url):
    response = client.get(url)
    html = response.get_data(as_text=True)
    return response.status_code, html, text(html)


def common(html, words, where):
    check("LinkedIn didn't connect" in words, f"{where}: no 'LinkedIn didn't connect' heading")
    check("Could Not Generate Article" not in words and "rate limits" not in words,
          f"{where}: still the article error page")
    check(len(re.findall(r'class="btn btn-primary"', html)) == 1, f"{where}: not exactly one primary action")
    check(re.search(r'href="/linkedin/auth[^"]*"[^>]*>Try again<', html), f"{where}: no Try again into /linkedin/auth")
    check('href="/accounts"' in html and "Back to Accounts" in words, f"{where}: no way back to Accounts")
    check("&amp;quot;" not in html and "&quot;openid&quot;" not in words, f"{where}: LinkedIn's escaped quotes shown raw")
    check(EM not in html and EN not in html, f"{where}: a long dash on the page")


def main():
    directory, database, web, publisher, client = isolated_app()
    import linkedin_client

    # 1. The real refusal, with LINKEDIN_SCOPES unset: the built-in list asks for openid.
    os.environ.pop("LINKEDIN_SCOPES", None)
    linkedin_client.LINKEDIN_SCOPES = "openid profile w_member_social"
    status, html, words = page(client, REAL_REFUSAL)
    check(status == 200, f"refusal: status {status}")
    common(html, words, "refusal")
    for needle in ('not allowed the "openid" permission',
                   '"Sign In with LinkedIn using OpenID Connect"',
                   "Nothing changed in Insights",
                   "its built-in list, because LINKEDIN_SCOPES is not set in .env",
                   'Set LINKEDIN_SCOPES="w_member_social" in .env, restart Insights',
                   "asks you for your Member ID once"):
        check(needle in words, f"refusal: the page does not say {needle!r}: {words[:900]}")
    check(words.index("Products") < words.index("Or stop asking for it"),
          "refusal: adding the product should be the first fix")
    print("refusal: names openid and its product, both fixes, a working LINKEDIN_SCOPES, Try again")

    # 2. A scope set in .env, refusing the Community Management API permission.
    os.environ["LINKEDIN_SCOPES"] = "openid profile w_member_social w_member_social_feed"
    linkedin_client.LINKEDIN_SCOPES = os.environ["LINKEDIN_SCOPES"]
    try:
        status, html, words = page(client, "/linkedin/callback?error=unauthorized_scope_error&error_description="
                                   "Scope+%26quot;w_member_social_feed%26quot;+is+not+authorized+for+your+application")
    finally:
        os.environ.pop("LINKEDIN_SCOPES", None)
    common(html, words, "feed refusal")
    for needle in ('"Community Management API"', "from LINKEDIN_SCOPES in .env",
                   'Set LINKEDIN_SCOPES="openid profile w_member_social" in .env'):
        check(needle in words, f"feed refusal: the page does not say {needle!r}: {words[:900]}")
    check("Member ID" not in words, "feed refusal: the Member ID note belongs to the sign-in scopes only")
    print("feed refusal: names the Community Management API and keeps the other scopes")

    # 3. A refusal LinkedIn words differently still gets steps and LinkedIn's own words.
    linkedin_client.LINKEDIN_SCOPES = "openid profile w_member_social"
    status, html, words = page(client, "/linkedin/callback?error=unauthorized_scope_error"
                               "&error_description=Some+permission+is+not+allowed")
    common(html, words, "unparsed refusal")
    check("refused one of the permissions" in words and "LinkedIn said: Some permission is not allowed" in words
          and "Products tab" in words, f"unparsed refusal: {words[:600]}")
    print("unparsed refusal: general steps and LinkedIn's words")

    # 4. Cancelling, an unknown error, a stale sign-in, a missing code.
    _, html, words = page(client, "/linkedin/callback?error=user_cancelled_authorize&error_description=x")
    common(html, words, "cancel")
    check("You cancelled on LinkedIn" in words, f"cancel: {words[:400]}")
    _, html, words = page(client, "/linkedin/callback?error=server_error&error_description=Try+later")
    common(html, words, "other error")
    check("did not finish connecting" in words and "LinkedIn said: Try later" in words, f"other error: {words[:400]}")
    _, html, words = page(client, "/linkedin/callback?code=abc&state=nope")
    common(html, words, "stale")
    check("expired or was finished in another window" in words, f"stale: {words[:400]}")
    with client.session_transaction() as s:
        s["linkedin_oauth_state"] = "st"
    _, html, words = page(client, "/linkedin/callback?state=st")
    common(html, words, "no code")
    check("without a sign-in code" in words, f"no code: {words[:400]}")
    print("cancel, other errors, a stale sign-in and a missing code each say what happened")

    # 5. The token exchange failing names it, with the detail kept small.
    class Failing:
        def exchange_code_for_token(self, code):
            raise RuntimeError("token endpoint said no")
    original = web.get_linkedin_client
    web.get_linkedin_client = lambda: Failing()
    try:
        with client.session_transaction() as s:
            s["linkedin_oauth_state"] = "st2"
        _, html, words = page(client, "/linkedin/callback?code=c&state=st2")
    finally:
        web.get_linkedin_client = original
    common(html, words, "exchange")
    check("could not finish connecting LinkedIn" in words and "Details: token endpoint said no" in words,
          f"exchange: {words[:400]}")
    print("exchange failure: says so, with the detail")

    # 6. Try again goes back the way the user came: from Accounts, to Accounts.
    with client.session_transaction() as s:
        s["oauth_return"] = "accounts"
    _, html, _ = page(client, REAL_REFUSAL)
    check('href="/linkedin/auth?return=accounts"' in html, "Try again from Accounts loses the return to Accounts")
    print("Try again keeps the return to Accounts")

    # 7. None of this saved a LinkedIn connection.
    import sqlite3
    rows = sqlite3.connect(database.DB_PATH).execute("SELECT COUNT(*) FROM linkedin_tokens").fetchone()[0]
    check(rows == 0, f"a failed connection saved {rows} LinkedIn tokens")
    print("nothing was saved")

    # 8. The helper reads LinkedIn's escaped wording and nothing else.
    check(linkedin_client.refused_scope("Scope &quot;openid&quot; is not authorized for your application") == "openid"
          and linkedin_client.refused_scope('Scope "r_basicprofile" is not authorized') == "r_basicprofile"
          and linkedin_client.refused_scope("no scope named") is None
          and linkedin_client.refused_scope(None) is None, "refused_scope misreads LinkedIn's description")
    print("refused_scope reads the scope out of LinkedIn's description")
    print("LINKEDIN_CONNECT_ERROR_OK")


if __name__ == "__main__":
    main()
