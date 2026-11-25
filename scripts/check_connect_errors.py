"""Gate: a connection that fails says what happened and how to fix it, on every platform.

    python scripts/check_connect_errors.py     CONNECT_ERRORS_OK

Drives the real /<platform>/auth and /<platform>/callback routes of LinkedIn,
Threads, Instagram, Facebook and X on a throwaway database, with fake clients
that never reach the network. For each platform: cancelling, a refused
permission, any other error, a stale sign-in, a missing code and a failed token
exchange each land on the "<Platform> didn't connect" page, never the article
error page, with one primary action, Try again back into the same Connect or
Reconnect button and Back to Accounts. Nothing is saved.

LinkedIn's refused scope uses the query string LinkedIn really sent on
2026-11-25 (an app without "Sign In with LinkedIn using OpenID Connect" asked
for openid), with its HTML-escaped quotes, and must name the permission, the
product and a LINKEDIN_SCOPES value that works.
"""

import os
import re
import sqlite3
import sys
from html import unescape
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from _accounts_gate import isolated_app, check  # noqa: E402

EM, EN = chr(0x2014), chr(0x2013)
REAL_LINKEDIN_REFUSAL = ("/linkedin/callback?error=unauthorized_scope_error&error_description="
                         "Scope+%26quot;openid%26quot;+is+not+authorized+for+your+application&state=s1")

# platform: (name on the page, the Reconnect arguments its auth route takes, portal, setting)
PLATFORMS = {
    "linkedin": ("LinkedIn", {}, "LinkedIn Developer Portal", "LINKEDIN_SCOPES"),
    "threads": ("Threads", {"replies": "1"}, "Meta for Developers app dashboard", "THREADS_SCOPES"),
    "instagram": ("Instagram", {"engage": "1"}, "Meta for Developers app dashboard", "INSTAGRAM_SCOPES"),
    "facebook": ("Facebook", {"engage": "dm"}, "Meta for Developers app dashboard", "FACEBOOK_SCOPES"),
    "twitter": ("X", {}, "X Developer Portal", "TWITTER_SCOPES"),
}


class FakeClient:
    """Configured, hands out a fixed state, and fails the token exchange."""

    def __init__(self, platform):
        self.platform = platform

    def is_configured(self):
        return True

    def get_authorization_url(self, scopes=None, state=None):
        url = f"https://auth.invalid/{self.platform}"
        return (url, "st", "verifier") if self.platform == "twitter" else (url, "st")

    def exchange_code_for_token(self, *args):
        raise RuntimeError(f"{self.platform} token endpoint said no")


def text(html):
    """The page's words, with tags dropped and entities read."""
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html)))


def page(client, url):
    response = client.get(url)
    html = response.get_data(as_text=True)
    return response.status_code, html, text(html)


def retry_href(html):
    match = re.search(r'href="([^"]*)"[^>]*>Try again<', html)
    return unescape(match.group(1)) if match else None


def common(html, words, name, where, platform, args=None):
    check(f"{name} didn't connect" in words, f"{where}: no '{name} didn't connect' heading: {words[:300]}")
    check("Could Not Generate Article" not in words and "rate limits" not in words,
          f"{where}: still the article error page")
    check(len(re.findall(r'class="btn btn-primary"', html)) == 1, f"{where}: not exactly one primary action")
    href = retry_href(html)
    check(href and urlparse(href).path == f"/{platform}/auth", f"{where}: Try again goes to {href}")
    if args is not None:
        got = {k: v[0] for k, v in parse_qs(urlparse(href).query).items()}
        check(got == args, f"{where}: Try again repeats {got}, not the start's {args}")
    check('href="/accounts"' in html and "Back to Accounts" in words, f"{where}: no way back to Accounts")
    check("&amp;quot;" not in html and "&quot;" not in words, f"{where}: escaped quotes shown raw")
    check(EM not in html and EN not in html, f"{where}: a long dash on the page")


def start(client, platform, args):
    """Press Connect (or the Reconnect button) from Accounts."""
    query = "&".join(f"{k}={v}" for k, v in {"return": "accounts", **args}.items())
    response = client.get(f"/{platform}/auth?{query}")
    check(response.status_code == 302 and "auth.invalid" in response.headers.get("Location", ""),
          f"{platform}: /auth did not send the user to the platform: {response.status_code}")


def every_platform(web, client):
    for platform, (name, extra, portal, setting) in PLATFORMS.items():
        args = {"return": "accounts", **extra}
        setattr(web, f"get_{platform}_client", lambda p=platform: FakeClient(p))

        start(client, platform, extra)
        _, html, words = page(client, f"/{platform}/callback?error=access_denied&error_reason=user_denied"
                                      f"&error_description=Permissions+error")
        common(html, words, name, f"{platform} cancel", platform, args)
        check(f"You cancelled on {name}" in words and "nothing was connected" in words, f"{platform} cancel: {words[:300]}")

        _, html, words = page(client, f"/{platform}/callback?error=invalid_scope"
                                      f"&error_description=The+requested+scope+is+invalid")
        common(html, words, name, f"{platform} scope", platform, args)
        for needle in (f"{name} refused one of the permissions", f"In the {portal}", f"{setting} in .env",
                       "Then press Try again", f"{name} said: The requested scope is invalid"):
            check(needle in words, f"{platform} scope: the page does not say {needle!r}: {words[:700]}")
        check(words.index(f"In the {portal}") < words.index("Then press Try again"),
              f"{platform} scope: steps out of order")

        _, html, words = page(client, f"/{platform}/callback?error=server_error&error_description=Try+later")
        common(html, words, name, f"{platform} other", platform, args)
        check(f"{name} did not finish connecting" in words and f"{name} said: Try later" in words,
              f"{platform} other: {words[:300]}")

        _, html, words = page(client, f"/{platform}/callback?code=abc&state=nope")
        common(html, words, name, f"{platform} stale", platform, args)
        check(f"This {name} sign-in expired or was finished in another window" in words, f"{platform} stale: {words[:300]}")

        start(client, platform, extra)
        _, html, words = page(client, f"/{platform}/callback?state=st")
        common(html, words, name, f"{platform} no code", platform, args)
        check(f"{name} came back without a sign-in code" in words, f"{platform} no code: {words[:300]}")

        start(client, platform, extra)
        _, html, words = page(client, f"/{platform}/callback?code=c&state=st")
        common(html, words, name, f"{platform} exchange", platform, args)
        check(f"Insights could not finish connecting {name}." in words
              and f"Details: {platform} token endpoint said no" in words, f"{platform} exchange: {words[:300]}")

        # Plain Connect, not from Accounts: Try again is plain Connect too.
        client.get(f"/{platform}/auth")
        _, html, _ = page(client, f"/{platform}/callback?error=server_error")
        check(retry_href(html) == f"/{platform}/auth", f"{platform}: plain Connect retries as {retry_href(html)}")
        print(f"{platform}: cancel, refused permission, other error, stale, no code and failed exchange "
              f"each say what happened; Try again repeats {args}")

    # X keeps its PKCE verifier in the session: a callback without it is a stale sign-in.
    start(client, "twitter", {})
    with client.session_transaction() as s:
        s.pop("twitter_code_verifier", None)
    _, html, words = page(client, "/twitter/callback?code=c&state=st")
    common(html, words, "X", "twitter no verifier", "twitter")
    check("This X sign-in expired" in words, f"twitter no verifier: {words[:300]}")
    print("twitter: a callback without its PKCE verifier reads as a stale sign-in")


def linkedin_scopes(web, client):
    import linkedin_client
    start(client, "linkedin", {})

    # The real refusal, with LINKEDIN_SCOPES unset: the built-in list asks for openid.
    os.environ.pop("LINKEDIN_SCOPES", None)
    linkedin_client.LINKEDIN_SCOPES = "openid profile w_member_social"
    status, html, words = page(client, REAL_LINKEDIN_REFUSAL)
    check(status == 200, f"linkedin refusal: status {status}")
    common(html, words, "LinkedIn", "linkedin refusal", "linkedin", {"return": "accounts"})
    for needle in ('not allowed the "openid" permission',
                   '"Sign In with LinkedIn using OpenID Connect"',
                   "Nothing changed in Insights",
                   "its built-in list, because LINKEDIN_SCOPES is not set in .env",
                   'Set LINKEDIN_SCOPES="w_member_social" in .env, restart Insights',
                   'Without "openid", Insights cannot read your LinkedIn name',
                   "asks you for your Member ID once"):
        check(needle in words, f"linkedin refusal: the page does not say {needle!r}: {words[:900]}")
    check(words.index("Products") < words.index("Or stop asking for it"),
          "linkedin refusal: adding the product should be the first fix")

    # A scope set in .env, refusing the Community Management API permission.
    os.environ["LINKEDIN_SCOPES"] = "openid profile w_member_social w_member_social_feed"
    linkedin_client.LINKEDIN_SCOPES = os.environ["LINKEDIN_SCOPES"]
    try:
        _, html, words = page(client, "/linkedin/callback?error=unauthorized_scope_error&error_description="
                                      "Scope+%26quot;w_member_social_feed%26quot;+is+not+authorized+for+your+application")
    finally:
        os.environ.pop("LINKEDIN_SCOPES", None)
    common(html, words, "LinkedIn", "linkedin feed refusal", "linkedin")
    for needle in ('"Community Management API"', "from LINKEDIN_SCOPES in .env",
                   'Set LINKEDIN_SCOPES="openid profile w_member_social" in .env'):
        check(needle in words, f"linkedin feed refusal: the page does not say {needle!r}: {words[:900]}")
    check("Member ID" not in words, "linkedin feed refusal: the Member ID note belongs to the sign-in scopes only")

    # Worded so no scope can be read out: the general refused-permission page.
    linkedin_client.LINKEDIN_SCOPES = "openid profile w_member_social"
    _, html, words = page(client, "/linkedin/callback?error=unauthorized_scope_error"
                                  "&error_description=Some+permission+is+not+allowed")
    common(html, words, "LinkedIn", "linkedin unparsed refusal", "linkedin")
    check("LinkedIn refused one of the permissions" in words and "LinkedIn Developer Portal" in words
          and "LinkedIn said: Some permission is not allowed" in words, f"linkedin unparsed refusal: {words[:600]}")

    # LinkedIn's own cancel codes.
    _, html, words = page(client, "/linkedin/callback?error=user_cancelled_authorize&error_description=x")
    check("You cancelled on LinkedIn" in words, f"linkedin cancel code: {words[:300]}")

    check(linkedin_client.refused_scope("Scope &quot;openid&quot; is not authorized for your application") == "openid"
          and linkedin_client.refused_scope('Scope "r_basicprofile" is not authorized') == "r_basicprofile"
          and linkedin_client.refused_scope("no scope named") is None
          and linkedin_client.refused_scope(None) is None, "refused_scope misreads LinkedIn's description")
    print("linkedin: a refused scope names the permission, its product, and a LINKEDIN_SCOPES that works")


def no_callback_uses_article_error():
    source = open(os.path.join(ROOT, "insights_web.py")).read()
    for platform in PLATFORMS:
        body = re.search(rf"^def {platform}_callback\(\):\n(.*?)(?=^@app\.route|^def )", source, re.M | re.S)
        check(body, f"{platform}_callback not found")
        check("article_error.html" not in body.group(1), f"{platform}_callback still renders article_error.html")
    print("no connect callback renders the article error page")


def main():
    directory, database, web, publisher, client = isolated_app()
    web._maybe_attach_link_image = lambda *a, **k: None
    every_platform(web, client)
    linkedin_scopes(web, client)
    no_callback_uses_article_error()

    conn = sqlite3.connect(database.DB_PATH)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_tokens'")]
    saved = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    check(tables and not any(saved.values()), f"a failed connection saved tokens: {saved}")
    print(f"nothing was saved ({len(tables)} token tables empty)")
    print("CONNECT_ERRORS_OK")


if __name__ == "__main__":
    main()
