"""Shared fakes for the Instagram collaborators and people-tag gates.

The Instagram client talks to graph.instagram.com through ``requests``. These
gates replace ``requests.post`` / ``requests.get`` inside ``instagram_client``
with a recorder, so what is asserted is the HTTP request the client would have
sent: method, URL and query parameters in order. Nothing leaves the machine.

``FakeGraph`` answers the way the live API did when the reel flow was proven on
2026-10-05: POST /me/media returns a container id, the container reports
FINISHED, POST /me/media_publish returns a post id, and GET /<post id> returns a
permalink. A test makes it refuse a container (``reject``) to simulate Meta
rejecting a collaborator.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

API = "https://graph.instagram.com"
TOKEN = "tok-ig"


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeGraph:
    """Records every request and answers like the Instagram Graph API.

    ``reject`` is a function ``params -> error payload or None``: it is asked
    about every container creation, and a payload makes that creation fail with
    HTTP 400 and that body.
    """

    def __init__(self, reject=None):
        self.requests = []          # (method, url, params-as-ordered-list)
        self.reject = reject
        self._containers = 0

    # --- what a gate reads back -------------------------------------------
    def creations(self):
        """The query parameters of every POST /me/media, as dicts."""
        return [dict(p) for m, u, p in self.requests
                if m == "POST" and u == f"{API}/me/media"]

    def publishes(self):
        return [dict(p) for m, u, p in self.requests
                if m == "POST" and u == f"{API}/me/media_publish"]

    def signature(self):
        """Every request as (method, url, ordered params), for byte comparison."""
        return [[m, u, [list(kv) for kv in p]] for m, u, p in self.requests]

    # --- the fake transport -----------------------------------------------
    def post(self, url, params=None, data=None, timeout=None, **_kw):
        self.requests.append(("POST", url, list((params or data or {}).items())))
        if url == f"{API}/me/media":
            if self.reject:
                body = self.reject(dict(params or {}))
                if body:
                    return FakeResponse(400, body)
            self._containers += 1
            return FakeResponse(200, {"id": f"c{self._containers}"})
        if url == f"{API}/me/media_publish":
            return FakeResponse(200, {"id": "p1"})
        return FakeResponse(404, {"error": {"message": f"unexpected POST {url}"}})

    def get(self, url, params=None, timeout=None, **_kw):
        self.requests.append(("GET", url, list((params or {}).items())))
        if url == f"{API}/p1":
            return FakeResponse(200, {"permalink": "https://instagram.test/p/x/",
                                      "shortcode": "x"})
        if url.startswith(f"{API}/c"):
            return FakeResponse(200, {"status_code": "FINISHED", "status": "ok"})
        return FakeResponse(404, {"error": {"message": f"unexpected GET {url}"}})


def install_fake_graph(reject=None):
    """Point instagram_client at a FakeGraph; returns (graph, client, restore)."""
    import instagram_client
    graph = FakeGraph(reject=reject)
    original = (instagram_client.requests.post, instagram_client.requests.get,
                instagram_client.time.sleep)
    instagram_client.requests.post = graph.post
    instagram_client.requests.get = graph.get
    instagram_client.time.sleep = lambda *_a, **_k: None

    def restore():
        (instagram_client.requests.post, instagram_client.requests.get,
         instagram_client.time.sleep) = original

    return graph, instagram_client.InstagramClient("app", "secret"), restore


def bad_user_error(name=None):
    """The payload Meta returns for a collaborator it cannot use.

    Shaped like every other Graph error: code 100 with a sub-code. ``name`` puts
    the offending handle in the message, the way the clearest of Meta's
    rejections do; without it the payload says only that a user was invalid.
    """
    message = "Invalid user id" if name is None else f"The user @{name} is not visible"
    return {"error": {"message": message, "type": "OAuthException", "code": 100,
                      "error_subcode": 2207066, "fbtrace_id": "x"}}


def collaborators_in(params):
    """The decoded ``collaborators`` parameter of a creation request, or None."""
    raw = params.get("collaborators")
    return json.loads(raw) if raw is not None else None


def tags_in(params):
    raw = params.get("user_tags")
    return json.loads(raw) if raw is not None else None
