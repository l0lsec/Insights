"""Gates for posting videos.

One script, one section per claim, picked by argument. Each section proves its
claim against the real code and prints its own token only at the end:

    python scripts/check_video.py schema     VIDEO_SCHEMA_OK
    python scripts/check_video.py router     VIDEO_ROUTER_OK
    python scripts/check_video.py linkedin   VIDEO_LINKEDIN_OK
    python scripts/check_video.py twitter    VIDEO_TWITTER_OK
    python scripts/check_video.py threads    VIDEO_THREADS_OK
    python scripts/check_video.py facebook   VIDEO_FACEBOOK_OK
    python scripts/check_video.py instagram  VIDEO_INSTAGRAM_OK
    python scripts/check_video.py routes     VIDEO_ROUTES_OK
    python scripts/check_video.py fanout     VIDEO_FANOUT_OK
    python scripts/check_video.py copy       VIDEO_COPY_OK
    python scripts/check_video.py compat     VIDEO_COMPAT_OK
    python scripts/check_video.py download   VIDEO_DOWNLOAD_OK
    python scripts/check_video.py ui         VIDEO_UI_OK

The client sections drive the real platform clients against a fake HTTP layer
that records every request, then compare a hash of the bytes the "platform"
received with a hash of the file that was served. Nothing here posts to a real
account or reaches a real host; the platforms' request shapes come from their
published API documentation, so they prove the client speaks the documented
protocol, not that the live service accepts it.
"""

import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _video_gate import (  # noqa: E402
    FakeApi, FakeResponse, VideoServer, MB, check, fail, form_bytes,
    make_sample_mp4, no_wait, sha, temp_videos,
)

VIDEO = "https://cdn.example.test/clip.mp4"
LI_PARTS = 4 * MB


def trim(value, limit=240):
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def section_schema():
    from _accounts_gate import isolated_app
    directory, database, web, publisher, client = isolated_app()
    P = database.DB_PATH

    # A saved post carries a video, and it reads back, changes and clears.
    pid = database.add_standalone_post("manual", "gate", "linkedin", "Copy", video_url=VIDEO, db_path=P)
    check(database.get_standalone_post(pid, db_path=P)["video_url"] == VIDEO,
          "video_url did not persist on a new post")
    database.update_standalone_post_video(pid, "https://cdn.example.test/other.mp4", db_path=P)
    check(database.get_standalone_post(pid, db_path=P)["video_url"] == "https://cdn.example.test/other.mp4",
          "video_url did not update")
    database.update_standalone_post_video(pid, None, db_path=P)
    check(database.get_standalone_post(pid, db_path=P)["video_url"] is None, "video_url did not clear")
    plain = database.add_standalone_post("manual", "gate", "linkedin", "No video", db_path=P)
    check(database.get_standalone_post(plain, db_path=P)["video_url"] is None,
          "a post saved without a video should have none")

    # An install from before videos existed upgrades in place and keeps its rows.
    old = os.path.join(directory, "old.db")
    with sqlite3.connect(old) as conn:
        conn.execute(
            "CREATE TABLE standalone_posts (id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT, "
            "source_content TEXT, platform TEXT, content TEXT, image_url TEXT, created_at TEXT, "
            "used INTEGER DEFAULT 0, repost INTEGER DEFAULT 0)")
        conn.execute("INSERT INTO standalone_posts (source_type, source_content, platform, content, image_url, "
                     "created_at) VALUES ('m','s','linkedin','Legacy copy','https://x.test/i.jpg','2024-01-01')")
    database.init_db(old)
    database.init_db(old)  # idempotent
    with sqlite3.connect(old) as conn:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(standalone_posts)")]
        row = conn.execute("SELECT content, image_url, video_url FROM standalone_posts").fetchone()
    check(columns.count("video_url") == 1, f"video_url column missing or duplicated: {columns}")
    check(row == ("Legacy copy", "https://x.test/i.jpg", None), f"legacy row was altered by the upgrade: {row}")

    # The video is part of what makes a card: same words + different video are
    # different cards, same words + same video (or none) are one.
    def rows(*specs):
        ids = [database.add_standalone_post("manual", "g", plat, "Shared words", video_url=vid, db_path=P)
               for plat, vid in specs]
        return [database.get_standalone_post(i, db_path=P) for i in ids]

    same = web._group_standalone_posts(rows(("threads", VIDEO), ("twitter", VIDEO)))
    check(len(same) == 1 and len(same[0]["platforms"]) == 2,
          f"rows sharing copy and video should be one card: {len(same)} cards")
    split = web._group_standalone_posts(rows(("facebook", VIDEO), ("instagram", "https://cdn.example.test/z.mp4")))
    check(len(split) == 2, f"rows with different videos must be different cards: {len(split)} cards")
    bare = web._group_standalone_posts(rows(("linkedin", None), ("threads", "")))
    check(len(bare) == 1, "a NULL video and an empty one should group together")

    print("video_url persists, upgrades an old database in place, and groups cards")
    print("VIDEO_SCHEMA_OK")


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------


def section_router():
    from _accounts_gate import isolated_app, install_fake_clients, connect
    directory, database, web, publisher, client = isolated_app()
    clients = install_fake_clients(publisher, web)
    accounts = {p: connect(database, p, f"{p}-1", f"{p}-label")
                for p in ("linkedin", "threads", "twitter", "facebook", "instagram")}
    tokens = {p: f"token-{p}-{p}-1" for p in accounts}

    for platform, account_id in accounts.items():
        c = clients[platform]
        before = len(c.calls)
        result = publisher.publish(platform, account_id, content="Words",
                                   image_url="https://x.test/i.jpg", video_url=VIDEO)
        check(result["success"], f"{platform} video publish failed: {result}")
        calls = c.calls[before:]
        check(len(calls) == 1 and calls[0]["kind"] == "video",
              f"{platform}: a video post must make exactly one video call, made {[x['kind'] for x in calls]}")
        check(calls[0]["video"] == VIDEO, f"{platform}: wrong video URL handed over: {calls[0]}")
        check(calls[0]["token"] == tokens[platform], f"{platform}: wrong account's token: {calls[0]['token']}")

        # Positive control: with no video the same publisher still takes the old
        # route, so "never the image method" above is a real absence.
        before = len(c.calls)
        publisher.publish(platform, account_id, content="Words", image_url="https://x.test/i.jpg")
        kinds = [x["kind"] for x in c.calls[before:]]
        check(kinds and kinds[0] in ("image", "text") and "video" not in kinds,
              f"{platform}: a post without a video should not send one: {kinds}")

    # A failing video is a failure, with the platform's reason, and the
    # 'this can never work' flag survives so the scheduler stops retrying.
    class Refusing:
        def __init__(self, guard):
            self.guard = guard

        def create_video_post(self, **kwargs):
            return {"success": False, "error": {"message": "video too long"}, "guard_error": self.guard}

        def publish_video_post(self, *args, **kwargs):
            return self.create_video_post(**kwargs)

    publisher.get_threads_client = lambda: Refusing(True)
    result = publisher.publish("threads", accounts["threads"], content="Words", video_url=VIDEO)
    check(not result["success"] and "video too long" in result["error"] and result["permanent"],
          f"a refused video should fail permanently with its reason: {result}")
    publisher.get_threads_client = lambda: Refusing(False)
    result = publisher.publish("threads", accounts["threads"], content="Words", video_url=VIDEO)
    check(not result["success"] and not result["permanent"], f"a retryable failure was marked permanent: {result}")

    class NeedsMedia:
        def create_video_post(self, **kwargs):
            return {"success": False, "error": {"message": "reconnect X"}, "needs_reconnect": True,
                    "guard_error": True}

    publisher.get_twitter_client = lambda: NeedsMedia()
    result = publisher.publish("twitter", accounts["twitter"], content="Words", video_url=VIDEO)
    check(not result["success"] and result["needs_reconnect"], f"needs_reconnect was dropped: {result}")

    # publish_targets carries the video to every target.
    publisher.get_linkedin_client = lambda: clients["linkedin"]
    publisher.get_facebook_client = lambda: clients["facebook"]
    before = len(clients["linkedin"].calls), len(clients["facebook"].calls)
    results = publisher.publish_targets(
        [{"platform": "linkedin", "account_id": accounts["linkedin"]},
         {"platform": "facebook", "account_id": accounts["facebook"]}],
        content="Words", video_url=VIDEO)
    check(all(r["success"] for r in results), f"publish_targets with a video failed: {results}")
    check(clients["linkedin"].calls[before[0]]["kind"] == "video"
          and clients["facebook"].calls[before[1]]["kind"] == "video",
          "publish_targets did not deliver the video to each target")

    print("the shared publisher sends a video to each platform's video method with the right token")
    print("VIDEO_ROUTER_OK")


# ---------------------------------------------------------------------------
# linkedin
# ---------------------------------------------------------------------------


def _linkedin_api(api, *, parts, statuses, put_status=200, etag=True, init_status=200,
                  post_status=201):
    """Register LinkedIn's Videos API + Posts API on ``api``; returns a state dict."""
    state = {"puts": [], "polls": 0}
    urn = "urn:li:video:C5505AQTEST"

    def init(call):
        body = call["json"]["initializeUploadRequest"]
        state["init"] = body
        state["init_headers"] = call["headers"]
        if init_status != 200:
            return FakeResponse(init_status, {"message": "not allowed"})
        size = body["fileSizeBytes"]
        instructions = []
        for i in range(parts):
            first = i * LI_PARTS
            last = min(first + LI_PARTS, size) - 1
            instructions.append({"uploadUrl": f"https://www.linkedin.com/dms-uploads/{i}",
                                 "firstByte": first, "lastByte": last})
        return FakeResponse(200, {"value": {"video": urn, "uploadToken": "tok-1",
                                            "uploadUrlsExpireAt": 1, "uploadInstructions": instructions}})

    def put(call):
        state["puts"].append(call)
        i = len(state["puts"]) - 1
        headers = {"etag": f'"etag-{i}"'} if etag else {}
        return FakeResponse(put_status, {}, headers)

    def finalize(call):
        state["finalize"] = call["json"]["finalizeUploadRequest"]
        return FakeResponse(200, {})

    def poll(call):
        state["polls"] += 1
        status = statuses[min(state["polls"] - 1, len(statuses) - 1)]
        body = {"status": status}
        if status == "PROCESSING_FAILED":
            body["processingFailureReason"] = "unsupported codec"
        return FakeResponse(200, body)

    def post(call):
        state["post"] = call["json"]
        if post_status != 201:
            return FakeResponse(post_status, {"message": "denied"})
        return FakeResponse(201, {}, {"x-restli-id": "urn:li:share:777"})

    api.on("POST", "action=initializeUpload", init)
    api.on("PUT", "dms-uploads", put)
    api.on("POST", "action=finalizeUpload", finalize)
    api.on("GET", "/rest/videos/", poll)
    api.on("POST", "/rest/posts", post)
    state["urn"] = urn
    return state


def section_linkedin():
    import linkedin_client
    no_wait(linkedin_client)
    data = os.urandom(9 * MB + 123_457)  # three parts, the last one ragged
    routes = {"/clip.mp4": {"body": data, "type": "video/mp4"}}
    client = linkedin_client.LinkedInClient("id", "secret")
    owner = "urn:li:person:ABC"

    def run(state_kwargs, url_path="/clip.mp4", server=None):
        with FakeApi() as api:
            state = _linkedin_api(api, parts=3, **state_kwargs)
            result = client.create_video_post("li-token", owner, "Caption text", server.url(url_path))
            return api, state, result

    leaked_before = temp_videos()
    with VideoServer(routes) as server:
        # --- the whole protocol, byte for byte ---------------------------
        api, state, result = run({"statuses": ["PROCESSING", "PROCESSING", "AVAILABLE"]}, server=server)
        check(result.get("success"), f"the happy path failed: {result}")
        check(result.get("post_urn") == "urn:li:share:777", f"wrong post urn: {result}")
        check(state["init"] == {"owner": owner, "fileSizeBytes": len(data),
                                "uploadCaptions": False, "uploadThumbnail": False},
              f"initializeUpload body is not the documented shape: {state['init']}")
        check(state["init_headers"].get("LinkedIn-Version") and
              state["init_headers"].get("X-Restli-Protocol-Version") == "2.0.0"
              and state["init_headers"]["Authorization"] == "Bearer li-token",
              f"initializeUpload is missing its required headers: {state['init_headers']}")
        check(len(state["puts"]) == 3, f"expected 3 part uploads, saw {len(state['puts'])}")
        check(all(p["headers"].get("Content-Type") == "application/octet-stream" for p in state["puts"]),
              "parts must be uploaded as application/octet-stream")
        received = b"".join(p["data"] for p in state["puts"])
        check(sha(received) == sha(data), "the bytes LinkedIn received are not the bytes that were served")
        check([len(p["data"]) for p in state["puts"][:2]] == [LI_PARTS, LI_PARTS],
              "parts must be 4 MB each except the last")
        check(state["finalize"] == {"video": state["urn"], "uploadToken": "tok-1",
                                    "uploadedPartIds": ["etag-0", "etag-1", "etag-2"]},
              f"finalizeUpload must list the ETags in order, unquoted: {state['finalize']}")
        check(state["polls"] == 3, f"should wait for AVAILABLE (3 polls), polled {state['polls']}")
        check(api.calls.index(next(c for c in api.calls if "/rest/posts" in c["url"]))
              > api.calls.index(next(c for c in api.calls if "action=finalizeUpload" in c["url"])),
              "the post was created before the upload was finalized")
        payload = state["post"]
        check(payload["content"] == {"media": {"id": state["urn"]}} and payload["commentary"] == "Caption text"
              and payload["author"] == owner and payload["lifecycleState"] == "PUBLISHED",
              f"post payload does not reference the video: {trim(payload)}")

        # --- every failure fails, and nothing is posted -------------------
        cases = [
            ("init refused", {"statuses": ["AVAILABLE"], "init_status": 403}, "403", True),
            ("part rejected", {"statuses": ["AVAILABLE"], "put_status": 500}, "part 1", False),
            ("no ETag", {"statuses": ["AVAILABLE"], "etag": False}, "ETag", False),
            ("processing failed", {"statuses": ["PROCESSING", "PROCESSING_FAILED"]}, "unsupported codec", True),
        ]
        for name, kwargs, needle, permanent in cases:
            api, state, result = run(kwargs, server=server)
            check(result.get("success") is False, f"{name}: should have failed, got {result}")
            message = result["error"]["message"]
            check(needle in message, f"{name}: the reason should mention {needle!r}: {message}")
            check(bool(result.get("guard_error")) == permanent, f"{name}: guard_error should be {permanent}: {result}")
            check("post" not in state, f"{name}: a post was created even though the video failed")
        # never AVAILABLE: retryable, and it does wait a bounded number of times
        old_attempts = linkedin_client.VIDEO_POLL_ATTEMPTS
        linkedin_client.VIDEO_POLL_ATTEMPTS = 4
        try:
            api, state, result = run({"statuses": ["PROCESSING"]}, server=server)
        finally:
            linkedin_client.VIDEO_POLL_ATTEMPTS = old_attempts
        check(result.get("success") is False and "still processing" in result["error"]["message"]
              and not result.get("guard_error") and state["polls"] == 4 and "post" not in state,
              f"a video that never finishes should fail retryably after a bounded wait: {result} polls={state['polls']}")

        # too big for LinkedIn: refused before any LinkedIn call at all
        limits = linkedin_client.video_media.PLATFORM_LIMITS["linkedin"]
        keep = limits["max_bytes"]
        limits["max_bytes"] = 1 * MB
        try:
            api, state, result = run({"statuses": ["AVAILABLE"]}, server=server)
        finally:
            limits["max_bytes"] = keep
        check(result.get("success") is False and result.get("guard_error") and not api.calls,
              f"an oversize video must be refused permanently before touching LinkedIn: {result} calls={len(api.calls)}")

        # a URL that is not a video: same, and nothing sent
        server.httpd.routes["/page"] = {"body": b"<html>login</html>", "type": "text/html"}
        api, state, result = run({"statuses": ["AVAILABLE"]}, url_path="/page", server=server)
        check(result.get("success") is False and not api.calls and result.get("guard_error"),
              f"a web page must not be uploaded as a video: {result}")
        api, state, result = run({"statuses": ["AVAILABLE"]}, url_path="/missing", server=server)
        check(result.get("success") is False and "404" in result["error"]["message"] and result.get("guard_error"),
              f"a 404 should fail permanently: {result}")

    check(temp_videos() <= leaked_before, f"temp video files were left behind: {temp_videos() - leaked_before}")
    print("LinkedIn gets the exact bytes in 4 MB parts with ordered ETags, waits for AVAILABLE, and every failure posts nothing")
    print("VIDEO_LINKEDIN_OK")


# ---------------------------------------------------------------------------
# twitter
# ---------------------------------------------------------------------------


def _twitter_api(api, *, states, init_status=200, append_status=204, media_id="1880028106020515840"):
    state = {"appends": [], "status_calls": 0}

    def init(call):
        state["init"] = call["json"]
        state["init_headers"] = call["headers"]
        if init_status != 200:
            return FakeResponse(init_status, {"detail": "Forbidden"})
        return FakeResponse(200, {"data": {"id": media_id, "media_key": "13_" + media_id,
                                           "expires_after_secs": 86400}})

    def append(call):
        state["appends"].append(call)
        return FakeResponse(append_status, {})

    def finalize(call):
        state["finalized"] = True
        first = states[0]
        info = None if first is None else {"state": first, "check_after_secs": 1}
        if first == "failed":
            info["error"] = {"message": "InvalidMedia"}
        return FakeResponse(200, {"data": {"id": media_id, "processing_info": info} if info
                                  else {"id": media_id}})

    def status(call):
        state["status_calls"] += 1
        state["last_status_params"] = call["params"]
        current = states[min(state["status_calls"], len(states) - 1)]
        info = {"state": current, "check_after_secs": 1}
        if current == "failed":
            info["error"] = {"message": "InvalidMedia"}
        return FakeResponse(200, {"data": {"id": media_id, "processing_info": info}})

    def tweet(call):
        state["tweet"] = call["json"]
        return FakeResponse(201, {"data": {"id": "999", "text": call["json"]["text"]}})

    api.on("POST", "/2/media/upload/initialize", init)
    api.on("POST", "/append", append)
    api.on("POST", "/finalize", finalize)
    api.on("GET", "/2/media/upload", status)
    api.on("POST", "/2/tweets", tweet)
    state["media_id"] = media_id
    return state


def section_twitter():
    import twitter_client
    no_wait(twitter_client)
    check("media.write" in twitter_client.TWITTER_SCOPES.split(),
          f"the default X scopes must include media.write: {twitter_client.TWITTER_SCOPES}")
    data = os.urandom(11 * MB + 4_321)  # three segments of at most 5 MB
    routes = {"/clip.mp4": {"body": data, "type": "video/mp4"},
              "/clip.mov": {"body": data, "type": "video/quicktime"}}
    client = twitter_client.TwitterClient("id", "secret")

    def run(server, path="/clip.mp4", **kwargs):
        with FakeApi() as api:
            state = _twitter_api(api, **kwargs)
            result = client.create_video_post("tw-token", "Caption", server.url(path))
            return api, state, result

    leaked_before = temp_videos()
    with VideoServer(routes) as server:
        api, state, result = run(server, states=["pending", "in_progress", "succeeded"])
        check(result.get("success"), f"the happy path failed: {result}")
        check(state["init"] == {"media_type": "video/mp4", "total_bytes": len(data),
                                "media_category": "tweet_video"}, f"initialize body wrong: {state['init']}")
        check(state["init_headers"]["Authorization"] == "Bearer tw-token", "initialize is not authenticated")
        appends = state["appends"]
        indexes = [a["data"]["segment_index"] for a in appends]
        check(indexes == [str(i) for i in range(len(appends))] and len(appends) == 3,
              f"segments must be numbered 0..n-1 in order: {indexes}")
        sizes = [len(form_bytes(a, "media")) for a in appends]
        check(all(size <= 5 * MB for size in sizes), f"a segment is over X's 5 MB limit: {sizes}")
        check(sha(b"".join(form_bytes(a, "media") for a in appends)) == sha(data),
              "the bytes X received are not the bytes that were served")
        check(all(f"/{state['media_id']}/append" in a["url"] for a in appends), "append addressed the wrong media")
        check(state["last_status_params"] == {"command": "STATUS", "media_id": state["media_id"]},
              f"STATUS poll wrong: {state['last_status_params']}")
        check(state["status_calls"] == 2, f"should poll until succeeded (2 status calls), made {state['status_calls']}")
        check(state["tweet"] == {"text": "Caption", "media": {"media_ids": [state["media_id"]]}},
              f"the post must attach the uploaded media id: {state['tweet']}")
        finalize_at = next(i for i, c in enumerate(api.calls) if "/finalize" in c["url"])
        tweet_at = next(i for i, c in enumerate(api.calls) if "/2/tweets" in c["url"])
        check(tweet_at > finalize_at, "tweeted before the upload was finalized")

        # a .mov is declared as QuickTime; media that needs no processing skips polling
        api, state, result = run(server, path="/clip.mov", states=[None])
        check(result.get("success") and state["init"]["media_type"] == "video/quicktime"
              and state["status_calls"] == 0, f".mov / no-processing path wrong: {result} {state.get('init')}")

        # missing media.write -> reconnect, nothing tweeted
        api, state, result = run(server, states=["succeeded"], init_status=403)
        check(result.get("success") is False and result.get("needs_reconnect") and "media.write" in result["error"]["message"]
              and "tweet" not in state and not state["appends"],
              f"a 403 should ask for a reconnect and post nothing: {result}")

        # 401 means the same thing to the person: the login can't upload media
        api, state, result = run(server, states=["succeeded"], init_status=401)
        check(result.get("success") is False and result.get("needs_reconnect") and "tweet" not in state,
              f"a 401 should also ask for a reconnect: {result}")

        # append fails / processing fails / never finishes
        api, state, result = run(server, states=["succeeded"], append_status=500)
        check(result.get("success") is False and "segment 1" in result["error"]["message"]
              and "tweet" not in state and "finalized" not in state,
              f"a rejected segment should stop the upload: {result}")
        api, state, result = run(server, states=["pending", "failed"])
        check(result.get("success") is False and "InvalidMedia" in result["error"]["message"]
              and result.get("guard_error") and "tweet" not in state,
              f"failed processing should fail permanently with X's reason: {result}")
        old = twitter_client.VIDEO_POLL_ATTEMPTS
        twitter_client.VIDEO_POLL_ATTEMPTS = 3
        try:
            api, state, result = run(server, states=["pending", "in_progress"])
        finally:
            twitter_client.VIDEO_POLL_ATTEMPTS = old
        check(result.get("success") is False and "still processing" in result["error"]["message"]
              and not result.get("guard_error") and "tweet" not in state,
              f"a video that never finishes should fail retryably: {result}")
    check(temp_videos() <= leaked_before, "temp video files were left behind")

    # The image path shares the initialize reply, and the v2 shape is
    # {"data": {"id": ...}}: it must read it (it used to look for "media_id").
    check(twitter_client._media_id({"data": {"id": "1"}}) == "1"
          and twitter_client._media_id({"media_id": 2}) == "2"
          and twitter_client._media_id({"data": {"media_id": 3}}) == "3"
          and twitter_client._media_id({"data": {}}) is None, "media id parsing is wrong")
    with FakeApi() as api:
        state = _twitter_api(api, states=[None], media_id="55")
        api.on("GET", "https://img.example.test/pic.jpg",
               lambda call: _Image())
        media_id = client.upload_media("tw-token", "https://img.example.test/pic.jpg")
    check(media_id == "55", f"image upload no longer finds the media id in X's v2 reply: {media_id}")

    print("X gets the exact bytes in <=5 MB segments, waits for processing, and a missing media.write asks for a reconnect")
    print("VIDEO_TWITTER_OK")


class _Image(FakeResponse):
    def __init__(self):
        super().__init__(200, {})
        self.content = b"\xff\xd8\xff" + b"x" * 100
        self.headers["Content-Type"] = "image/jpeg"

    def raise_for_status(self):
        pass


# ---------------------------------------------------------------------------
# threads
# ---------------------------------------------------------------------------


def _threads_api(api, *, statuses, create_status=200, publish_status=200, error_message=None):
    state = {"order": [], "polls": 0}

    def create(call):
        state["create"] = call["params"]
        state["order"].append("create")
        if create_status != 200:
            return FakeResponse(create_status, {"error": {"message": "bad container"}})
        return FakeResponse(200, {"id": "container-1"})

    def status(call):
        state["polls"] += 1
        value = statuses[min(state["polls"] - 1, len(statuses) - 1)]
        state["order"].append(value)
        body = {"status": value}
        if value == "ERROR":
            body["error_message"] = error_message or "Unsupported video"
        return FakeResponse(200, body)

    def publish(call):
        state["publish"] = call["params"]
        state["order"].append("publish")
        return FakeResponse(publish_status, {"id": "post-9"})

    def details(call):
        return FakeResponse(200, {"permalink": "https://www.threads.net/@x/post/abc", "shortcode": "abc"})

    api.on("POST", "/me/threads_publish", publish)
    api.on("POST", "/me/threads", create)
    api.on("GET", "container-1", status)
    api.on("GET", "post-9", details)
    return state


def section_threads():
    import threads_client
    no_wait(threads_client)
    client = threads_client.ThreadsClient("id", "secret")

    # --- video ---------------------------------------------------------
    with FakeApi() as api:
        state = _threads_api(api, statuses=["IN_PROGRESS", "IN_PROGRESS", "FINISHED"])
        result = client.publish_video_post("th-token", "Caption", VIDEO)
    check(result.get("success") and result["post_id"] == "post-9" and result["permalink"].endswith("/abc"),
          f"video publish failed: {result}")
    check(state["create"]["media_type"] == "VIDEO" and state["create"]["video_url"] == VIDEO
          and state["create"]["text"] == "Caption" and "image_url" not in state["create"],
          f"container params are wrong for a video: {state['create']}")
    check(state["order"] == ["create", "IN_PROGRESS", "IN_PROGRESS", "FINISHED", "publish"],
          f"must wait for FINISHED before publishing: {state['order']}")
    check(state["publish"]["creation_id"] == "container-1", "published the wrong container")

    # --- video failures ------------------------------------------------
    with FakeApi() as api:
        state = _threads_api(api, statuses=["IN_PROGRESS", "ERROR"])
        result = client.publish_video_post("th-token", "Caption", VIDEO)
    check(result.get("success") is False and "Unsupported video" in result["error"]["message"]
          and result.get("guard_error") and "publish" not in state,
          f"a rejected video should fail permanently and not publish: {result}")
    with FakeApi() as api:
        state = _threads_api(api, statuses=["IN_PROGRESS"])
        old = threads_client.VIDEO_POLL_ATTEMPTS
        threads_client.VIDEO_POLL_ATTEMPTS = 5
        try:
            result = client.publish_video_post("th-token", "Caption", VIDEO)
        finally:
            threads_client.VIDEO_POLL_ATTEMPTS = old
    check(result.get("success") is False and "Video processing timed out" in result["error"]["message"]
          and not result.get("guard_error") and state["polls"] == 5 and "publish" not in state,
          f"a video that never finishes should time out retryably: {result} polls={state['polls']}")
    with FakeApi() as api:
        state = _threads_api(api, statuses=["FINISHED"], create_status=400)
        result = client.publish_video_post("th-token", "Caption", VIDEO)
    check(result.get("success") is False and result.get("status_code") == 400 and "publish" not in state,
          f"a refused container should fail: {result}")
    with FakeApi() as api:
        state = _threads_api(api, statuses=["FINISHED"], publish_status=500)
        result = client.publish_video_post("th-token", "Caption", VIDEO)
    check(result.get("success") is False and result.get("status_code") == 500, f"a failed publish should fail: {result}")
    with FakeApi() as api:
        state = _threads_api(api, statuses=["FINISHED"])
        client.publish_video_post("th-token", "x" * 600, VIDEO)
    check(len(state["create"]["text"]) == 500 and state["create"]["text"].endswith("..."),
          "caption should be cut to Threads' 500 characters")

    # --- the image path, moved onto the shared helper, is unchanged -------
    with FakeApi() as api:
        state = _threads_api(api, statuses=["IN_PROGRESS", "FINISHED"])
        result = client.publish_image_post("th-token", "Words", "https://x.test/i.jpg")
    check(result.get("success") and result["post_id"] == "post-9", f"image publish regressed: {result}")
    check(state["create"] == {"text": "Words", "media_type": "IMAGE", "image_url": "https://x.test/i.jpg",
                              "access_token": "th-token"}, f"image container params changed: {state['create']}")
    check(state["order"] == ["create", "IN_PROGRESS", "FINISHED", "publish"], f"image flow order changed: {state['order']}")
    with FakeApi() as api:
        state = _threads_api(api, statuses=["ERROR"], error_message="Image rejected")
        result = client.publish_image_post("th-token", "Words", "https://x.test/i.jpg")
    check(result == {"success": False, "error": {"message": "Image rejected"}, "guard_error": False},
          f"image ERROR shape changed: {result}")
    with FakeApi() as api:
        state = _threads_api(api, statuses=["IN_PROGRESS"])
        result = client.publish_image_post("th-token", "Words", "https://x.test/i.jpg")
    check(result["error"]["message"] == "Image processing timed out" and state["polls"] == 30,
          f"image timeout changed: {result} polls={state['polls']}")

    print("Threads creates a VIDEO container, waits for FINISHED, then publishes; the image path is unchanged")
    print("VIDEO_THREADS_OK")


# ---------------------------------------------------------------------------
# facebook
# ---------------------------------------------------------------------------


def section_facebook():
    import facebook_client
    client = facebook_client.FacebookClient("id", "secret")

    def run(reply, **kwargs):
        with FakeApi() as api:
            api.on("POST", "/PAGE1/videos", lambda call: reply)
            api.on("POST", "/PAGE1/photos", lambda call: FakeResponse(200, {"post_id": "PAGE1_55"}))
            api.on("POST", "/PAGE1/feed", lambda call: FakeResponse(200, {"id": "PAGE1_66"}))
            result = client.publish_smart_post("page-token", "PAGE1", "Caption", **kwargs)
            return api, result

    api, result = run(FakeResponse(200, {"id": "777"}), video_url=VIDEO)
    call = api.calls_to("POST", "/PAGE1/videos")
    check(result.get("success") and result["post_id"] == "777"
          and result["permalink"] == "https://www.facebook.com/PAGE1/videos/777", f"video publish failed: {result}")
    check(len(call) == 1 and call[0]["data"] == {"file_url": VIDEO, "description": "Caption",
                                                 "access_token": "page-token"},
          f"the /videos request is not the documented shape: {call and call[0]['data']}")

    # video beats image: never both, and the photo edge is not touched
    api, result = run(FakeResponse(200, {"id": "777"}), video_url=VIDEO, image_url="https://x.test/i.jpg")
    check(result.get("success") and not api.calls_to("POST", "/photos") and len(api.calls) == 1,
          "a video and an image together must post only the video")

    # failures
    api, result = run(FakeResponse(400, {"error": {"message": "Video URL not reachable"}}), video_url=VIDEO)
    check(result.get("success") is False and result["status_code"] == 400
          and "not reachable" in json.dumps(result["error"]), f"a refused video should fail with the reason: {result}")
    api, result = run(FakeResponse(200, {}), video_url=VIDEO)
    check(result.get("success") is False and "no video id" in result["error"]["message"],
          f"a 200 without an id is not a success: {result}")
    api, result = run(FakeResponse(200, {"id": "9"}), video_url=VIDEO)
    check(result.get("success"), "control: an accepted video is a success")

    # unchanged: image-only and text-only
    api, result = run(None, image_url="https://x.test/i.jpg")
    check(result.get("success") and api.calls_to("POST", "/PAGE1/photos") and not api.calls_to("POST", "/PAGE1/videos"),
          f"image-only post regressed: {result}")
    api, result = run(None)
    check(result.get("success") and api.calls_to("POST", "/PAGE1/feed") and not api.calls_to("POST", "/PAGE1/videos"),
          f"text-only post regressed: {result}")

    print("Facebook posts a video to the Page videos edge by file_url; video beats image; failures are reported")
    print("VIDEO_FACEBOOK_OK")


# ---------------------------------------------------------------------------
# instagram
# ---------------------------------------------------------------------------


class _IgClient:
    def __init__(self):
        self.calls = []

    def _ok(self, kind, **kw):
        self.calls.append({"kind": kind, **kw})
        return {"success": True, "permalink": f"https://instagram.test/{kind}"}

    def publish_image_post(self, token, caption, image_url, user_tags=None):
        return self._ok("image", url=image_url)

    def publish_reel_post(self, token, caption, video_url):
        return self._ok("reel", url=video_url)

    def publish_carousel_post(self, token, caption, items):
        return self._ok("carousel", items=items)

    def publish_story_post(self, token, image_url=None, video_url=None):
        return self._ok("story", image=image_url, video=video_url)


def section_instagram():
    from _accounts_gate import isolated_app, connect
    directory, database, web, publisher, client = isolated_app()
    P = database.DB_PATH
    web._maybe_attach_link_image = lambda *a, **k: None
    account = connect(database, "instagram", "ig-1", "brandco")
    for hhmm in ("09:00", "18:00"):  # slots for the queue guard to place posts in
        database.add_time_slot(-1, hhmm, True, None, db_path=P)
    ig = _IgClient()
    web.get_instagram_client = lambda: ig
    publisher.get_instagram_client = lambda: ig
    # If the media guard went looking for a stock photo, that is a failure here.
    stock_lookups = []
    web.get_image_for_post = lambda *a, **k: stock_lookups.append(a) or None

    def row(**fields):
        pid = database.add_standalone_post("manual", "g", "instagram", "Caption", db_path=P,
                                           account_id=account, **{k: v for k, v in fields.items()
                                                                   if k in ("image_url", "video_url")})
        if "ig_post_type" in fields:
            database.set_standalone_post_media(pid, fields["ig_post_type"], fields.get("media_items"), db_path=P)
        return pid

    def publish(pid, **kwargs):
        ig.calls.clear()
        post = database.get_standalone_post(pid, db_path=P)
        return publisher.publish("instagram", account, content="Caption", image_url=post["image_url"],
                                 video_url=post["video_url"], standalone_post_id=pid, **kwargs)

    # default format + card video -> a Reel; no image, no stock lookup
    pid = row(video_url=VIDEO)
    result = publish(pid)
    check(result["success"] and ig.calls == [{"kind": "reel", "url": VIDEO}], f"default format + video should be a Reel: {ig.calls} {result}")
    check(not stock_lookups, "a video post must not go looking for a stock image")

    # the video wins over an image on the same card
    pid = row(video_url=VIDEO, image_url="https://x.test/i.jpg")
    publish(pid)
    check(ig.calls == [{"kind": "reel", "url": VIDEO}], f"video + image should post only the video: {ig.calls}")

    # control: no video -> the old feed image route
    pid = row(image_url="https://x.test/i.jpg")
    publish(pid)
    check(ig.calls == [{"kind": "image", "url": "https://x.test/i.jpg"}], f"image-only feed regressed: {ig.calls}")

    # explicit formats are left alone
    items = [{"url": "https://x.test/1.jpg", "kind": "image"}, {"url": "https://x.test/2.mp4", "kind": "video"}]
    pid = row(video_url=VIDEO, ig_post_type="carousel", media_items=items)
    publish(pid)
    check(ig.calls == [{"kind": "carousel", "items": items}], f"an explicit carousel must keep its own media: {ig.calls}")
    pid = row(video_url=VIDEO, ig_post_type="reel", media_items=[])
    publish(pid)
    check(ig.calls == [{"kind": "reel", "url": VIDEO}], f"an empty reel builder should use the card's video: {ig.calls}")
    pid = row(video_url=VIDEO, ig_post_type="reel",
              media_items=[{"url": "https://x.test/mine.mp4", "kind": "video"}])
    publish(pid)
    check(ig.calls == [{"kind": "reel", "url": "https://x.test/mine.mp4"}], f"a reel's own video must win: {ig.calls}")
    pid = row(video_url=VIDEO, ig_post_type="story", media_items=[])
    publish(pid)
    check(ig.calls == [{"kind": "story", "image": None, "video": VIDEO}], f"an empty story builder should use the card's video: {ig.calls}")

    # the media guard: a video is enough, an empty post still is not
    resolved, error = web._ensure_instagram_media("c", None, None, [], video_url=VIDEO)
    check(error is None and resolved == [{"url": VIDEO, "kind": "video"}], f"guard rejected a video-only post: {error}")
    resolved, error = web._ensure_instagram_media("c", None, None, [])
    check(error and "requires an image" in error, f"control: a post with no media must still be refused: {error}")

    # queueing: the queue guard accepts a video-only row, and refuses the control
    with_video = database.get_standalone_post(row(video_url=VIDEO), db_path=P)
    without = database.get_standalone_post(row(), db_path=P)
    queued, skipped, _ = web._queue_unscheduled_rows("instagram", [with_video])
    check((queued, skipped) == (1, 0), f"a video-only Instagram row should queue: queued={queued} skipped={skipped}")
    queued, skipped, _ = web._queue_unscheduled_rows("instagram", [without])
    check((queued, skipped) == (0, 1), f"control: a media-less Instagram row must not queue: queued={queued} skipped={skipped}")

    print("an Instagram card with a video publishes as a Reel; explicit formats are kept; the guard accepts a video")
    print("VIDEO_INSTAGRAM_OK")


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

PUBLIC_URL = "http://93.184.216.34/clip.mp4"  # a public IP literal: passes the SSRF guard with no DNS
PUBLIC_URL_2 = "http://93.184.216.34/other.mp4"


def _rig():
    from check_bulk_platform import Rig
    return Rig()


def section_routes():
    import io
    rig = _rig()
    db, P, web, client = rig.database, rig.P, rig.web, rig.client
    sizes = {"value": 5 * MB}
    web.video_media.head_size = lambda url: sizes["value"]  # no network in a gate

    ids = rig.card("Route words", ["linkedin", "threads", "twitter"], image_url=None)
    card_ids = ",".join(str(i) for i in ids.values())
    other = rig.card("Somebody else's post", ["linkedin"], image_url=None)

    def videos(row_ids):
        return [db.get_standalone_post(i, db_path=P)["video_url"] for i in row_ids]

    def post_video(post_id, **form):
        response = client.post(f"/compose/post/{post_id}/video", data=form)
        return response.status_code, (response.get_json(silent=True) or {})

    # attach: every row of the card, and it stays one card
    status, res = post_video(ids["linkedin"], video_url=PUBLIC_URL, post_ids=card_ids)
    check(status == 200 and res.get("success"), f"attach failed: {status} {trim(res)}")
    check(sorted(res["updated_ids"]) == sorted(ids.values()), f"not every row of the card was updated: {res['updated_ids']}")
    check(videos(ids.values()) == [PUBLIC_URL] * 3, "the card's rows do not all carry the video")
    cards = [g for g in rig.cards() if g["head"]["content"] == "Route words"]
    check(len(cards) == 1 and len(cards[0]["platforms"]) == 3, f"attaching a video split the card: {len(cards)} cards")
    check(res["size_bytes"] == 5 * MB and res["warnings"] == [], f"a small video should carry no warnings: {trim(res)}")

    # replace
    status, res = post_video(ids["threads"], video_url=PUBLIC_URL_2, post_ids=card_ids)
    check(status == 200 and videos(ids.values()) == [PUBLIC_URL_2] * 3, "replace did not reach every row")
    check(len([g for g in rig.cards() if g["head"]["content"] == "Route words"]) == 1, "replace split the card")

    # per-platform warnings come with the attach (LinkedIn 500 MB, Instagram 300 MB)
    sizes["value"] = 600 * MB
    status, res = post_video(ids["linkedin"], video_url=PUBLIC_URL, post_ids=card_ids)
    hard = sorted(w["platform"] for w in res["warnings"] if w["level"] == "error")
    check(status == 200 and hard == ["instagram", "linkedin"], f"size warnings wrong: {hard} {trim(res)}")
    sizes["value"] = 5 * MB

    # a stale id from another card is not touched
    status, res = post_video(ids["linkedin"], video_url=PUBLIC_URL_2,
                             post_ids=f"{card_ids},{other['linkedin']}")
    check(other["linkedin"] not in res["updated_ids"] and videos([other["linkedin"]]) == [None],
          "a row that is not on the card was rewritten")

    # unsafe or malformed URLs are refused and change nothing
    before = videos(ids.values())
    for bad in ("ftp://93.184.216.34/x.mp4", "javascript:alert(1)", "file:///etc/passwd",
                "http://127.0.0.1/x.mp4", "http://169.254.169.254/latest/meta-data",
                "http://10.0.0.5/x.mp4", "http://user:pw@93.184.216.34/x.mp4", "not a url"):
        status, res = post_video(ids["linkedin"], video_url=bad, post_ids=card_ids)
        check(status == 400 and res.get("error") and not res.get("success"), f"{bad!r} should be refused: {status} {trim(res)}")
        check(videos(ids.values()) == before, f"{bad!r} was refused but still changed the card")

    # clear
    status, res = post_video(ids["linkedin"], video_url="", post_ids=card_ids)
    check(status == 200 and res["video_url"] is None and videos(ids.values()) == [None] * 3, f"clear failed: {trim(res)}")
    check(post_video(999999, video_url=PUBLIC_URL)[0] == 404, "an unknown post should be a 404")

    # --- upload ---------------------------------------------------------
    sample = make_sample_mp4(rig.directory, seconds=2)
    with open(sample, "rb") as fh:
        clip = fh.read()
    uploads = []

    def fake_upload(data, **kwargs):
        uploads.append({"bytes": len(data), **kwargs})
        return {"secure_url": "https://res.cloudinary.test/insights/abc123.mp4",
                "public_id": "insights/abc123", "bytes": len(data)}

    web.cloudinary.uploader.upload = fake_upload

    def upload(name, data, post_id=None, **form):
        url = f"/compose/post/{post_id}/video" if post_id else "/compose/upload-video"
        field = {"video": (io.BytesIO(data), name), **form}
        response = client.post(url, data=field, content_type="multipart/form-data")
        return response.status_code, (response.get_json(silent=True) or {})

    web.CLOUDINARY_CONFIGURED = False
    status, res = upload("clip.mp4", clip, ids["linkedin"], post_ids=card_ids)
    check(status == 400 and "Cloudinary" in res["error"] and videos(ids.values()) == [None] * 3 and not uploads,
          f"without Cloudinary the upload should be refused cleanly: {status} {trim(res)}")

    web.CLOUDINARY_CONFIGURED = True
    for name, data in (("clip.avi", clip), ("notes.txt", b"hello"), ("empty.mp4", b"")):
        status, res = upload(name, data, ids["linkedin"], post_ids=card_ids)
        check(status == 400 and res.get("error") and videos(ids.values()) == [None] * 3 and not uploads,
              f"{name} should be refused before any upload: {status} {trim(res)}")

    status, res = upload("clip.mp4", clip, ids["linkedin"], post_ids=card_ids)
    check(status == 200 and res.get("success"), f"upload failed: {status} {trim(res)}")
    check(len(uploads) == 1 and uploads[0]["resource_type"] == "video" and uploads[0]["bytes"] == len(clip),
          f"Cloudinary should receive the video as resource_type=video: {uploads}")
    check(videos(ids.values()) == ["https://res.cloudinary.test/insights/abc123.mp4"] * 3,
          "the uploaded video was not attached to every row of the card")
    check(res["duration"] is not None and 1.5 < res["duration"] < 2.6,
          f"the upload should report the probed duration (~2s): {res['duration']}")
    short = sorted(w["platform"] for w in res["warnings"] if "at least 3 seconds" in w["message"])
    check(short == ["instagram", "linkedin"],
          f"a 2s clip should be flagged too short for LinkedIn and Instagram: {trim(res['warnings'])}")
    check(len([g for g in rig.cards() if g["head"]["content"] == "Route words"]) == 1, "upload split the card")

    def failing_upload(data, **kwargs):
        raise RuntimeError("cloudinary is down")

    web.cloudinary.uploader.upload = failing_upload
    before = videos(ids.values())
    status, res = upload("clip.mp4", clip, ids["linkedin"], post_ids=card_ids)
    check(status == 400 and "cloudinary is down" in res["error"] and videos(ids.values()) == before,
          f"a failed upload must say why and change nothing: {status} {trim(res)}")

    # the Instagram builder's own upload route still works and reports the same
    web.cloudinary.uploader.upload = fake_upload
    status, res = upload("clip.mp4", clip)
    check(status == 200 and res.get("success") and res["video_url"].endswith("abc123.mp4") and "warnings" in res,
          f"/compose/upload-video regressed: {status} {trim(res)}")

    # the library keeps videos apart from images
    library = client.get("/compose/list-images").get_json()
    check(library["success"] and all(i["media_type"] == "image" for i in library["images"])
          and not any(i["url"].endswith("abc123.mp4") for i in library["images"]),
          "the image library must not list videos")
    library = client.get("/compose/list-images?type=video").get_json()
    check([i["url"] for i in library["images"]] == ["https://res.cloudinary.test/insights/abc123.mp4"]
          and library["images"][0]["media_type"] == "video", f"the video library is wrong: {trim(library)}")

    print("the video route attaches, replaces and clears across a whole card, refuses unsafe URLs, and reports platform limits")
    print("VIDEO_ROUTES_OK")


# ---------------------------------------------------------------------------
# fanout
# ---------------------------------------------------------------------------


def section_fanout():
    from datetime import datetime, timedelta
    rig = _rig()
    db, P, web, client = rig.database, rig.P, rig.web, rig.client
    platforms = ("linkedin", "threads", "twitter", "facebook", "instagram")
    clients = {p: getattr(rig.publisher, f"get_{p}_client")() for p in platforms}

    def card(content, video):
        ids = rig.card(content, list(platforms), image_url="https://x.test/i.jpg")
        for row_id in ids.values():
            db.update_standalone_post_video(row_id, video, db_path=P)
        return ids

    def last(platform):
        return clients[platform].calls[-1]

    # post to every target on the card at once
    ids = card("Fan out with video", VIDEO)
    res = client.post(f"/compose/post/{ids['linkedin']}/publish",
                      data={"post_ids": ",".join(str(i) for i in ids.values())}).get_json()
    check(res["success"] and res["published"] == 5, f"post-to-all failed: {trim(res)}")
    for p in platforms:
        check(last(p)["kind"] == "video" and last(p)["video"] == VIDEO,
              f"{p} did not receive the video on post-to-all: {last(p)}")
        check(all(c["kind"] != "image" for c in clients[p].calls), f"{p} was sent the image as well as the video")

    # post now, one target at a time (the per-platform endpoints share one route)
    ids = card("Post now with video", VIDEO)
    for p in platforms:
        before = len(clients[p].calls)
        response = client.post(f"/compose/post/{ids[p]}/{p}", data={"post_ids": str(ids[p])})
        check(response.status_code == 200 and response.get_json()["success"],
              f"post-now to {p} failed: {response.status_code} {trim(response.get_json())}")
        check(len(clients[p].calls) == before + 1 and last(p)["kind"] == "video",
              f"post-now to {p} did not send the video: {last(p)}")

    # one failing target does not stop the rest
    ids = card("One fails", VIDEO)
    clients["twitter"].fail = True
    clients["twitter"].error = "X refused the video"
    res = client.post(f"/compose/post/{ids['linkedin']}/publish",
                      data={"post_ids": ",".join(str(i) for i in ids.values())}).get_json()
    by = {r["platform"]: r for r in res["results"]}
    check(res["partial"] and res["published"] == 4 and not by["twitter"]["success"]
          and "X refused the video" in by["twitter"]["error"], f"partial failure not reported: {trim(res)}")
    check(all(by[p]["success"] for p in platforms if p != "twitter"), "a failing target stopped the others")
    check(not db.get_standalone_post(ids["twitter"], db_path=P)["used"]
          and db.get_standalone_post(ids["linkedin"], db_path=P)["used"], "used flags do not reflect what landed")
    clients["twitter"].fail = False

    # the queue: every way of reading a scheduled entry carries the video, and
    # the entry the background worker reads (get_pending_scheduled_posts) is
    # the one that gets published -- a query that dropped the column would
    # silently post the queued card without its video.
    ids = card("Queued with video", VIDEO)
    past = (datetime.now() - timedelta(minutes=5)).isoformat(timespec="seconds")
    queued = {}
    for p in platforms:
        queued[p] = db.add_scheduled_post(social_post_id=None, article_id=None, standalone_post_id=ids[p],
                                          post_type="standalone", platform=p, scheduled_for=past,
                                          status="pending", db_path=P)
    due = {row["id"]: row for row in db.get_pending_scheduled_posts(db_path=P)}
    listed = {row["id"]: row for row in db.list_scheduled_posts(db_path=P)}
    for p, sid in queued.items():
        check(sid in due and sid in listed, f"the queued {p} entry is missing from the worker's or the schedule's query")
        for source, row in (("get_scheduled_post", db.get_scheduled_post(sid, db_path=P)),
                            ("get_pending_scheduled_posts", due[sid]),
                            ("list_scheduled_posts", listed[sid])):
            content, image, video, error = web._scheduled_post_content(row)
            check(error is None and video == VIDEO,
                  f"{source} does not hand the {p} entry its video: {video}")
        before = len(clients[p].calls)
        result = web._publish_scheduled_entry(due[sid])
        check(result["success"], f"queued {p} publish failed: {result}")
        check(len(clients[p].calls) == before + 1 and last(p)["kind"] == "video" and last(p)["video"] == VIDEO,
              f"the scheduler did not send {p} the video: {last(p)}")
        check(db.get_scheduled_post(sid, db_path=P)["status"] == "posted", f"queued {p} was not marked posted")

    # control: a card without a video still sends the image, so "video" above is not a default
    ids = card("No video here", None)
    client.post(f"/compose/post/{ids['linkedin']}/publish",
                data={"post_ids": ",".join(str(i) for i in ids.values())})
    check(last("linkedin")["kind"] == "image" and last("threads")["kind"] == "image"
          and last("linkedin").get("video") is None, "a card without a video must post as it always did")

    print("post-to-all, post-now and the schedule queue each deliver the video to every platform; a failing target does not stop the rest")
    print("VIDEO_FANOUT_OK")


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def section_copy():
    rig = _rig()
    db, P, web, client = rig.database, rig.P, rig.web, rig.client

    def card_of(content):
        return [g for g in rig.cards() if g["head"]["content"] == content]

    ids = rig.card("Copy with video", ["linkedin"], image_url=None)
    db.update_standalone_post_video(ids["linkedin"], VIDEO, db_path=P)
    plain = rig.card("Copy without video", ["linkedin"], image_url=None)

    # tick another platform
    res = client.post(f"/compose/post/{ids['linkedin']}/platform",
                      data={"platform": "threads", "action": "add", "post_ids": str(ids["linkedin"])})
    check(res.status_code == 200 and res.get_json().get("success"), f"tick failed: {res.status_code} {trim(res.get_json())}")
    threads = [r for r in rig.rows_for("Copy with video") if r["platform"] == "threads"]
    check(len(threads) == 1 and threads[0]["video_url"] == VIDEO, f"the ticked row did not get the video: {threads}")
    check(len(card_of("Copy with video")) == 1 and len(card_of("Copy with video")[0]["platforms"]) == 2,
          "ticking a platform split the card because the video was not copied")

    # a second account of the same platform
    res = client.post(f"/compose/post/{ids['linkedin']}/platform",
                      data={"platform": "linkedin", "action": "add", "account_id": str(rig.studio),
                            "post_ids": f"{ids['linkedin']},{threads[0]['id']}"})
    check(res.status_code == 200, f"tick of the second account failed: {res.status_code}")
    check(len(card_of("Copy with video")) == 1 and len(card_of("Copy with video")[0]["platforms"]) == 3,
          "a second account's row did not stay on the card")

    # bulk Add Platform
    status, res = rig.post({"post_ids": [ids["linkedin"], plain["linkedin"]], "targets": ["twitter"], "queue": False})
    check(status == 200 and res.get("success") and res["added"] == 2, f"bulk add failed: {status} {trim(res)}")
    twitter = [r for r in rig.rows_for("Copy with video") if r["platform"] == "twitter"]
    check(len(twitter) == 1 and twitter[0]["video_url"] == VIDEO, f"bulk add did not copy the video: {twitter}")
    check(len(card_of("Copy with video")) == 1 and len(card_of("Copy with video")[0]["platforms"]) == 4,
          "bulk Add Platform split the video card")
    # control: the card without a video is copied without one, and is still its own card
    plain_twitter = [r for r in rig.rows_for("Copy without video") if r["platform"] == "twitter"]
    check(len(plain_twitter) == 1 and plain_twitter[0]["video_url"] is None
          and len(card_of("Copy without video")) == 1, "a card without a video changed shape")

    # re-rendering the card (which the ticks do) keeps the video
    html = client.post(f"/compose/post/{ids['linkedin']}/card",
                       data={"post_ids": ",".join(str(r["id"]) for r in rig.rows_for("Copy with video"))}).get_json()["html"]
    check(VIDEO in html, "the re-rendered card lost the video")

    print("ticking a platform, a second account and bulk Add Platform all copy the video and keep one card")
    print("VIDEO_COPY_OK")


# ---------------------------------------------------------------------------
# compat
# ---------------------------------------------------------------------------


def section_compat():
    import video_media as vm
    directory = tempfile.mkdtemp(prefix="video_compat_")
    sample = make_sample_mp4(directory, seconds=2, size="320x240")

    info = vm.probe_file(sample)
    check(info.get("width") == 320 and info.get("height") == 240 and info.get("codec") == "h264"
          and 1.8 < info.get("duration", 0) < 2.4, f"ffprobe read the sample wrongly: {info}")
    size = os.path.getsize(sample)
    check(size > 1000 and vm.probe_file(os.path.join(directory, "missing.mp4")) == {}, "probe of a missing file should be {}")

    # what the real file means for each platform: derived from the probe, not typed in
    issues = vm.check_compat(size_bytes=size, duration_seconds=info["duration"])
    flagged = sorted((i["platform"], i["level"]) for i in issues)
    check(flagged == [("instagram", "error"), ("linkedin", "error")],
          f"a real 2-second clip should be too short for exactly LinkedIn and Instagram: {flagged}")
    check(all("at least 3 seconds" in i["message"] for i in issues), f"the reason should say why: {issues}")

    def only(platform, **kw):
        return vm.check_compat(platforms=[platform], **kw)

    # size boundaries: at the limit is fine, one byte over is refused
    for platform, limit in (("linkedin", 500 * MB), ("instagram", 300 * MB), ("facebook", 1024 * MB),
                            ("threads", 1024 * MB), ("twitter", 1024 * MB)):
        check(only(platform, size_bytes=limit) == [], f"{platform}: exactly {limit // MB} MB should be accepted")
        over = only(platform, size_bytes=limit + 1)
        check(len(over) == 1 and over[0]["level"] == "error" and over[0]["platform"] == platform,
              f"{platform}: one byte over {limit // MB} MB should be an error: {over}")
        check(vm.preflight(platform, limit) is None and vm.preflight(platform, limit + 1),
              f"{platform}: preflight disagrees with check_compat at the boundary")

    # duration boundaries
    for platform, low, high in (("linkedin", 3, 1800), ("instagram", 3, 900), ("twitter", None, 140),
                                ("threads", None, 300), ("facebook", None, 1200)):
        check(only(platform, duration_seconds=high) == [], f"{platform}: {high}s should be accepted")
        over = only(platform, duration_seconds=high + 1)
        check(len(over) == 1 and over[0]["level"] == "warn", f"{platform}: {high + 1}s should warn: {over}")
        if low:
            check(only(platform, duration_seconds=low) == [], f"{platform}: {low}s should be accepted")
            under = only(platform, duration_seconds=low - 0.01)
            check(len(under) == 1 and under[0]["level"] == "error", f"{platform}: under {low}s should be an error: {under}")
    check(vm.check_compat() == [], "with nothing known there is nothing to say")
    check(vm.check_compat(size_bytes=None, duration_seconds=None) == [], "unknown size/duration must not be judged")
    check(vm.preflight("linkedin", None) is None and vm.preflight("nowhere", 10 ** 12) is None,
          "preflight must be silent for unknown values and platforms")
    message = only("linkedin", size_bytes=501 * MB)[0]["message"]
    check("LinkedIn" in message and "500 MB" in message and "501 MB" in message, f"the message should be readable: {message}")

    print("a real MP4 is probed, and each platform's size and length limits are enforced at the boundary")
    print("VIDEO_COMPAT_OK")


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def section_download():
    import video_media as vm
    clip = os.urandom(3 * MB + 17)
    routes = {
        "/ok.mp4": {"body": clip, "type": "video/mp4"},
        "/octet": {"body": clip, "type": "application/octet-stream"},
        "/chunked": {"body": clip, "type": "video/mp4", "chunked": True},
        "/page": {"body": b"<html>sign in</html>", "type": "text/html"},
        "/json": {"body": b"{}", "type": "application/json"},
        "/empty": {"body": b"", "type": "video/mp4"},
        "/redirect": {"status": 302, "headers": {"Location": "/ok.mp4"}},
        "/loop": {"status": 302, "headers": {"Location": "/loop"}},
        "/to-file": {"status": 302, "headers": {"Location": "file:///etc/passwd"}},
        "/err503": {"status": 503, "body": b"", "type": "text/plain"},
        "/err429": {"status": 429, "body": b"", "type": "text/plain"},
        "/err403": {"status": 403, "body": b"", "type": "text/plain"},
    }
    before = temp_videos()
    with VideoServer(routes) as server:
        routes["/redir-local"] = {"status": 302,
                                  "headers": {"Location": f"http://localhost:{server.port}/ok.mp4"}}

        # the file comes down byte for byte, exists inside the block, and is gone after
        for path in ("/ok.mp4", "/octet", "/chunked", "/redirect"):
            with vm.downloaded(server.url(path)) as (local, size):
                check(size == len(clip) and os.path.getsize(local) == len(clip), f"{path}: wrong size {size}")
                with open(local, "rb") as fh:
                    check(sha(fh.read()) == sha(clip), f"{path}: the downloaded bytes differ from the served ones")
                inside = local
            check(not os.path.exists(inside), f"{path}: the temp file outlived the block")

        def refused(path, needle, permanent, **kwargs):
            try:
                with vm.downloaded(server.url(path), **kwargs):
                    fail(f"{path}: should have been refused")
            except vm.VideoError as exc:
                check(needle in str(exc), f"{path}: the reason should mention {needle!r}: {exc}")
                check(exc.permanent is permanent, f"{path}: permanent should be {permanent}: {exc.permanent}")

        # caps, declared and undeclared
        refused("/ok.mp4", "limit", True, max_bytes=2 * MB)          # Content-Length says too big
        refused("/chunked", "limit", True, max_bytes=2 * MB)         # no length: cut off mid-stream
        with vm.downloaded(server.url("/ok.mp4"), max_bytes=len(clip)):
            pass                                                        # exactly at the cap is fine
        # not a video / broken
        refused("/page", "not a video", True)
        refused("/json", "not a video", True)
        refused("/empty", "empty", True)
        refused("/missing", "404", True)
        refused("/err403", "403", True)
        refused("/err503", "503", False)                              # worth retrying
        refused("/err429", "429", False)
        refused("/loop", "redirects", True)
        refused("/to-file", "http", True)                             # a redirect cannot leave http(s)

        # unreachable host is transient
        try:
            with vm.downloaded("http://127.0.0.1:9/never.mp4", timeout=2):
                fail("a closed port should not download")
        except vm.VideoError as exc:
            check(exc.permanent is False, "an unreachable host is worth retrying")

        # the guard runs before any request, and again on every redirect hop
        def guard(url):
            if "127.0.0.1" in url:
                raise ValueError("blocked by guard")

        vm.set_url_guard(guard)
        hits = len(server.hits)
        refused("/ok.mp4", "blocked by guard", True)
        check(len(server.hits) == hits, "a guarded URL must not reach the server at all")
        vm.set_url_guard(lambda url: (_ for _ in ()).throw(ValueError("no localhost"))
                         if "localhost" in url else None)
        hits = len(server.hits)
        refused("/redir-local", "no localhost", True)
        check(server.hits[hits:] == [("GET", "/redir-local")],
              f"a redirect into a blocked host must not be followed: {server.hits[hits:]}")
        vm.set_url_guard(None)

        # HEAD size
        check(vm.head_size(server.url("/ok.mp4")) == len(clip), "head_size should read Content-Length")
        check(vm.head_size(server.url("/missing")) is None or True, "head_size on 404 must not raise")
        check(vm.head_size("http://127.0.0.1:9/x.mp4", timeout=2) is None, "head_size on an unreachable host must be None")
        vm.set_url_guard(guard)
        check(vm.head_size(server.url("/ok.mp4")) is None, "head_size must respect the guard and never raise")
        vm.set_url_guard(None)

    check(temp_videos() <= before, f"temp video files were left behind: {temp_videos() - before}")

    # the app registers its own SSRF guard: internal addresses are refused, public ones pass
    import insights_web  # noqa: F401
    for bad in ("http://127.0.0.1:5001/x.mp4", "http://localhost/x.mp4", "http://169.254.169.254/x",
                "http://10.1.2.3/x.mp4", "ftp://93.184.216.34/x.mp4", "http://[::1]/x.mp4"):
        try:
            vm.check_url(bad)
            fail(f"the app's guard let {bad} through to the downloader")
        except vm.VideoError:
            pass
    check(vm.check_url(" http://93.184.216.34/x.mp4 ") == "http://93.184.216.34/x.mp4", "a public URL should pass, trimmed")

    print("the downloader streams to disk within its cap, refuses unsafe or non-video URLs at every hop, and cleans up")
    print("VIDEO_DOWNLOAD_OK")


SECTIONS = {
    "schema": section_schema,
    "router": section_router,
    "linkedin": section_linkedin,
    "twitter": section_twitter,
    "threads": section_threads,
    "facebook": section_facebook,
    "instagram": section_instagram,
    "routes": section_routes,
    "fanout": section_fanout,
    "copy": section_copy,
    "compat": section_compat,
    "download": section_download,
}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        from check_video_ui import section_ui
        SECTIONS["ui"] = section_ui
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_video.py {'|'.join(list(SECTIONS) + ['ui'])}")
        sys.exit(2)
    SECTIONS[name]()
