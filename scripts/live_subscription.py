"""Live checks for the subscription providers and the MCP server, on this computer.

    python scripts/live_subscription.py claude       LIVE_CLAUDE_OK
    python scripts/live_subscription.py codex        LIVE_CODEX_OK
    python scripts/live_subscription.py mcp-claude   LIVE_MCP_CLAUDE_OK
    python scripts/live_subscription.py mcp-codex    LIVE_MCP_CODEX_OK

These call the real ``claude`` and ``codex`` CLIs, so they need them installed
and signed in to a Claude or ChatGPT plan, and each run uses a little of that
plan's allowance (a few short prompts). They are deliberately not in
run_all_gates.py: CI has neither CLI.

The proof that a call ran on the plan and not on an API key: before anything
runs, this process sets ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` /
``CODEX_API_KEY`` to values that cannot authenticate. If any of them reached a
CLI, it would use the key (``claude -p`` prefers an API key to the signed-in
plan), fail to authenticate, and the check would fail.

Everything runs in a throwaway directory with its own database, so the app's
real insights.db is never read or written.
"""

import base64
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
import zlib
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

EM, EN = chr(0x2014), chr(0x2013)
BOGUS = {
    "ANTHROPIC_API_KEY": "sk-ant-api03-not-a-real-key-" + "x" * 40,
    "OPENAI_API_KEY": "sk-proj-not-a-real-key-" + "x" * 40,
    "CODEX_API_KEY": "sk-proj-not-a-real-key-" + "x" * 40,
}


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def check(condition, message):
    if not condition:
        fail(message)


def sandbox():
    """A throwaway working directory and database, with the bogus keys set."""
    directory = tempfile.mkdtemp(prefix="live_sub_")
    os.chdir(directory)
    os.environ.update(BOGUS)
    for name in ("CLAUDE_CLI_PATH", "CODEX_CLI_PATH", "LLM_PROVIDER"):
        os.environ.pop(name, None)
    import database
    database.DB_PATH = os.path.join(directory, "insights.db")
    database.init_db(database.DB_PATH)
    return directory, database


def red_png():
    width = height = 32
    raw = b"".join(b"\x00" + bytes((220, 20, 20)) * width for _ in range(height))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


def generation(provider, quick_model):
    directory, database = sandbox()
    import insights
    import subscription_llm as sub

    info = sub.status(provider, refresh=True)
    check(info["available"] and info["signed_in"] and info["subscription"],
          f"{provider} is not signed in to a plan here: {info}")

    nonce = uuid.uuid4().hex[:10].upper()
    client, model, resolved = insights._get_llm_client(provider=provider, model=quick_model)
    check(resolved == provider, f"routed to {resolved}")
    started = time.monotonic()
    reply = client.chat.completions.create(model=model, max_tokens=50, temperature=0.2, messages=[
        {"role": "system", "content": "Reply with exactly the code word the user gives you and nothing else."},
        {"role": "user", "content": f"The code word is {nonce}."}])
    text = reply.choices[0].message.content
    check(nonce in text, f"the reply did not carry the code word: {text!r}")
    print(f"{provider} {model}: code word back in {time.monotonic() - started:.1f}s, "
          f"{reply.usage.input_tokens} in / {reply.usage.output_tokens} out")

    reply = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [
        {"type": "text", "text": "What single colour fills this image? Answer with one lowercase word."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + red_png()}}]}])
    colour = reply.choices[0].message.content.strip().lower()
    check("red" in colour, f"the image did not arrive: {colour!r}")
    print(f"{provider}: saw the image ({colour})")

    started = time.monotonic()
    posts = insights.generate_posts_from_prompt(
        "Why passkeys are safer than passwords", platforms=["twitter", "linkedin"],
        posts_per_platform=1, provider=provider)
    took = time.monotonic() - started
    for platform in ("twitter", "linkedin"):
        value = posts.get(platform)
        value = value[0] if isinstance(value, list) and value else value
        check(isinstance(value, str) and len(value.strip()) > 20, f"no {platform} post: {posts}")
        check(EM not in value and EN not in value, f"the {platform} post has a long dash: {value!r}")
    _, default_model, _ = insights._get_llm_client(provider=provider)
    print(f"{provider} {default_model}: wrote a tweet and a LinkedIn post in {took:.1f}s")
    print("  tweet: " + (posts["twitter"][0] if isinstance(posts["twitter"], list) else posts["twitter"])[:200])

    rows = sqlite3.connect(database.DB_PATH).execute(
        "SELECT COUNT(*), SUM(cost_usd) FROM usage_events WHERE provider = ?", (provider,)).fetchone()
    check(rows[0] >= 1 and rows[1] == 0, f"the generator's call is not metered at $0: {rows}")
    print(f"{provider}: the generator's call metered at $0 while {', '.join(BOGUS)} held unusable keys")


def section_claude():
    generation("claude-code", "claude-haiku-4-5")
    print("LIVE_CLAUDE_OK")


def section_codex():
    generation("codex", "gpt-5.6-sol")
    print("LIVE_CODEX_OK")


def seeded_db():
    directory, database = sandbox()
    conn = sqlite3.connect(database.DB_PATH)
    conn.execute("INSERT INTO feeds (url, title) VALUES ('https://feed.test/rss', 'Security Podcast')")
    conn.execute(
        "INSERT INTO episodes (feed_id, url, title, transcript, summary, status) VALUES "
        "(1, 'https://ep.test/42', 'Passkeys in the wild', ?, ?, 'complete')",
        ("Host: today we look at passkeys. Guest: passkeys are bound to the site, so a phishing "
         "page cannot replay them. Host: and there is no shared secret on the server to steal.",
         "Why passkeys resist phishing and server breaches."))
    conn.commit()
    conn.close()
    return directory, database


MCP_TASK = (
    "Use the insights MCP tools, and only them. First call get_writing_guide for twitter. "
    "Then call list_sources and read the episode with get_source. Then call save_drafts once "
    "with exactly one twitter post about that episode, with source_label set to {label}. "
    "Follow the guide (no em or en dashes). When the draft is saved, reply DONE."
)


def mcp_rows(database, label):
    return sqlite3.connect(database.DB_PATH).execute(
        "SELECT platform, content, used, source_type FROM standalone_posts WHERE source_content = ?",
        (label,)).fetchall()


def check_mcp_rows(database, label, who):
    rows = mcp_rows(database, label)
    check(len(rows) == 1, f"{who} saved {len(rows)} drafts with the label: {rows}")
    platform, content, used, source_type = rows[0]
    check(platform == "twitter" and used == 0 and source_type == "mcp", f"the draft row: {rows[0]}")
    check(EM not in content and EN not in content, f"the draft has a long dash: {content!r}")
    queued = sqlite3.connect(database.DB_PATH).execute("SELECT COUNT(*) FROM scheduled_posts").fetchone()[0]
    check(queued == 0, "something was queued")
    print(f"{who}: saved an unused twitter draft through MCP: {content[:160]!r}")


def section_mcp_claude():
    directory, database = seeded_db()
    import subscription_llm as sub
    label = "LIVE-" + uuid.uuid4().hex[:8]
    config = os.path.join(directory, "mcp.json")
    with open(config, "w") as handle:
        json.dump({"mcpServers": {"insights": {
            "command": sys.executable,
            "args": [os.path.join(ROOT, "insights_mcp.py"), "--db", database.DB_PATH]}}}, handle)
    tools = ",".join("mcp__insights__" + n for n in (
        "get_writing_guide", "list_sources", "get_source", "list_posts", "save_drafts"))
    started = time.monotonic()
    run = subprocess.run(
        [sub.find_cli("claude-code"), "-p", "--output-format", "json", "--model", "claude-sonnet-5-5",
         "--mcp-config", config, "--strict-mcp-config", "--setting-sources", "", "--tools", "",
         "--allowedTools", tools, "--no-session-persistence", MCP_TASK.format(label=label)],
        capture_output=True, text=True, timeout=600, env=sub.child_env(), cwd=directory,
        stdin=subprocess.DEVNULL)
    check(run.returncode == 0, f"claude exited {run.returncode}: {run.stderr[-800:]} {run.stdout[-800:]}")
    result = json.loads(run.stdout)
    check(not result.get("is_error"), f"claude reported an error: {result.get('result')}")
    print(f"mcp-claude: claude -p finished in {time.monotonic() - started:.0f}s: {str(result.get('result'))[:80]!r}")
    check_mcp_rows(database, label, "Claude Code")
    print("LIVE_MCP_CLAUDE_OK")


def section_mcp_codex():
    directory, database = seeded_db()
    import subscription_llm as sub
    label = "LIVE-" + uuid.uuid4().hex[:8]
    args = [sub.find_cli("codex"), "exec", "--json", "--skip-git-repo-check", "--ephemeral",
            "--ignore-user-config", "--ignore-rules", "--sandbox", "read-only", "--color", "never",
            "--cd", directory,
            "-c", f"mcp_servers.insights.command={json.dumps(sys.executable)}",
            "-c", "mcp_servers.insights.args=" + json.dumps(
                [os.path.join(ROOT, "insights_mcp.py"), "--db", database.DB_PATH]),
            "-c", 'mcp_servers.insights.default_tools_approval_mode="approve"',
            "-c", 'web_search="disabled"', "-c", 'model_reasoning_effort="low"']
    # code_mode_host stays on: Codex 0.153 calls MCP tools through it, and
    # disabling it hides them (checked 2026-11-25). The plain-generation client
    # in subscription_llm.py turns it off, because there it has no MCP to reach.
    for feature in ("shell_tool", "unified_exec", "computer_use", "browser_use",
                    "plugins", "apps", "multi_agent", "image_generation"):
        args += ["--disable", feature]
    args.append(MCP_TASK.format(label=label))
    started = time.monotonic()
    run = subprocess.run(args, capture_output=True, text=True, timeout=600, env=sub.child_env(),
                         cwd=directory, stdin=subprocess.DEVNULL)
    events = [json.loads(l) for l in run.stdout.splitlines() if l.startswith("{")]
    calls = [e["item"] for e in events if e.get("type") == "item.completed"
             and (e.get("item") or {}).get("type") == "mcp_tool_call"]
    failed = [e for e in events if e.get("type") in ("turn.failed", "error")]
    print(f"mcp-codex: codex exec finished in {time.monotonic() - started:.0f}s with exit {run.returncode}; "
          f"tool calls: {[c.get('tool') for c in calls]}")
    check(not failed, f"codex reported: {failed[-1] if failed else ''}")
    check_mcp_rows(database, label, "Codex")
    print("LIVE_MCP_CODEX_OK")


SECTIONS = {"claude": section_claude, "codex": section_codex,
            "mcp-claude": section_mcp_claude, "mcp-codex": section_mcp_codex}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SECTIONS:
        print(f"usage: {sys.argv[0]} {{{'|'.join(SECTIONS)}}}")
        sys.exit(2)
    SECTIONS[sys.argv[1]]()
