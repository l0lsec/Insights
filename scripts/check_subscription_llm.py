"""Gates for running Insights' model calls on a Claude or ChatGPT subscription.

    python scripts/check_subscription_llm.py claude      SUB_CLAUDE_OK
    python scripts/check_subscription_llm.py codex       SUB_CODEX_OK
    python scripts/check_subscription_llm.py routing     SUB_ROUTING_OK
    python scripts/check_subscription_llm.py status      SUB_STATUS_OK
    python scripts/check_subscription_llm.py ui          SUB_UI_OK
    python scripts/check_subscription_llm.py mcp         SUB_MCP_OK
    python scripts/check_subscription_llm.py docs        SUB_DOCS_OK
    python scripts/check_subscription_llm.py registered  SUB_REGISTERED_OK
    python scripts/check_subscription_llm.py nodash      SUB_NODASH_OK

``claude`` and ``codex`` run the two CLI clients against fake ``claude`` and
``codex`` executables that write down exactly what they were given (arguments,
standard input, environment, working directory, image files) and answer from a
script, so every flag that keeps the CLIs to plain, unbilled text generation is
read back rather than assumed. ``routing`` and ``status`` drive the real
``_get_llm_client``, generators, metering and routes on a throwaway database.
``ui`` runs the Compose picker's own functions under node against a fake page.
``mcp`` starts the real MCP server as a subprocess and talks JSON-RPC to it.

None of them needs a key, a signed-in CLI or the network; scripts/live_subscription.py
holds the live checks.
"""

import base64
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
# Pin the default provider to the app's own default, as on CI. insights_web's
# load_dotenv() searches parent folders, so from a worktree it reaches the main
# checkout's .env; a live LLM_PROVIDER=claude-code there would change what the
# checks below see. load_dotenv never overrides a variable already set.
os.environ["LLM_PROVIDER"] = "openai"
from _accounts_gate import isolated_app, check  # noqa: E402

PY = sys.executable
# Built from code points so this file never contains the characters it hunts for.
EM, EN = chr(0x2014), chr(0x2013)

FAKE_CLAUDE = r'''#!__PY__
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
mode = json.load(open(os.path.join(HERE, "mode.json")))
args = sys.argv[1:]
if args[:1] == ["--version"]:
    print("9.9.9 (Claude Code)"); sys.exit(0)
if args[:2] == ["auth", "status"]:
    print(json.dumps(mode.get("auth", {"loggedIn": True, "authMethod": "claude.ai"}))); sys.exit(0)
stdin = sys.stdin.read()
cwd = os.getcwd()
record = {"argv": args, "stdin": stdin, "env": dict(os.environ), "cwd": cwd,
          "cwd_listing": os.listdir(cwd), "pid": os.getpid()}
with open(os.path.join(HERE, "record.json"), "w") as handle:
    json.dump(record, handle)
reply = mode.get("reply", "ok")
print(json.dumps({"type": "system", "subtype": "init", "tools": [], "mcp_servers": []}))
sys.stdout.flush()
if reply == "sleep":
    time.sleep(60); sys.exit(0)
if reply == "crash":
    sys.stderr.write("boom: something broke\n"); sys.exit(3)
result = {"type": "result", "subtype": "success", "is_error": False,
          "result": mode.get("text", "hello from claude"), "stop_reason": "end_turn",
          "usage": {"input_tokens": 10, "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 1, "output_tokens": 7},
          "modelUsage": {mode.get("model_used", "claude-opus-5-5"): {}}}
if reply == "error":
    result.update(is_error=True, result="There's an issue with the selected model (x).",
                  api_error_status=404, terminal_reason="api_error")
if reply == "login":
    result.update(is_error=True, result="Not logged in. Please run /login", terminal_reason="api_error")
if reply == "refusal":
    result.update(stop_reason="refusal", result="")
print(json.dumps(result))
'''

FAKE_CODEX = r'''#!__PY__
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
mode = json.load(open(os.path.join(HERE, "mode.json")))
args = sys.argv[1:]
if args[:1] == ["--version"]:
    if mode.get("broken"):
        sys.stderr.write("Error: spawn ENOENT\n"); sys.exit(1)
    print("codex-cli 9.9.9"); sys.exit(0)
if args[:2] == ["login", "status"]:
    text, code = mode.get("login", ["Logged in using ChatGPT", 0])
    print(text); sys.exit(code)
stdin = sys.stdin.read()
images = []
for i, a in enumerate(args):
    if a == "--image":
        p = args[i + 1]
        images.append({"path": p, "exists": os.path.exists(p),
                       "sha": hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None})
last = args[args.index("--output-last-message") + 1]
cwd = os.getcwd()
record = {"argv": args, "stdin": stdin, "env": dict(os.environ), "cwd": cwd,
          "cwd_listing": sorted(os.listdir(cwd)), "images": images}
with open(os.path.join(HERE, "record.json"), "w") as handle:
    json.dump(record, handle)
reply = mode.get("reply", "ok")
text = mode.get("text", "hello from codex")
print(json.dumps({"type": "thread.started", "thread_id": "t"}))
print(json.dumps({"type": "turn.started"}))
if reply == "ok":
    open(last, "w").write(text)
    print(json.dumps({"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": text}}))
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 50, "output_tokens": 9}}))
elif reply == "fail":
    inner = json.dumps({"type": "error", "status": 400, "error": {
        "type": "invalid_request_error",
        "message": "The 'x' model is not supported when using Codex with a ChatGPT account."}})
    print(json.dumps({"type": "error", "message": inner}))
    print(json.dumps({"type": "turn.failed", "error": {"message": inner}}))
    sys.exit(1)
elif reply == "nologin":
    print(json.dumps({"type": "turn.failed", "error": {"message": "401 Unauthorized: please run codex login"}}))
    sys.exit(1)
elif reply == "empty":
    sys.exit(1)
'''


class Fakes:
    """A directory holding fake ``claude`` and ``codex`` and their scripts."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="sub_fakes_")
        self.claude_dir = os.path.join(self.dir, "claude_bin")
        self.codex_dir = os.path.join(self.dir, "codex_bin")
        self.claude = self._write(self.claude_dir, "claude", FAKE_CLAUDE)
        self.codex = self._write(self.codex_dir, "codex", FAKE_CODEX)
        self.mode("claude", {})
        self.mode("codex", {})
        os.environ["CLAUDE_CLI_PATH"] = self.claude
        os.environ["CODEX_CLI_PATH"] = self.codex

    @staticmethod
    def _write(directory, name, body):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        with open(path, "w") as handle:
            handle.write(body.replace("__PY__", PY))
        os.chmod(path, 0o755)
        return path

    def mode(self, which, value):
        directory = self.claude_dir if which == "claude" else self.codex_dir
        with open(os.path.join(directory, "mode.json"), "w") as handle:
            json.dump(value, handle)
        record = os.path.join(directory, "record.json")
        if os.path.exists(record):
            os.remove(record)

    def record(self, which):
        directory = self.claude_dir if which == "claude" else self.codex_dir
        path = os.path.join(directory, "record.json")
        check(os.path.exists(path), f"the fake {which} was never run")
        with open(path) as handle:
            return json.load(handle)


SECRETS = {
    "ANTHROPIC_API_KEY": "sk-ant-should-never-reach-the-cli",
    "ANTHROPIC_MODEL": "claude-opus-4-8",
    "ANTHROPIC_BASE_URL": "https://example.invalid",
    "OPENAI_API_KEY": "sk-should-never-reach-the-cli",
    "OPENAI_BASE_URL": "https://example.invalid",
    "CODEX_API_KEY": "sk-codex-should-never-reach-the-cli",
    "TWILIO_AUTH_TOKEN": "twilio-secret",
    "CLOUDINARY_URL": "cloudinary://secret",
    "CLAUDECODE": "1",
}


def set_secrets():
    os.environ.update(SECRETS)
    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = "plan-token-kept"


def check_env(env, who):
    leaked = sorted(k for k in SECRETS if k in env)
    check(not leaked, f"{who} was given {leaked}; an API key there would bill the API")
    check(env.get("CLAUDE_CODE_OAUTH_TOKEN") == "plan-token-kept",
          f"{who} lost CLAUDE_CODE_OAUTH_TOKEN, the plan sign-in a server uses")
    check(env.get("HOME") and env.get("PATH"), f"{who} lost HOME or PATH, needed to find its sign-in")
    allowed = __import__("subscription_llm")._ENV_ALLOW
    extra = sorted(k for k in env if k not in allowed and not k.startswith(("LC_", "XDG_"))
                   and k not in ("PWD", "SHLVL", "_", "__CF_USER_TEXT_ENCODING", "OLDPWD"))
    check(not extra, f"{who} was given variables outside the allowlist: {extra}")


def flag_value(argv, flag):
    check(flag in argv, f"{flag} missing from {argv}")
    return argv[argv.index(flag) + 1]


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
PNG_B64 = base64.b64encode(PNG).decode()


# ── claude ───────────────────────────────────────────────────────────────

def section_claude():
    fakes = Fakes()
    set_secrets()
    import subscription_llm as sub
    import insights

    client = sub.ClaudeCodeClient()
    system = "-starts with a dash, and is the system prompt"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe these."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
                    {"type": "image_url", "image_url": {"url": "https://example.test/pic.jpg"}}]}]
    response = client.chat.completions.create(model="claude-opus-5-5", messages=messages,
                                              max_tokens=160, temperature=0.8)
    rec = fakes.record("claude")
    argv = rec["argv"]
    check(argv[0] == "-p", f"not print mode: {argv[:2]}")
    check(flag_value(argv, "--tools") == "", "tools are not switched off (--tools \"\")")
    check(flag_value(argv, "--setting-sources") == "", "settings files (hooks, plugins) are not skipped")
    check("--strict-mcp-config" in argv and "--mcp-config" not in argv, "MCP servers are not shut out")
    check("--no-session-persistence" in argv, "sessions would be saved")
    check(flag_value(argv, "--input-format") == "stream-json"
          and flag_value(argv, "--output-format") == "stream-json", "not stream-json in and out")
    check(f"--system-prompt={system}" in argv, "the system prompt is not passed whole in --system-prompt=")
    check("--model=claude-opus-5-5" in argv, "the model is not passed")
    check(flag_value(argv, "--effort") == "low", "Opus 5 is not given the low effort")
    banned = [a for a in argv if a in ("--dangerously-skip-permissions", "--allowedTools",
                                       "--permission-mode", "--bare", "--add-dir")]
    check(not banned, f"the call opens up the CLI: {banned}")
    check_env(rec["env"], "claude")
    check(rec["cwd_listing"] == [], f"the working directory was not empty: {rec['cwd_listing']}")
    check(not os.path.exists(rec["cwd"]), "the working directory was left behind")
    check(os.path.realpath(rec["cwd"]) != os.path.realpath(ROOT), "ran inside the repository")

    line = json.loads(rec["stdin"].strip())
    content = line["message"]["content"]
    check(line["type"] == "user" and content[0] == {"type": "text", "text": "Describe these."},
          f"the prompt is not the user's text: {content[0]}")
    check(content[1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
          "the data: image did not arrive as a base64 image block")
    check(content[2] == {"type": "image", "source": {"type": "url", "url": "https://example.test/pic.jpg"}},
          "the http image did not arrive as a url image block")

    check(response.choices[0].message.content == "hello from claude", "the answer text is wrong")
    check(response.usage.input_tokens == 16 and response.usage.prompt_tokens == 16,
          "input tokens should count cache reads and writes (10+5+1)")
    check(response.usage.output_tokens == 7 and response.usage.completion_tokens == 7, "output tokens wrong")
    check(response.model == "claude-opus-5-5", "the model the CLI used is not reported")

    # A model that does not think by default takes no effort flag.
    client.chat.completions.create(model="claude-haiku-4-5", messages=[{"role": "user", "content": "hi"}])
    rec = fakes.record("claude")
    check("--effort" not in rec["argv"], "Haiku 4.5 was given --effort, which it does not take")
    check(f"--system-prompt={sub.DEFAULT_SYSTEM}" in rec["argv"], "no default system prompt without one")

    # A conversation becomes a transcript ending on the last user message.
    client.chat.completions.create(model="claude-opus-5-5", messages=[
        {"role": "system", "content": "S"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"}])
    text = json.loads(fakes.record("claude")["stdin"])["message"]["content"][0]["text"]
    check("User: first question" in text and "Assistant: first answer" in text
          and text.rstrip().endswith("second question") and "Now reply" in text,
          f"the conversation was not replayed: {text!r}")

    def fails(mode, model="claude-opus-5-5"):
        fakes.mode("claude", mode)
        try:
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "x"}])
        except Exception as exc:  # noqa: BLE001
            return exc
        fail_msg = f"mode {mode} did not raise"
        check(False, fail_msg)

    exc = fails({"reply": "login"})
    check(isinstance(exc, sub.SubscriptionLLMError) and "not signed in" in str(exc)
          and "claude" in str(exc), f"not-signed-in reads: {exc}")
    exc = fails({"reply": "error"})
    check(isinstance(exc, sub.SubscriptionLLMError) and "issue with the selected model" in str(exc),
          f"a CLI error is not passed on: {exc}")
    exc = fails({"reply": "refusal"})
    check(isinstance(exc, insights.LLMRefusal), f"a refusal is not an LLMRefusal: {exc!r}")
    exc = fails({"reply": "crash"})
    check(isinstance(exc, sub.SubscriptionLLMError) and "exit 3" in str(exc) and "boom" in str(exc),
          f"a crash does not say why: {exc}")

    os.environ["SUBSCRIPTION_LLM_TIMEOUT"] = "2"
    started = time.monotonic()
    exc = fails({"reply": "sleep"})
    took = time.monotonic() - started
    os.environ.pop("SUBSCRIPTION_LLM_TIMEOUT")
    check(isinstance(exc, sub.SubscriptionLLMError) and "longer than 2 seconds" in str(exc),
          f"a hung CLI is not stopped: {exc}")
    check(took < 15, f"the timeout took {took:.0f}s")
    pid = fakes.record("claude")["pid"]
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    check(not alive, "the hung CLI process was left running")

    os.environ["CLAUDE_CLI_PATH"] = "/nonexistent/claude"
    exc = fails({})
    check(isinstance(exc, sub.SubscriptionLLMError) and "CLAUDE_CLI_PATH" in str(exc),
          f"a missing CLI does not say how to fix it: {exc}")
    print("claude: no tools, no MCP, no settings, no keys, empty temp dir; text, images, "
          "conversation, usage; login, error, refusal, crash, timeout, missing CLI")
    print("SUB_CLAUDE_OK")


# ── codex ────────────────────────────────────────────────────────────────

def section_codex():
    fakes = Fakes()
    set_secrets()
    import subscription_llm as sub

    sub._fetch_image = lambda url: ("image/jpeg", b"remote-jpeg-bytes")
    client = sub.CodexClient()
    response = client.chat.completions.create(model="gpt-5.6-sol", max_tokens=100, temperature=0.3, messages=[
        {"role": "system", "content": "Write like a pirate."},
        {"role": "user", "content": [
            {"type": "text", "text": "Caption this."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
            {"type": "image_url", "image_url": {"url": "https://example.test/remote.jpg"}}]}])
    rec = fakes.record("codex")
    argv = rec["argv"]
    check(argv[0] == "exec", f"not codex exec: {argv[:2]}")
    check(argv[-1] == "-", "the prompt is not read from standard input")
    for flag in ("--json", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--ignore-rules"):
        check(flag in argv, f"{flag} missing")
    check(flag_value(argv, "--sandbox") == "read-only", "the sandbox is not read-only")
    configs = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
    check('web_search="disabled"' in configs, f"web search is not off: {configs}")
    check('model_reasoning_effort="low"' in configs, "reasoning effort is not low")
    disabled = {argv[i + 1] for i, a in enumerate(argv) if a == "--disable"}
    for feature in ("shell_tool", "unified_exec", "code_mode_host", "computer_use", "browser_use",
                    "plugins", "apps", "multi_agent", "image_generation", "hooks"):
        check(feature in disabled, f"Codex feature {feature} is left on")
    check(flag_value(argv, "--model") == "gpt-5.6-sol", "the model is not passed")
    banned = [a for a in argv if "dangerously" in a or a in ("danger-full-access", "workspace-write", "--full-auto")]
    check(not banned, f"the call opens up the sandbox: {banned}")
    check(os.path.realpath(flag_value(argv, "--cd")) == os.path.realpath(rec["cwd"]),
          "--cd is not the working directory")
    check_env(rec["env"], "codex")
    check(not os.path.exists(rec["cwd"]), "the working directory was left behind")

    images = rec["images"]
    check(len(images) == 2 and all(i["exists"] for i in images), f"images were not files: {images}")
    check(images[0]["sha"] == hashlib.sha256(PNG).hexdigest(), "the data: image bytes changed on the way")
    check(images[1]["sha"] == hashlib.sha256(b"remote-jpeg-bytes").hexdigest() and images[1]["path"].endswith(".jpg"),
          "the http image was not downloaded to a .jpg file")
    first_flag = min(i for i, a in enumerate(argv) if a.startswith("--") and a != "--image")
    check(all(argv.index("--image") < first_flag for _ in [0]), "images must come before the other flags")

    stdin = rec["stdin"]
    check(stdin.startswith(sub.CODEX_PREAMBLE), "the plain-text preamble is not first")
    check("<instructions>\nWrite like a pirate.\n</instructions>" in stdin, "the system prompt is not in the prompt")
    check(stdin.rstrip().endswith("Caption this."), "the user's text is not last")

    check(response.choices[0].message.content == "hello from codex", "the answer text is wrong")
    check(response.usage.input_tokens == 100 and response.usage.output_tokens == 9, "usage is wrong")

    def fails(mode):
        fakes.mode("codex", mode)
        try:
            client.chat.completions.create(model="gpt-6-astra", messages=[{"role": "user", "content": "x"}])
        except sub.SubscriptionLLMError as exc:
            return str(exc)
        check(False, f"mode {mode} did not raise")

    message = fails({"reply": "fail"})
    check(message == "Codex could not answer: The 'x' model is not supported when using Codex with a ChatGPT account.",
          f"the API's own message is not unwrapped: {message}")
    message = fails({"reply": "nologin"})
    check("not signed in" in message and "codex login" in message, f"not-signed-in reads: {message}")
    message = fails({"reply": "empty"})
    check("without an answer" in message and "exit 1" in message, f"an empty answer reads: {message}")
    print("codex: user config ignored, read-only, web search and acting tools off, no keys; "
          "images as files, instructions in the prompt, usage; API error, login, empty")
    print("SUB_CODEX_OK")


# ── routing ──────────────────────────────────────────────────────────────

def section_routing():
    directory, database, web, publisher, http = isolated_app()
    fakes = Fakes()
    set_secrets()
    import insights
    import subscription_llm as sub

    def route(provider=None, model=None):
        client, m, p = insights._get_llm_client(provider=provider, model=model)
        return type(client).__name__, m, p

    check(route("claude-code") == ("ClaudeCodeClient", insights.CLAUDE_CODE_MODEL, "claude-code"),
          f"claude-code: {route('claude-code')}")
    check(route("claude-code", "claude-haiku-4-5")[1] == "claude-haiku-4-5", "a Claude model is not kept")
    check(route("claude-code", "sonnet")[1] == "sonnet", "a Claude Code alias is not kept")
    check(route("claude-code", "gpt-6-luna")[1] == insights.CLAUDE_CODE_MODEL, "a GPT model reached Claude Code")
    check(route("codex") == ("CodexClient", insights.CODEX_MODEL, "codex"), f"codex: {route('codex')}")
    check(route("codex", "gpt-5.6-sol")[1] == "gpt-5.6-sol", "a plan model is not kept")
    check(route("codex", "gpt-6-luna")[1] == insights.CODEX_MODEL,
          "an API-only model (CONTENT_AGENT_MODEL) reached Codex, which a ChatGPT plan refuses")
    check(route("codex", "claude-opus-5-5")[1] == insights.CODEX_MODEL, "a Claude model reached Codex")

    original = insights.LLM_PROVIDER
    try:
        insights.LLM_PROVIDER = "openai"
        check(route("subscription")[2] == "claude-code", "subscription should mean Claude Code by default")
        check(route()[0] == "_OpenAIChatClient", "the OpenAI path changed")
        check(route("anthropic")[0] == "_AnthropicChatClient", "the Anthropic path changed")
        insights.LLM_PROVIDER = "codex"
        check(route("subscription")[2] == "codex", "subscription should follow LLM_PROVIDER=codex")
        check(route()[:1] == ("CodexClient",) and route(model=os.environ.get("CONTENT_AGENT_MODEL", "gpt-6-luna"))[1]
              == insights.CODEX_MODEL, "a caller naming no provider does not follow LLM_PROVIDER=codex")
        options = insights.provider_model_options()
        check(options["default_provider"] == "subscription" and options["default_subscription"] == "codex",
              f"page defaults: {options['default_provider']}, {options['default_subscription']}")
        check(options["models"]["codex"][0] == insights.CODEX_MODEL
              and set(options["models"]["claude-code"]) >= set(insights.MODEL_CHOICES["anthropic"]),
              "the subscription model lists are wrong")
        insights.LLM_PROVIDER = "claude-code"
        check(route()[2] == "claude-code", "a caller naming no provider does not follow LLM_PROVIDER=claude-code")
    finally:
        insights.LLM_PROVIDER = original

    # The web form's provider reaches the generators as sent.
    for value in ("claude-code", "codex", "subscription"):
        parsed = web._parse_ai_provider({"provider": value, "model": "m"})
        check(parsed == (value, "m", False), f"_parse_ai_provider({value}) = {parsed}")

    # Background callers get their chat clients only through _get_llm_client.
    for name in ("research_engine.py", "outreach_agent.py", "content_library.py", "content_agent.py",
                 "inbox.py", "outreach_social.py", "outreach_worker.py"):
        source = open(os.path.join(ROOT, name)).read()
        direct = re.findall(r"\b(OpenAI|Anthropic)\(", source)
        check(not direct, f"{name} builds its own {direct} client, which would bypass the subscription")

    # End to end: a generator, through the CLI, parsed, and metered at no cost.
    fakes.mode("claude", {"text": '```json\n{"twitter": ["tweet one"], "linkedin": ["post one"]}\n```'})
    result = insights.generate_posts_from_prompt("coffee", platforms=["twitter", "linkedin"],
                                                 posts_per_platform=1, provider="claude-code")
    flat = json.dumps(result)
    check("tweet one" in flat and "post one" in flat, f"the generator did not return the CLI's posts: {result}")
    rows = sqlite3.connect(database.DB_PATH).execute(
        "SELECT provider, model, prompt_tokens, completion_tokens, cost_usd FROM usage_events "
        "WHERE provider = 'claude-code'").fetchall()
    check(rows and rows[-1][2:] == (16, 7, 0.0), f"the call is not metered as free: {rows}")

    fakes.mode("codex", {"text": '{"instagram": "a caption"}'})
    insights.generate_posts_from_images([{"mime_type": "image/png", "base64": PNG_B64}],
                                        platforms=["instagram"], posts_per_platform=1, provider="codex")
    rec = fakes.record("codex")
    check(rec["images"] and rec["images"][0]["sha"] == hashlib.sha256(PNG).hexdigest(),
          "the image generator's picture did not reach Codex")

    # A subscription failure reaches the user in plain words, not as "Failed to generate".
    fakes.mode("claude", {"reply": "login"})
    response = http.post("/compose/generate", data={
        "source_type": "freeform", "content": "coffee", "platforms": ["twitter"],
        "posts_per_platform": "1", "provider": "claude-code"})
    body = response.get_json() or {}
    check(response.status_code == 500 and "not signed in" in body.get("error", ""),
          f"the route hid the reason: {response.status_code} {body}")
    print("routing: claude-code, codex, subscription and LLM_PROVIDER reach the CLIs with models they take; "
          "paid paths unchanged; generators work through them; metered free; failures explained")
    print("SUB_ROUTING_OK")


# ── status ───────────────────────────────────────────────────────────────

def section_status():
    directory, database, web, publisher, http = isolated_app()
    web._maybe_attach_link_image = lambda *a, **k: None
    fakes = Fakes()
    set_secrets()
    import subscription_llm as sub

    def claude_status(auth):
        fakes.mode("claude", {"auth": auth})
        return sub.status("claude-code", refresh=True)

    info = claude_status({"loggedIn": True, "authMethod": "claude.ai"})
    check(info["available"] and info["signed_in"] and info["subscription"], f"plan sign-in: {info}")
    info = claude_status({"loggedIn": True, "authMethod": "apiKey"})
    check(info["signed_in"] and not info["subscription"] and "billed" in info["detail"], f"API-key sign-in: {info}")
    info = claude_status({"loggedIn": False})
    check(info["available"] and not info["signed_in"] and "sign in" in info["detail"], f"signed out: {info}")

    def codex_status(login):
        fakes.mode("codex", {"login": login})
        return sub.status("codex", refresh=True)

    info = codex_status(["Logged in using ChatGPT", 0])
    check(info["signed_in"] and info["subscription"], f"ChatGPT sign-in: {info}")
    info = codex_status(["Logged in using an API key - sk-...abc", 0])
    check(info["signed_in"] and not info["subscription"] and "billed" in info["detail"], f"API-key sign-in: {info}")
    info = codex_status(["Not logged in", 1])
    check(not info["signed_in"] and "codex login" in info["detail"], f"signed out: {info}")

    os.environ["CODEX_CLI_PATH"] = "/nonexistent/codex"
    info = sub.status("codex", refresh=True)
    check(not info["available"] and "CODEX_CLI_PATH" in info["detail"], f"missing: {info}")

    # Without an override, a broken shim on PATH is passed over for a working copy.
    os.environ.pop("CODEX_CLI_PATH")
    broken_dir = tempfile.mkdtemp(prefix="broken_codex_")
    broken = Fakes._write(broken_dir, "codex", FAKE_CODEX)
    with open(os.path.join(broken_dir, "mode.json"), "w") as handle:
        json.dump({"broken": True}, handle)
    saved_path, saved_candidates = os.environ["PATH"], sub._CANDIDATES["codex"]
    os.environ["PATH"] = broken_dir + os.pathsep + saved_path
    sub._CANDIDATES["codex"] = ("CODEX_CLI_PATH", "codex", (fakes.codex,))
    sub._cli_cache.clear()
    try:
        found = sub.find_cli("codex")
    finally:
        os.environ["PATH"] = saved_path
        sub._CANDIDATES["codex"] = saved_candidates
        os.environ["CODEX_CLI_PATH"] = fakes.codex
        sub._cli_cache.clear()
    check(os.path.realpath(found) == os.path.realpath(fakes.codex) and found != broken,
          f"picked {found} over the working copy")

    # The route: both CLIs, which are ready, and refresh skips the cache.
    fakes.mode("claude", {"auth": {"loggedIn": True, "authMethod": "claude.ai"}})
    fakes.mode("codex", {"login": ["Not logged in", 1]})
    data = http.get("/compose/subscription-status?refresh=1").get_json()
    check(set(data["providers"]) == {"claude-code", "codex"} and data["ready"] == ["claude-code"],
          f"route: {data}")
    fakes.mode("codex", {"login": ["Logged in using ChatGPT", 0]})
    cached = http.get("/compose/subscription-status").get_json()
    fresh = http.get("/compose/subscription-status?refresh=1").get_json()
    check(cached["ready"] == ["claude-code"] and fresh["ready"] == ["claude-code", "codex"],
          f"cache/refresh: {cached['ready']} then {fresh['ready']}")

    # Compose generation with the subscription saves the CLI's posts.
    fakes.mode("claude", {"text": '{"twitter": ["from the plan"], "linkedin": ["also from the plan"]}'})
    response = http.post("/compose/generate", data={
        "source_type": "freeform", "content": "coffee", "platforms": ["twitter", "linkedin"],
        "posts_per_platform": "1", "provider": "claude-code", "model": "claude-sonnet-5-5"})
    check(response.status_code == 200, f"/compose/generate: {response.status_code} {response.get_data(as_text=True)[:300]}")
    check("--model=claude-sonnet-5-5" in fakes.record("claude")["argv"], "the picked model did not reach the CLI")
    saved = sqlite3.connect(database.DB_PATH).execute(
        "SELECT platform, content FROM standalone_posts ORDER BY id").fetchall()
    check(("twitter", "from the plan") in saved and ("linkedin", "also from the plan") in saved,
          f"the generated posts were not saved: {saved}")
    print("status: installed, signed in, plan or API key, for both; broken shim skipped; route and refresh; "
          "Compose generation saved the CLI's posts")
    print("SUB_STATUS_OK")


# ── ui ───────────────────────────────────────────────────────────────────

def inline_scripts(html):
    blocks = []
    for match in re.finditer(r"<script([^>]*)>(.*?)</script>", html, re.S | re.I):
        attrs, body = match.group(1), match.group(2)
        if "src=" not in attrs and body.strip():
            blocks.append(body)
    return blocks


def lift(source, names, declarations):
    pieces = []
    for pattern in declarations:
        match = re.search(pattern, source, re.M | re.S)
        check(match, f"could not lift {pattern}")
        pieces.append(match.group(0))
    for name in names:
        found = re.findall(rf"^(?:async\s+)?function {name}\(", source, re.M)
        check(len(found) == 1, f"{name} is declared {len(found)} times on the page, expected once")
        match = re.search(rf"^(?:async\s+)?function {name}\(.*?^\}}\n", source, re.M | re.S)
        pieces.append(match.group(0))
    return "\n".join(pieces)


UI_PRELUDE = r"""
class El {
  constructor(id, tag) { this.id = id; this.tagName = (tag || 'div').toUpperCase(); this.children = [];
    this.dataset = {}; this.disabled = false; this.checked = false; this.title = ''; this._class = new Set();
    this.textContent = ''; this.label = ''; this.value_ = ''; this._selected = false; this.owner = null; }
  get className() { return [...this._class].join(' '); }
  set className(v) { this._class = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get classList() { const s = this; return {
    toggle(c, f) { const on = f === undefined ? !s._class.has(c) : !!f; on ? s._class.add(c) : s._class.delete(c); return on; },
    contains(c) { return s._class.has(c); }, add(c) { s._class.add(c); }, remove(c) { s._class.delete(c); } }; }
  set innerHTML(v) { this.children = []; }
  appendChild(c) { c.parentNode = this; this.children.push(c);
    const sel = this.tagName === 'SELECT' ? this : this.owner;
    if (sel) { c.owner = sel; c.children.forEach(k => { k.owner = sel; }); }
    return c; }
  get value() { if (this.tagName !== 'SELECT') return this.value_;
    const o = this.selectedOptions[0]; return o ? o.value : ''; }
  set value(v) { if (this.tagName !== 'SELECT') { this.value_ = v; return; }
    const o = this.options.find(x => x.value === v); this.options.forEach(x => { x._selected = (x === o); }); }
  get options() { const out = []; const walk = n => n.children.forEach(k => {
      if (k.tagName === 'OPTION') out.push(k); else walk(k); }); walk(this); return out; }
  get selectedOptions() { const opts = this.options; const s = opts.find(o => o._selected);
    if (s) return [s]; const first = opts.find(o => !o.disabled); return first ? [first] : []; }
  get selected() { return this._selected; }
  set selected(v) { if (v && this.owner) this.owner.options.forEach(o => { o._selected = (o === this); }); else this._selected = !!v; }
}
const els = {};
function mk(id, tag) { els[id] = new El(id, tag); return els[id]; }
const radios = ['openai', 'anthropic', 'subscription', 'local'].map(p => {
  const r = mk('provider-' + p, 'input'); r.value_ = p; r.name = 'aiProvider'; return r; });
mk('model-select', 'select'); mk('subscription-status-text', 'span'); mk('subscription-status-dot', 'span');
mk('provider-subscription-label', 'label'); mk('ollama-status-text', 'span'); mk('ollama-status-dot', 'span');
els['subscription-status-text'].className = 'small ms-1 d-none';
const document = {
  getElementById: id => els[id] || null,
  createElement: tag => new El(null, tag),
  querySelector: q => q === 'input[name="aiProvider"]:checked' ? (radios.find(r => r.checked) || null) : null,
  querySelectorAll: q => q === 'input[name="aiProvider"]' ? radios : [],
};
radios.forEach(r => Object.defineProperty(r, 'value', { get() { return this.value_; } }));
let fetchResponse = null, fetchFail = false; const fetchCalls = [];
async function fetch(url) { fetchCalls.push(url); if (fetchFail) throw new Error('offline');
  return { json: async () => fetchResponse }; }
function choose(p) { radios.forEach(r => { r.checked = (r.value === p); }); }
function state() { const sel = els['model-select'], txt = els['subscription-status-text'];
  return { provider: currentProvider(), choice: currentProviderChoice(), value: sel.value,
    groups: sel.children.filter(c => c.tagName === 'OPTGROUP').map(g => g.label),
    options: sel.options.map(o => [o.dataset.provider || '', o.value, o.disabled]),
    statusHidden: txt.classList.contains('d-none'), statusText: txt.textContent, statusClass: txt.className,
    dot: els['subscription-status-dot'].className, radioDisabled: els['provider-subscription'].disabled,
    title: els['provider-subscription-label'].title }; }
"""

UI_SCENARIOS_DEFAULT = r"""
(async () => {
  const out = {};
  choose('subscription'); populateModelDropdown(); out.a = state();
  els['model-select'].options.find(o => o.dataset.provider === 'codex' && o.value === 'gpt-5.6-sol').selected = true;
  out.b = state();
  fetchResponse = {providers: {'claude-code': {available: true, signed_in: true, detail: 'Signed in with your Claude plan.'},
                               codex: {available: true, signed_in: false, detail: 'Not signed in: run `codex login` in a terminal.'}},
                   ready: ['claude-code']};
  await checkSubscriptionStatus(); out.c = state();
  fetchResponse = {providers: {'claude-code': {available: false, signed_in: false, detail: 'Claude Code was not found.'},
                               codex: {available: true, signed_in: false, detail: 'Not signed in.'}}, ready: []};
  await checkSubscriptionStatus(); out.d = state();
  choose('openai'); populateModelDropdown(); out.e = state();
  fetchFail = true; await checkSubscriptionStatus(); out.f = state();
  out.fetchCalls = fetchCalls;
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e.stack); process.exit(1); });
"""

UI_SCENARIOS_CONFIGURED = r"""
(async () => {
  const out = {};
  (document.getElementById('provider-' + MODEL_OPTIONS.default_provider) || document.getElementById('provider-openai')).checked = true;
  populateModelDropdown(); out.a = state();
  fetchResponse = {providers: {'claude-code': {available: true, signed_in: false, detail: 'Not signed in.'},
                               codex: {available: true, signed_in: false, detail: 'Not signed in.'}}, ready: []};
  await checkSubscriptionStatus(); out.b = state();
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e.stack); process.exit(1); });
"""

UI_NAMES = ("currentProviderChoice", "currentProvider", "subscriptionReady", "populateSubscriptionModels",
            "populateModelDropdown", "showSubscriptionStatus", "checkSubscriptionStatus")
UI_DECLS = (r"^const MODEL_OPTIONS = .*?;\n", r"^let _ollamaModels = .*?;", r"^let _subscriptionStatus = null;",
            r"^const SUBSCRIPTION_LABELS = .*?;\n")


def run_ui(source, scenarios):
    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    script = UI_PRELUDE + "\n" + lift(source, UI_NAMES, UI_DECLS) + "\n" + scenarios
    work = tempfile.mkdtemp(prefix="sub_ui_")
    path = os.path.join(work, "ui.js")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(script)
    run = subprocess.run([node, path], capture_output=True, text=True, timeout=60)
    shutil.rmtree(work, ignore_errors=True)
    check(run.returncode == 0, f"the harness crashed:\n{run.stderr[-1500:]}")
    return json.loads(run.stdout.strip().splitlines()[-1])


def section_ui():
    directory, database, web, publisher, http = isolated_app()
    web._maybe_attach_link_image = lambda *a, **k: None
    import insights

    database.add_standalone_post("mcp", "From Claude", "twitter", "drafted over MCP", db_path=database.DB_PATH)
    database.add_standalone_post("manual", "", "twitter", "written by hand", db_path=database.DB_PATH)
    html = http.get("/compose").get_data(as_text=True)
    for needle in ('id="provider-subscription" value="subscription"', "My subscription",
                   'id="subscription-status-text"', 'id="subscription-status-dot"',
                   '<option value="assistant">'):
        check(needle in html, f"/compose is missing {needle!r}")
    check(html.index('id="provider-anthropic"') < html.index('id="provider-subscription"')
          < html.index('id="provider-local"'), "My subscription is not between the paid APIs and Local")
    check("🤝 Claude or Codex" in html and 'data-source="assistant"' in html,
          "a draft saved over MCP is not labelled on its card")
    filtered = http.get("/compose/posts/more?offset=0&source=assistant").get_data(as_text=True)
    check("drafted over MCP" in filtered and "written by hand" not in filtered,
          "the Source filter does not separate Claude or Codex drafts")
    check(web._post_source_kind("agent") == "agent" and web._post_source_kind("freeform") == "manual",
          "the existing Source buckets changed")

    source = "\n".join(inline_scripts(html))
    generate = re.search(r"^function generatePosts\(.*?^\}\n", source, re.M | re.S).group(0)
    check("const aiProvider = currentProvider();" in generate and "formData.append('provider', aiProvider);" in generate,
          "generatePosts does not send the picked CLI as the provider")

    out = run_ui(source, UI_SCENARIOS_DEFAULT)
    a, b, c, d, e, f = (out[k] for k in "abcdef")
    check(a["groups"] == ["Claude Code", "Codex"] and a["provider"] == "claude-code"
          and a["value"] == insights.CLAUDE_CODE_MODEL and not a["statusHidden"]
          and "Checking" in a["statusText"], f"before the check answers: {a}")
    check(b["provider"] == "codex" and b["value"] == "gpt-5.6-sol", f"picking a Codex model: {b}")
    codex_opts = [o for o in c["options"] if o[0] == "codex"]
    check(all(o[2] for o in codex_opts) and not any(o[2] for o in c["options"] if o[0] == "claude-code"),
          f"only signed-in models should be pickable: {c['options']}")
    check(c["groups"] == ["Claude Code", "Codex (not signed in)"] and c["provider"] == "claude-code",
          f"a Codex pick is moved to a signed-in model: {c}")
    check("No API credits used." in c["statusText"] and "codex login" in c["statusText"]
          and "text-muted" in c["statusClass"] and "text-success" in c["dot"]
          and not c["radioDisabled"] and "Claude Code" in c["title"], f"status line: {c}")
    check(d["radioDisabled"] and "text-danger" in d["statusClass"] and "Neither is ready." in d["statusText"]
          and "text-danger" in d["dot"] and "install Claude Code or Codex" in d["title"],
          f"nothing ready should disable it and say why: {d}")
    check(e["statusHidden"] and e["provider"] == "openai" and not e["groups"]
          and [o[1] for o in e["options"]][0] == insights.OPENAI_MODEL, f"OpenAI picked: {e}")
    check("text-danger" in f["dot"], f"a failed check is not shown: {f}")
    check(all(u.endswith("/compose/subscription-status") for u in out["fetchCalls"]), out["fetchCalls"])

    original = insights.LLM_PROVIDER
    insights.LLM_PROVIDER = "codex"
    try:
        configured = http.get("/compose").get_data(as_text=True)
    finally:
        insights.LLM_PROVIDER = original
    out = run_ui("\n".join(inline_scripts(configured)), UI_SCENARIOS_CONFIGURED)
    check(out["a"]["choice"] == "subscription" and out["a"]["provider"] == "codex"
          and out["a"]["value"] == insights.CODEX_MODEL, f"LLM_PROVIDER=codex default: {out['a']}")
    check(out["b"]["choice"] == "subscription" and not out["b"]["radioDisabled"]
          and "text-danger" in out["b"]["statusClass"],
          f"a configured subscription with no sign-in must stay picked and say so, not switch to a paid API: {out['b']}")
    print("ui: one My subscription choice, models grouped by CLI, the pick decides the provider, "
          "status says who is signed in, nothing ready disables it, never falls back to paid; MCP drafts labelled and filterable")
    print("SUB_UI_OK")


# ── mcp ──────────────────────────────────────────────────────────────────

class MCP:
    def __init__(self, db):
        self.proc = subprocess.Popen([PY, os.path.join(ROOT, "insights_mcp.py"), "--db", db],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, cwd=tempfile.mkdtemp(prefix="mcp_cwd_"))
        self.next_id = 0
        self.lines = []

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def read(self):
        line = self.proc.stdout.readline()
        if not line:  # read stderr only now: it blocks until the server exits
            check(False, f"the server stopped: {self.proc.stderr.read()[-800:]}")
        self.lines.append(line)
        return json.loads(line)

    def request(self, method, params=None):
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}})
        reply = self.read()
        check(reply.get("id") == self.next_id, f"reply id {reply.get('id')} for request {self.next_id}")
        return reply

    def call(self, name, **arguments):
        reply = self.request("tools/call", {"name": name, "arguments": arguments})
        result = reply["result"]
        return result["isError"], result["content"][0]["text"], result.get("structuredContent")

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def section_mcp():
    directory = tempfile.mkdtemp(prefix="mcp_gate_")
    db = os.path.join(directory, "insights.db")
    import database
    database.init_db(db)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO feeds (url, title) VALUES ('https://feed.test/rss', 'Feed')")
    transcript = "word " * 30000
    conn.execute("INSERT INTO episodes (feed_id, url, title, transcript, summary, status) "
                 "VALUES (1, 'https://ep.test/1', 'Episode about passkeys', ?, 'A short summary.', 'complete')",
                 (transcript,))
    conn.execute("INSERT INTO url_sources (url, title, description, content, created_at) "
                 "VALUES ('https://page.test', 'A saved page', 'desc', 'page body', '2026-01-01')")
    conn.execute("INSERT INTO articles (episode_id, topic, style, content, created_at) "
                 "VALUES (1, 'An article', 'blog', 'article body', '2026-01-02')")
    conn.commit()
    published = database.add_standalone_post("manual", "", "twitter", "already out", db_path=db)
    conn.execute("UPDATE standalone_posts SET used = 1 WHERE id = ?", (published,))
    conn.commit()
    queued = database.add_standalone_post("manual", "", "linkedin", "waiting in the queue", db_path=db)
    conn.execute("INSERT INTO scheduled_posts (post_type, platform, scheduled_for, status, standalone_post_id, created_at) "
                 "VALUES ('standalone', 'linkedin', '2099-01-01T09:00:00', 'pending', ?, '2026-01-01')", (queued,))
    conn.commit()
    conn.close()

    server = MCP(db)
    init = server.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                         "clientInfo": {"name": "gate", "version": "1"}})
    check(init["result"]["protocolVersion"] == "2025-06-18" and "tools" in init["result"]["capabilities"],
          f"initialize: {init}")
    server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    check(server.request("ping")["result"] == {}, "ping")
    tools = server.request("tools/list")["result"]["tools"]
    names = sorted(t["name"] for t in tools)
    check(names == sorted(["get_writing_guide", "list_sources", "get_source", "list_posts",
                           "save_drafts", "update_draft"]), f"tools: {names}")
    check(not [n for n in names if re.search(r"publish|schedul|queue|delete|send|token|post_now", n)],
          f"a tool can act outside drafts: {names}")
    check(all(t.get("inputSchema", {}).get("type") == "object" and t.get("description") for t in tools),
          "every tool needs a description and an object schema")

    bad, text, guide = server.call("get_writing_guide", platforms=["twitter"])
    check(not bad and guide["platforms"]["twitter"]["max_characters"] == 280
          and "em dash" in guide["house_style"] and guide["examples_of_published_posts"]["twitter"] == ["already out"],
          f"guide: {text[:300]}")

    bad, text, listed = server.call("list_sources")
    kinds = sorted(s["kind"] for s in listed["sources"])
    check(not bad and kinds == ["article", "episode", "url"], f"sources: {kinds}")
    bad, text, page = server.call("get_source", kind="episode", id=1, max_chars=1000)
    check(not bad and page["text"].startswith("SUMMARY\nA short summary.") and len(page["text"]) == 1000
          and page["next_offset"] == 1000 and page["total_chars"] > 150000, f"episode page: {text[:200]}")
    bad, text, page2 = server.call("get_source", kind="episode", id=1, offset=1000, max_chars=1000)
    check(not bad and page2["offset"] == 1000, "paging")
    check(server.call("get_source", kind="url", id=1)[2]["text"] == "desc\n\npage body", "url source")
    check(server.call("get_source", kind="article", id=1)[2]["text"] == "article body", "article")
    bad, text, _ = server.call("get_source", kind="episode", id=99)
    check(bad and "no episode 99" in text, f"missing source: {text}")

    bad, text, saved = server.call("save_drafts", source_label="Episode about passkeys", posts=[
        {"content": "Passkeys beat passwords. #security", "platforms": ["twitter", "X", "linkedin"]},
        {"content": "A longer LinkedIn take on passkeys.", "platforms": ["linkedin"],
         "image_url": "https://img.test/a.png"}])
    check(not bad, f"save_drafts refused good posts: {text}")
    rows = sqlite3.connect(db).execute(
        "SELECT id, source_type, source_content, platform, content, used, image_url FROM standalone_posts "
        "WHERE source_type = 'mcp' ORDER BY id").fetchall()
    check([(r[3], r[4]) for r in rows] == [
        ("twitter", "Passkeys beat passwords. #security"), ("linkedin", "Passkeys beat passwords. #security"),
        ("linkedin", "A longer LinkedIn take on passkeys.")], f"rows: {rows}")
    check(all(r[5] == 0 and r[2] == "Episode about passkeys" for r in rows), "drafts must be unused and labelled")
    check(saved["saved"][0]["platforms"] == ["twitter", "linkedin"], "X should alias to twitter, once")
    queued_rows = sqlite3.connect(db).execute(
        "SELECT COUNT(*) FROM scheduled_posts WHERE standalone_post_id IN (%s)" % ",".join(str(r[0]) for r in rows)
    ).fetchone()[0]
    check(queued_rows == 0, "saving a draft queued something")

    before = sqlite3.connect(db).execute("SELECT COUNT(*) FROM standalone_posts").fetchone()[0]
    for posts, needle in (
            ([{"content": "fine", "platforms": ["twitter"]}, {"content": "bad " + EM + " dash", "platforms": ["twitter"]}], "em dash"),
            ([{"content": "fine", "platforms": ["myspace"]}], "Unknown platform"),
            ([{"content": "  ", "platforms": ["twitter"]}], "empty"),
            ([{"content": "x", "platforms": ["twitter"], "image_url": "file:///etc/passwd"}], "http(s)")):
        bad, text, _ = server.call("save_drafts", posts=posts)
        check(bad and needle in text, f"{needle}: {text}")
    after = sqlite3.connect(db).execute("SELECT COUNT(*) FROM standalone_posts").fetchone()[0]
    check(before == after, "a refused batch still wrote rows")
    bad, text, long_save = server.call("save_drafts", posts=[{"content": "y" * 300, "platforms": ["twitter"]}])
    check(not bad and long_save["warnings"] and "280" in long_save["warnings"][0], "an over-long tweet is not warned")

    bad, text, listed = server.call("list_posts", status="drafts")
    drafts = {p["content"]: p for p in listed["posts"]}
    check("Passkeys beat passwords. #security" in drafts
          and drafts["Passkeys beat passwords. #security"]["platforms"] == ["twitter", "linkedin"]
          and "waiting in the queue" not in drafts and "already out" not in drafts, f"drafts: {list(drafts)}")
    check([p["content"] for p in server.call("list_posts", status="queued")[2]["posts"]] == ["waiting in the queue"],
          "queued listing")
    check("already out" in [p["content"] for p in server.call("list_posts", status="posted")[2]["posts"]],
          "posted listing")

    first = rows[0][0]
    bad, text, updated = server.call("update_draft", post_id=first, content="Passkeys, rewritten.")
    check(not bad and sorted(updated["updated"]) == sorted([rows[0][0], rows[1][0]]), f"update: {text}")
    contents = sqlite3.connect(db).execute(
        "SELECT content FROM standalone_posts WHERE id IN (?, ?)", (rows[0][0], rows[1][0])).fetchall()
    check(contents == [("Passkeys, rewritten.",)] * 2, "the card split: only some rows were edited")
    for post_id, needle in ((queued, "queued"), (published, "published")):
        bad, text, _ = server.call("update_draft", post_id=post_id, content="changed")
        check(bad and needle in text, f"edited a {needle} post: {text}")
    bad, text, _ = server.call("update_draft", post_id=first, content="a " + EN + " b")
    check(bad and "en dash" in text, "update_draft took an en dash")

    error = server.request("tools/call", {"name": "publish_now", "arguments": {}})
    check(error.get("error", {}).get("code") == -32602, f"unknown tool: {error}")
    error = server.request("resources/list")
    check(error.get("error", {}).get("code") == -32601, f"unknown method: {error}")
    server.send({"jsonrpc": "2.0", "id": 900, "method": "tools/call",
                 "params": {"name": "list_posts", "arguments": {"limit": "lots"}}})
    reply = server.read()
    check(reply["result"]["isError"] and "whole number" in reply["result"]["content"][0]["text"], "bad argument")
    server.proc.stdin.write("this is not json\n")
    server.proc.stdin.flush()
    check(server.read()["error"]["code"] == -32700, "a parse error is not reported")
    server.close()
    for line in server.lines:
        message = json.loads(line)
        check(message.get("jsonrpc") == "2.0" and ("result" in message or "error" in message),
              f"stdout carried a non-protocol line: {line[:200]}")

    missing = subprocess.run([PY, os.path.join(ROOT, "insights_mcp.py"), "--db", os.path.join(directory, "nope.db")],
                             capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    check(missing.returncode != 0 and missing.stdout == "" and "no database" in missing.stderr,
          f"a missing database: {missing.returncode} {missing.stderr[-200:]}")
    check(not os.path.exists(os.path.join(directory, "nope.db")), "a missing database was created")
    print("mcp: JSON-RPC only on stdout; six tools, none publishes; drafts as cards; batch refused whole; "
          "queued and published untouchable; sources paged; missing database refused")
    print("SUB_MCP_OK")


# ── docs, registration, dashes ───────────────────────────────────────────

def section_docs():
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    for needle in ("LLM_PROVIDER=claude-code", "LLM_PROVIDER=codex", "My subscription",
                   "claude mcp add", "insights_mcp.py", "[mcp_servers.insights]", "Claude Desktop",
                   "claude_desktop_config.json", "SUBSCRIPTION_LLM_TIMEOUT", "CLAUDE_CODE_OAUTH_TOKEN",
                   "CODEX_CLI_PATH", "web search", "personal", "save_drafts"):
        check(needle in readme, f"README does not mention {needle!r}")
    env_example = open(os.path.join(ROOT, ".env.example"), encoding="utf-8").read()
    for needle in ("claude-code", "CODEX_MODEL", "SUBSCRIPTION_LLM_TIMEOUT", "CLAUDE_CODE_OAUTH_TOKEN"):
        check(needle in env_example, f".env.example does not mention {needle!r}")
    example = os.path.join(ROOT, ".mcp.json.example")
    check(os.path.exists(example), ".mcp.json.example is missing")
    data = json.load(open(example))
    check("insights" in data.get("mcpServers", {}) and "insights_mcp.py" in json.dumps(data),
          ".mcp.json.example does not configure the server")
    print("docs: both ways in, the settings, what stays on the API, personal use")
    print("SUB_DOCS_OK")


SECTIONS_TOKENS = {
    "claude": "SUB_CLAUDE_OK", "codex": "SUB_CODEX_OK", "routing": "SUB_ROUTING_OK",
    "status": "SUB_STATUS_OK", "ui": "SUB_UI_OK", "mcp": "SUB_MCP_OK", "docs": "SUB_DOCS_OK",
    "registered": "SUB_REGISTERED_OK", "nodash": "SUB_NODASH_OK",
}


def section_registered():
    import run_all_gates
    entries = [g for g in run_all_gates.ALL_GATES if g[0] == "check_subscription_llm.py"]
    have = {(g[3][0] if len(g) > 3 else None): g[1] for g in entries}
    for section, token in SECTIONS_TOKENS.items():
        check(have.get(section) == token, f"section {section} is not registered under {token}")
    print("registered: every section is in run_all_gates.py")
    print("SUB_REGISTERED_OK")


NEW_FILES = ("subscription_llm.py", "insights_mcp.py", "scripts/check_subscription_llm.py",
             "scripts/live_subscription.py", ".mcp.json.example")
TOUCHED = ("insights.py", "insights_web.py", "templates/compose.html", "templates/partials/post_item.html",
           "README.md", "scripts/run_all_gates.py", ".env.example")


def section_nodash():
    dash = EM
    for name in NEW_FILES:
        path = os.path.join(ROOT, name)
        check(os.path.exists(path), f"{name} is missing")
        lines = [i for i, line in enumerate(open(path, encoding="utf-8"), 1) if dash in line]
        check(not lines, f"{name} has an em dash on lines {lines}")
    diff = subprocess.run(["git", "diff", "-U0", "HEAD", "--", *TOUCHED], cwd=ROOT,
                          capture_output=True, text=True).stdout
    added = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++") and dash in l]
    check(not added, f"an added line has an em dash: {added[:3]}")
    # Control: the same scan finds one when there is one.
    control = "+x " + EM + " y"
    check(dash in control and control.startswith("+"), "the scan cannot see a dash")
    print(f"nodash: {len(NEW_FILES)} new files and the added lines of {len(TOUCHED)} touched files are clean")
    print("SUB_NODASH_OK")


SECTIONS = {
    "claude": section_claude, "codex": section_codex, "routing": section_routing,
    "status": section_status, "ui": section_ui, "mcp": section_mcp, "docs": section_docs,
    "registered": section_registered, "nodash": section_nodash,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SECTIONS:
        print(f"usage: {sys.argv[0]} {{{'|'.join(SECTIONS)}}}")
        sys.exit(2)
    SECTIONS[sys.argv[1]]()
