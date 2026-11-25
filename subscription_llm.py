"""Run the app's model calls on a Claude or ChatGPT subscription instead of an API key.

Every generation feature in Insights asks ``insights._get_llm_client`` for a client
and then calls ``client.chat.completions.create(model=..., messages=...)``. This
module supplies two more clients with that same surface:

* ``claude-code`` runs the Claude Code CLI (``claude -p``) signed in with a
  Claude Pro or Max plan;
* ``codex`` runs the Codex CLI (``codex exec``) signed in with a ChatGPT plan.

The calls are then counted against the plan's usage limits, not billed to
``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``. Three things make sure of that and
keep the CLIs to plain text generation:

1. **The API keys never reach the CLI.** The child process gets a short allowlist
   of environment variables (home, path, locale, proxy, the CLIs' own config
   dirs and ``CLAUDE_CODE_OAUTH_TOKEN``), so a key the app loaded from ``.env``
   cannot be picked up and billed. ``claude -p`` prefers ``ANTHROPIC_API_KEY``
   over the signed-in plan when both exist, which is exactly what this prevents.
2. **No tools.** Claude runs with ``--tools ""``, no MCP servers and no settings
   files (so no hooks, plugins or CLAUDE.md). Codex runs with the user's config
   ignored, a read-only sandbox, web search off and its shell, browser,
   computer-use, plugin and patch tools disabled. Prompts carry untrusted text
   (transcripts, web pages, inbound messages), so the model is given nothing it
   could be talked into running.
3. **A fresh empty working directory** per call, deleted afterwards, with
   session persistence off.

A failure says what went wrong in plain words (not signed in, usage limit,
unknown model, timed out) and never falls back to the paid API on its own.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import types
import urllib.request

logger = logging.getLogger(__name__)

CLAUDE_CODE = "claude-code"
CODEX = "codex"
PROVIDERS = (CLAUDE_CODE, CODEX)
LABELS = {CLAUDE_CODE: "Claude Code", CODEX: "Codex"}
PLAN_NAMES = {CLAUDE_CODE: "Claude plan", CODEX: "ChatGPT plan"}

# Models a ChatGPT plan accepts through Codex. Checked live on 2026-11-25: the
# API-only names (gpt-6.1-sol, gpt-6-luna) are refused with "not supported when
# using Codex with a ChatGPT account". The first one is Codex's own default.
CODEX_MODELS = ["gpt-6-astra", "gpt-5.6-sol"]
# Short names the Claude Code CLI resolves itself.
CLAUDE_ALIASES = ("opus", "sonnet", "haiku", "fable", "default")

DEFAULT_SYSTEM = (
    "You are a writing assistant inside an application. Follow the user's "
    "instructions exactly and reply with only the requested output."
)
# Codex is an agent: say up front that this is a plain text task.
CODEX_PREAMBLE = (
    "You are being used as a plain text-generation model by an application. "
    "Do not run commands, read or write files, browse, or use any tool. "
    "Answer from the text below and reply with only the requested output."
)

# Codex features that give the model a way to act rather than write. Disabling
# a feature that a later Codex release removes is harmless; one it adds is not
# covered, which is why the sandbox is read-only and the config is ignored too.
CODEX_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "code_mode_host", "computer_use", "browser_use",
    "browser_use_external", "in_app_browser", "plugins", "apps", "multi_agent",
    "image_generation", "hooks", "goals", "skill_search", "tool_suggest",
    "sleep_tool", "view_image", "workspace_dependencies", "shell_snapshot",
)

# Variables the CLIs need to find their own sign-in and reach the network.
# Everything else in the app's environment (API keys, platform tokens, Twilio,
# Cloudinary ...) stays out of the child process.
_ENV_ALLOW = {
    "HOME", "PATH", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TZ", "TERM",
    "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR", "CODEX_HOME",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
    "ALL_PROXY", "all_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS",
    "REQUESTS_CA_BUNDLE", "SYSTEMROOT", "APPDATA", "LOCALAPPDATA", "USERPROFILE",
}
_ENV_ALLOW_PREFIXES = ("LC_", "XDG_")

_CANDIDATES = {
    CLAUDE_CODE: ("CLAUDE_CLI_PATH", "claude", (
        "~/.local/bin/claude", "~/.claude/local/claude",
        "/opt/homebrew/bin/claude", "/usr/local/bin/claude",
    )),
    CODEX: ("CODEX_CLI_PATH", "codex", (
        "/Applications/ChatGPT.app/Contents/Resources/codex",
        "/Applications/Codex.app/Contents/Resources/codex",
        "~/.local/bin/codex", "/opt/homebrew/bin/codex", "/usr/local/bin/codex",
    )),
}

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_STATUS_TTL = 60.0
_CLI_TTL = 300.0


class SubscriptionLLMError(RuntimeError):
    """A subscription CLI call failed; the message is meant for the user."""


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def timeout_seconds() -> int:
    return _env_int("SUBSCRIPTION_LLM_TIMEOUT", 600)


def effort() -> str:
    return os.environ.get("SUBSCRIPTION_EFFORT", "low").strip().lower() or "low"


# One CLI process is a whole agent runtime (hundreds of MB), so a burst of
# background work is queued rather than started all at once.
_slots = threading.BoundedSemaphore(_env_int("SUBSCRIPTION_LLM_CONCURRENCY", 2))


def child_env() -> dict:
    """The environment a CLI runs with: the allowlist above, nothing else."""
    return {k: v for k, v in os.environ.items()
            if k in _ENV_ALLOW or k.startswith(_ENV_ALLOW_PREFIXES)}


# ── finding the CLIs ─────────────────────────────────────────────────────

_cli_cache: dict = {}
_cli_lock = threading.Lock()


def _probe(path: str) -> bool:
    try:
        done = subprocess.run([path, "--version"], capture_output=True, text=True,
                              timeout=20, env=child_env(), stdin=subprocess.DEVNULL)
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def find_cli(provider: str) -> str:
    """Return the path of a working CLI for ``provider`` or raise.

    ``CLAUDE_CLI_PATH`` / ``CODEX_CLI_PATH`` win when set. Otherwise the PATH
    entry and the usual install locations are tried in turn, and the first that
    answers ``--version`` is used: an npm-installed ``codex`` can be a broken
    shim while the copy inside ChatGPT.app works.
    """
    env_name, command, extra = _CANDIDATES[provider]
    override = os.environ.get(env_name, "").strip()
    if override:
        path = os.path.expanduser(override)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        raise SubscriptionLLMError(
            f"{env_name} points at {path}, which is not an executable file.")

    with _cli_lock:
        hit = _cli_cache.get(provider)
        if hit and time.monotonic() - hit[1] < _CLI_TTL:
            return hit[0]
    seen = []
    for candidate in [shutil.which(command), *extra]:
        if not candidate:
            continue
        path = os.path.realpath(os.path.expanduser(candidate))
        if path in seen or not (os.path.isfile(path) and os.access(path, os.X_OK)):
            continue
        seen.append(path)
        if _probe(path):
            with _cli_lock:
                _cli_cache[provider] = (path, time.monotonic())
            return path
    hint = ("install Claude Code (https://claude.com/claude-code) and run `claude` once to sign in"
            if provider == CLAUDE_CODE else
            "install Codex (`npm i -g @openai/codex`, or the ChatGPT desktop app) and run `codex login`")
    raise SubscriptionLLMError(
        f"{LABELS[provider]} was not found on this computer: {hint}, "
        f"or set {env_name} to its path.")


# ── turning chat messages into one prompt ────────────────────────────────

def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def _images_of(content) -> list:
    if isinstance(content, str):
        return []
    return [(b.get("image_url") or {}).get("url", "") for b in content or []
            if b.get("type") == "image_url" and (b.get("image_url") or {}).get("url")]


def split_messages(messages) -> tuple:
    """Return ``(system_text, prompt_text, image_urls)`` for OpenAI-style messages.

    A single user message becomes the prompt as written. A longer exchange is
    replayed as a labelled transcript ending on the last user message, which is
    how a one-shot CLI call can answer a multi-turn conversation.
    """
    system, turns, images = [], [], []
    for message in messages or []:
        role = message.get("role")
        content = message.get("content", "")
        if role in ("system", "developer"):
            text = _text_of(content)
            if text:
                system.append(text)
            continue
        turns.append((role, _text_of(content)))
        images.extend(_images_of(content))

    if len(turns) == 1 and turns[0][0] == "user":
        prompt = turns[0][1]
    else:
        last_user = max((i for i, (r, _) in enumerate(turns) if r == "user"), default=None)
        earlier = turns if last_user is None else turns[:last_user]
        names = {"user": "User", "assistant": "Assistant"}
        lines = [f"{names.get(r, r or 'User')}: {t}" for r, t in earlier]
        parts = []
        if lines:
            parts.append("The conversation so far:\n\n" + "\n\n".join(lines))
        if last_user is not None:
            parts.append("Now reply to this latest message from the user:\n\n" + turns[last_user][1])
        prompt = "\n\n".join(parts)
    return "\n\n".join(system), prompt, images


def _decode_data_url(url: str) -> tuple:
    header, _, data = url.partition(",")
    media_type = header[len("data:"):].split(";")[0] or "image/png"
    return media_type, data


def _fetch_image(url: str) -> tuple:
    """Download an http(s) image for Codex, which only takes files."""
    request = urllib.request.Request(url, headers={"User-Agent": "Insights/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(_MAX_IMAGE_BYTES + 1)
        media_type = (response.headers.get_content_type() or "image/png")
    if len(body) > _MAX_IMAGE_BYTES:
        raise SubscriptionLLMError("An image for this request is larger than 20 MB.")
    return media_type, body


_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
               "image/gif": ".gif", "image/webp": ".webp"}


# ── running a CLI ────────────────────────────────────────────────────────

def _kill(process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, AttributeError):
        try:
            process.kill()
        except OSError:
            pass


def run_cli(provider: str, args: list, stdin_text: str, cwd: str) -> tuple:
    """Run one CLI call and return ``(returncode, stdout, stderr)``."""
    limit = timeout_seconds()
    if not _slots.acquire(timeout=limit):
        raise SubscriptionLLMError(
            f"{LABELS[provider]} is busy with other requests; try again in a moment.")
    try:
        try:
            process = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=cwd, env=child_env(),
                text=True, encoding="utf-8", errors="replace",
                start_new_session=True)
        except OSError as exc:
            raise SubscriptionLLMError(f"Could not start {LABELS[provider]}: {exc}") from exc
        try:
            out, err = process.communicate(stdin_text, timeout=limit)
        except subprocess.TimeoutExpired:
            _kill(process)
            process.communicate()
            raise SubscriptionLLMError(
                f"{LABELS[provider]} took longer than {limit} seconds and was stopped. "
                "Set SUBSCRIPTION_LLM_TIMEOUT to allow longer.")
        return process.returncode, out or "", err or ""
    finally:
        _slots.release()


def _tail(text: str, size: int = 400) -> str:
    text = (text or "").strip()
    return text[-size:]


def _response(text: str, model: str, input_tokens: int, output_tokens: int):
    usage = types.SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output_tokens,
        prompt_tokens=input_tokens, completion_tokens=output_tokens)
    message = types.SimpleNamespace(content=text, role="assistant")
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
        usage=usage, model=model)


class _SubscriptionClient:
    """``client.chat.completions.create(...)`` over a CLI; one subclass per CLI."""

    provider = ""

    def __init__(self):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **_ignored):
        # temperature / max_tokens / response_format have no CLI equivalent;
        # every caller already parses free text (and JSON out of fences).
        system, prompt, images = split_messages(messages)
        workdir = tempfile.mkdtemp(prefix=f"insights_{self.provider}_")
        try:
            return self._call(model, system, prompt, images, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _uses_effort(model: str) -> bool:
    import insights  # late: insights imports this module lazily
    return bool(insights._CLAUDE_THINKS_BY_DEFAULT.match(model or "")) or model in ("opus", "sonnet", "fable")


class ClaudeCodeClient(_SubscriptionClient):
    provider = CLAUDE_CODE

    def _call(self, model, system, prompt, images, workdir):
        cli = find_cli(CLAUDE_CODE)
        args = [cli, "-p",
                "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--tools", "", "--strict-mcp-config", "--setting-sources", "",
                "--no-session-persistence",
                # "=" form, so a prompt that starts with "-" is never read as a flag
                "--system-prompt=" + (system or DEFAULT_SYSTEM),
                "--model=" + model]
        if _uses_effort(model):
            args += ["--effort", effort()]

        content = [{"type": "text", "text": prompt}]
        for url in images:
            if url.startswith("data:"):
                media_type, data = _decode_data_url(url)
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": data}})
            else:
                content.append({"type": "image", "source": {"type": "url", "url": url}})
        line = json.dumps({"type": "user", "message": {"role": "user", "content": content}})

        code, out, err = run_cli(CLAUDE_CODE, args, line + "\n", workdir)
        result = None
        for raw in out.splitlines():
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                result = event
        if result is None:
            raise SubscriptionLLMError(
                f"Claude Code stopped without an answer (exit {code}): "
                f"{_tail(err) or _tail(out) or 'no output'}")
        text = result.get("result") or ""
        if result.get("is_error") or result.get("subtype") not in (None, "success"):
            raise SubscriptionLLMError(_claude_error(text, result))
        if result.get("stop_reason") == "refusal":
            from insights import LLMRefusal
            raise LLMRefusal("The model declined this request. "
                             "Try rephrasing it or choosing a different model.")
        usage = result.get("usage") or {}
        tokens_in = sum(int(usage.get(k) or 0) for k in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        used_model = next(iter(result.get("modelUsage") or {}), None) or model
        return _response(text, used_model, tokens_in, int(usage.get("output_tokens") or 0))


def _claude_error(text: str, result: dict) -> str:
    lowered = (text or "").lower()
    if "login" in lowered or "log in" in lowered or "not logged" in lowered \
            or "authentication" in lowered or result.get("api_error_status") == 401:
        return ("Claude Code is not signed in. Run `claude` in a terminal and sign in "
                "with your Claude account, then try again.")
    if "limit" in lowered and ("usage" in lowered or "reset" in lowered):
        return f"Your Claude plan's usage limit was reached: {text}"
    return f"Claude Code could not answer: {text or result.get('terminal_reason') or 'unknown error'}"


class CodexClient(_SubscriptionClient):
    provider = CODEX

    def _call(self, model, system, prompt, images, workdir):
        cli = find_cli(CODEX)
        args = [cli, "exec"]
        for index, url in enumerate(images):
            if url.startswith("data:"):
                media_type, data = _decode_data_url(url)
                body = base64.b64decode(data)
            else:
                media_type, body = _fetch_image(url)
            path = os.path.join(workdir, f"image{index}{_EXTENSIONS.get(media_type, '.png')}")
            with open(path, "wb") as handle:
                handle.write(body)
            args += ["--image", path]
        last = os.path.join(workdir, "last_message.txt")
        args += ["--json", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
                 "--ignore-user-config", "--sandbox", "read-only", "--color", "never",
                 "--cd", workdir, "--output-last-message", last,
                 "-c", 'web_search="disabled"',
                 "-c", f'model_reasoning_effort="{effort()}"']
        for feature in CODEX_DISABLED_FEATURES:
            args += ["--disable", feature]
        if model:
            args += ["--model", model]
        args.append("-")

        parts = [CODEX_PREAMBLE]
        if system:
            parts.append("<instructions>\n" + system + "\n</instructions>")
        parts.append(prompt)
        code, out, err = run_cli(CODEX, args, "\n\n".join(parts), workdir)

        messages, failure, usage = [], None, {}
        for raw in out.splitlines():
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            kind = event.get("type") if isinstance(event, dict) else None
            if kind == "item.completed" and (event.get("item") or {}).get("type") == "agent_message":
                messages.append(event["item"].get("text") or "")
            elif kind == "turn.completed":
                usage = event.get("usage") or {}
            elif kind == "turn.failed":
                failure = (event.get("error") or {}).get("message") or "the turn failed"
            elif kind == "error" and not failure:
                failure = event.get("message") or "unknown error"
        if failure:
            raise SubscriptionLLMError(_codex_error(failure))
        text = ""
        if os.path.exists(last):
            with open(last, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        text = text.strip() or (messages[-1] if messages else "")
        if code != 0 or not text:
            raise SubscriptionLLMError(
                f"Codex stopped without an answer (exit {code}): "
                f"{_tail(err) or 'no output'}")
        return _response(text, model or CODEX_MODELS[0],
                         int(usage.get("input_tokens") or 0),
                         int(usage.get("output_tokens") or 0))


def _codex_error(message: str) -> str:
    detail = message
    try:  # the API's error arrives as a JSON string inside the event
        detail = json.loads(message).get("error", {}).get("message") or message
    except (ValueError, AttributeError):
        pass
    lowered = detail.lower()
    if "login" in lowered or "unauthorized" in lowered or "401" in lowered:
        return "Codex is not signed in. Run `codex login` in a terminal, then try again."
    if "usage limit" in lowered or "rate limit" in lowered:
        return f"Your ChatGPT plan's usage limit was reached: {detail}"
    return f"Codex could not answer: {detail}"


def client_for(provider: str):
    return ClaudeCodeClient() if provider == CLAUDE_CODE else CodexClient()


# ── is it set up? ────────────────────────────────────────────────────────

_status_cache: dict = {}


def _status_uncached(provider: str) -> dict:
    info = {"provider": provider, "label": LABELS[provider], "plan": PLAN_NAMES[provider],
            "available": False, "signed_in": False, "subscription": False, "detail": ""}
    try:
        cli = find_cli(provider)
    except SubscriptionLLMError as exc:
        info["detail"] = str(exc)
        return info
    info["available"] = True
    try:
        if provider == CLAUDE_CODE:
            done = subprocess.run([cli, "auth", "status"], capture_output=True, text=True,
                                  timeout=20, env=child_env(), stdin=subprocess.DEVNULL)
            data = json.loads(done.stdout or "{}")
            info["signed_in"] = bool(data.get("loggedIn"))
            method = str(data.get("authMethod") or "")
            info["subscription"] = info["signed_in"] and method in (
                "claude.ai", "claude_ai", "oauth_token", "oauth")
            if not info["signed_in"]:
                info["detail"] = "Not signed in: run `claude` in a terminal and sign in."
            elif not info["subscription"]:
                info["detail"] = (f"Signed in with {method or 'an API key'}, so calls are billed "
                                  "to that key. Sign in with your Claude account to use the plan.")
            else:
                info["detail"] = "Signed in with your Claude plan."
        else:
            done = subprocess.run([cli, "login", "status"], capture_output=True, text=True,
                                  timeout=20, env=child_env(), stdin=subprocess.DEVNULL)
            said = ((done.stdout or "") + (done.stderr or "")).strip()
            lowered = said.lower()
            info["signed_in"] = done.returncode == 0 and "logged in" in lowered and "not logged" not in lowered
            info["subscription"] = info["signed_in"] and "chatgpt" in lowered
            if not info["signed_in"]:
                info["detail"] = "Not signed in: run `codex login` in a terminal."
            elif not info["subscription"]:
                info["detail"] = ("Signed in with an API key, so calls are billed to it. "
                                  "Run `codex login` and choose ChatGPT to use the plan.")
            else:
                info["detail"] = "Signed in with your ChatGPT plan."
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        info["detail"] = f"Could not check the sign-in: {exc}"
    return info


def status(provider: str, refresh: bool = False) -> dict:
    hit = _status_cache.get(provider)
    if hit and not refresh and time.monotonic() - hit[1] < _STATUS_TTL:
        return dict(hit[0])
    info = _status_uncached(provider)
    _status_cache[provider] = (info, time.monotonic())
    return dict(info)


def all_status(refresh: bool = False) -> dict:
    """Both CLIs' status, checked side by side (each check is a process start)."""
    results = {}
    threads = [threading.Thread(target=lambda p=p: results.__setitem__(p, status(p, refresh)))
               for p in PROVIDERS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return {p: results[p] for p in PROVIDERS}
