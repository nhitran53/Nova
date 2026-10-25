"""HTTP chat generator: the `Generator` for a local OpenAI-compatible server (stdlib only, localhost-friendly).

    config = load_live_llm_config(base_url="http://localhost:8080/v1")    # turn_policy.json["live_llm"] < env < arguments
    ok, message = check_server(config["base_url"], config["health_timeout_s"])
    handler = TurnHandler(router, alert_source, registry, generator=HTTPChatGenerator(config))

One POST to `<base_url>/chat/completions` per `generate(req)`. The messages are `render_prompt(req)`; `max_tokens` is
`req.style.max_tokens`; `seed` is `req.seed` (not sent to mlx). The adapter returns the model's message content as
`GenerationResult.raw` and nothing else: it does NOT parse, validate or fix numbers (`postprocess` / `run_generation` do,
including retry and the fallback line). Any HTTP error, refused connection, timeout or malformed response RAISES; run_generation
turns a raising generator into the `error` outcome (tool_error line, no retry). Error messages carry the HTTP status and reason
only, never the server's response body, and the API key / any `Bearer ...` token is redacted from every message.

The request timeout is min(live_llm.request_timeout_cap_s, the remaining req.deadline_ms) in seconds.

`runtime` only changes the extra body fields (all PLACEHOLDER, none verified against a real server):
  llama-server : chat_template_kwargs {enable_thinking: false}; constrain "json_schema" -> `json_schema` = output_json_schema(req),
                 "gbnf" -> `grammar` = output_gbnf(req)
  ollama       : reasoning_effort "none"; constrain "json_schema" -> response_format {type: json_schema, ...}   (no GBNF)
  openai       : constrain "json_schema" -> response_format; no thinking field
  mlx          : none of them (mlx_lm.server has no constrained decoding; thinking-off is a LAUNCH flag, e.g.
                 --chat-template-args '{"enable_thinking":false}'; no `seed`). `constrain` is ignored for mlx.
`live_llm.extra_body` is merged last and wins over the fields above. `disable_thinking: false` omits the thinking-off field.

The model's `<think>...</think>` text is stripped and `reasoning_content` is never read, so reasoning can not reach `postprocess`
or the speaker. Only 127.0.0.1-style URLs are expected; proxies from the environment are ignored on purpose.
"""
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from pathlib import Path

from .generation import GenerationResult, output_gbnf, output_json_schema, render_prompt

HERE = Path(__file__).parent
POLICY_PATH = HERE / "turn_policy.json"
RUNTIMES = ("llama-server", "mlx", "ollama", "openai")
CONSTRAIN_MODES = ("none", "json_schema", "gbnf")
ENV_OVERRIDES = {"runtime": "NOVA_LLM_RUNTIME", "base_url": "NOVA_LLM_BASE_URL", "model": "NOVA_LLM_MODEL",
                 "constrain": "NOVA_LLM_CONSTRAIN"}
API_KEY_ENV = "NOVA_LLM_API_KEY"  # optional bearer token (hosted "openai" runtime); read from the environment only
CONFIG_KEYS = ("runtime", "base_url", "model", "request_timeout_cap_s", "health_timeout_s", "temperature", "constrain",
               "disable_thinking", "extra_body")
MAX_RESPONSE_BYTES = 1_000_000
BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+\S+")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # a proxy in the environment must never intercept localhost


def redact_secrets(text, api_key=None):
    """`text` with the API key (also URL-encoded) and any `Bearer <token>` removed; error text must never carry credentials."""
    text = BEARER_TOKEN.sub("Bearer [redacted]", text)
    for secret in {api_key, urllib.parse.quote(api_key, safe="")} if api_key else ():
        text = text.replace(secret, "[redacted]")
    return text


class LLMRequestError(RuntimeError):
    """The server could not give a usable reply (HTTP error, connection failure, malformed response)."""


class LLMTimeoutError(LLMRequestError, TimeoutError):
    """The request ran past min(request_timeout_cap_s, the request's remaining deadline)."""


def _positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def validate_live_llm_config(config, where="live_llm"):
    """Raise ValueError (clear message) for a config the adapter cannot trust. Returns the config.
    Messages name the invalid key, never its supplied value: a mistyped value can be a secret (e.g. the API key pasted into base_url)."""
    if not isinstance(config, dict):
        raise ValueError(f"{where} must be an object")
    missing = [k for k in CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"{where} is missing {missing}")
    if config["runtime"] not in RUNTIMES:
        raise ValueError(f"{where}.runtime must be one of {RUNTIMES}, not the supplied value")
    if config["constrain"] not in CONSTRAIN_MODES:
        raise ValueError(f"{where}.constrain must be one of {CONSTRAIN_MODES}, not the supplied value")
    if config["constrain"] == "gbnf" and config["runtime"] != "llama-server":
        raise ValueError(f"{where}.constrain 'gbnf' needs runtime 'llama-server' (only llama.cpp takes a grammar)")
    url = config["base_url"]
    if not isinstance(url, str) or not re.match(r"^https?://[^\s/]+", url):
        raise ValueError(f"{where}.base_url must be an http(s) URL (the supplied value is not shown)")
    if not isinstance(config["model"], str) or not config["model"].strip():
        raise ValueError(f"{where}.model must be a non-empty string")
    for key in ("request_timeout_cap_s", "health_timeout_s"):
        if not _positive_number(config[key]):
            raise ValueError(f"{where}.{key} must be a number > 0")
    if isinstance(config["temperature"], bool) or not isinstance(config["temperature"], (int, float)) or config["temperature"] < 0:
        raise ValueError(f"{where}.temperature must be a number >= 0")
    if not isinstance(config["disable_thinking"], bool):
        raise ValueError(f"{where}.disable_thinking must be true or false")
    if not isinstance(config["extra_body"], dict):
        raise ValueError(f"{where}.extra_body must be an object")
    try:
        json.dumps(config["extra_body"])
    except (TypeError, ValueError):
        raise ValueError(f"{where}.extra_body must be JSON-serializable") from None
    return config


def load_live_llm_config(path=POLICY_PATH, env=None, **overrides):
    """turn_policy.json["live_llm"], then NOVA_LLM_* from `env` (default os.environ), then non-None keyword `overrides`
    (CLI flags win). Validated. Returns a new dict."""
    config = deepcopy(json.loads(Path(path).read_text())["live_llm"])
    config.pop("status", None)
    env = os.environ if env is None else env
    for key, name in ENV_OVERRIDES.items():
        if env.get(name):
            config[key] = env[name]
    for key, value in overrides.items():
        if key not in CONFIG_KEYS:
            raise ValueError(f"unknown live_llm override {key!r}")
        if value is not None:
            config[key] = value
    return validate_live_llm_config(config)


def api_base(base_url):
    """-> base_url ending in /v1 with no trailing slash ("http://localhost:8080" and ".../v1/" both work)."""
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def strip_reasoning(content):
    """Remove `<think>...</think>` blocks from a model reply. A lone `</think>` (the chat template opened the block) keeps only
    the text after it; an unterminated `<think>` (cut off by max_tokens) drops everything from it on."""
    text = THINK_BLOCK.sub("", content)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


class HTTPChatGenerator:
    """`Generator` over an OpenAI-compatible /v1/chat/completions server. See the module docstring."""

    def __init__(self, config=None, api_key=None, opener=_OPENER):
        self.config = validate_live_llm_config(load_live_llm_config() if config is None else config)
        self.api_key = os.environ.get(API_KEY_ENV) if api_key is None else api_key
        self.opener = opener
        self.url = api_base(self.config["base_url"]) + "/chat/completions"
        self.calls = 0
        self.last_body = None  # the request body of the latest call (for logs and tests)

    @property
    def effective_constrain(self):
        return "none" if self.config["runtime"] == "mlx" else self.config["constrain"]

    def describe(self):
        c = self.config
        description = (f"{c['runtime']} model={c['model']} url={self.url} constrain={self.effective_constrain} "
                       f"thinking_off={'launch-time (mlx)' if c['runtime'] == 'mlx' else c['disable_thinking']}")
        return redact_secrets(description, self.api_key)

    def build_body(self, req):
        c, runtime, constrain = self.config, self.config["runtime"], self.effective_constrain
        body = {"model": c["model"], "messages": render_prompt(req), "max_tokens": req.style.max_tokens,
                "temperature": c["temperature"], "stream": False}
        if req.seed is not None and runtime != "mlx":
            body["seed"] = req.seed
        if runtime == "llama-server":
            if c["disable_thinking"]:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            if constrain == "json_schema":
                body["json_schema"] = output_json_schema(req)
            elif constrain == "gbnf":
                body["grammar"] = output_gbnf(req)
        else:
            if runtime == "ollama" and c["disable_thinking"]:
                body["reasoning_effort"] = "none"
            if constrain == "json_schema":
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "nova_reply", "schema": output_json_schema(req)}}
        body.update(deepcopy(c["extra_body"]))
        return body

    def request_timeout_s(self, req):
        """min(request_timeout_cap_s, remaining req.deadline_ms) in seconds; a spent deadline raises LLMTimeoutError."""
        remaining = req.deadline_ms / 1000.0
        if remaining <= 0:
            raise LLMTimeoutError(f"no time left for the request (deadline_ms={req.deadline_ms})")
        return min(self.config["request_timeout_cap_s"], remaining)

    def generate(self, req):
        try:
            return self._generate(req)
        except LLMRequestError as exc:
            error_type, message = type(exc), redact_secrets(str(exc), self.api_key)
        # raised outside the `except` block: a raise inside it would keep the original (with its secret-bearing cause) as __context__
        raise error_type(message) from None

    def _generate(self, req):
        timeout = self.request_timeout_s(req)
        body = self.last_body = self.build_body(req)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        self.calls += 1
        started = time.monotonic()
        try:
            with self.opener.open(http_request, timeout=timeout) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise LLMRequestError(f"HTTP {exc.code} {exc.reason} from {self.url}") from None  # the response body is never included
        except (TimeoutError, OSError, http.client.HTTPException) as exc:  # URLError, ConnectionError are OSError; socket.timeout is TimeoutError
            reason = getattr(exc, "reason", exc)
            if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                raise LLMTimeoutError(f"no reply from {self.url} within {timeout:.2f} s") from exc
            raise LLMRequestError(f"cannot reach {self.url}: {reason}") from exc
        latency_ms = int((time.monotonic() - started) * 1000)
        return GenerationResult(raw=self._content(payload), latency_ms=latency_ms, ttft_ms=None)

    @staticmethod
    def _content(payload):
        if len(payload) > MAX_RESPONSE_BYTES:
            raise LLMRequestError(f"response is over {MAX_RESPONSE_BYTES} bytes")
        try:
            content = json.loads(payload)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(f"malformed chat completion: {type(exc).__name__}: {exc}"[:200]) from exc
        if not isinstance(content, str):
            raise LLMRequestError(f"malformed chat completion: message content is {type(content).__name__}, not a string")
        return strip_reasoning(content)  # reasoning_content is never read


def _get(url, timeout):
    with _OPENER.open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
        return response.status, response.read(MAX_RESPONSE_BYTES)


def check_server(base_url, timeout=1.5, api_key=None):
    """-> (ok, message), with the API key (default: NOVA_LLM_API_KEY) and any `Bearer ...` redacted. GET <root>/health (llama-server), then GET <base_url>/models (all runtimes). Short timeout, never raises.
    A llama-server that answers /health with 503 is still loading its model and counts as not ready."""
    ok, message = _check_server(base_url, timeout)
    return ok, redact_secrets(message, os.environ.get(API_KEY_ENV) if api_key is None else api_key)


def _check_server(base_url, timeout):
    base = api_base(base_url)
    root = base[:-len("/v1")]
    problems = []
    for url in (root + "/health", base + "/models"):
        try:
            status, payload = _get(url, timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                return False, f"server at {base} is up but still loading the model (HTTP 503 from {url})"
            problems.append(f"HTTP {exc.code} from {url}")
            continue
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", exc)
            return False, f"no LLM server reachable at {base} ({reason})"
        if url.endswith("/models"):
            try:
                names = [m["id"] for m in json.loads(payload)["data"]][:3]
            except (ValueError, KeyError, TypeError):
                names = []
            return True, f"server reachable at {base}" + (f" (models: {', '.join(names)})" if names else "")
        return True, f"server reachable at {base} (/health ok)"
    return False, f"server at {base} did not look like an OpenAI-compatible server ({'; '.join(problems)})"
