#!/usr/bin/env python3
"""Generate images through a local grok2api OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "grok-4.20-fast"
IMAGE_TOOL_NAME = "generate_photo"
IMAGE_TOOL_MODEL = "grok-imagine-image-lite"
EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "q"}
DIRECT_IMAGE_COMMANDS = ("/image", "/img", "/photo", "/照片", "/生图")
IMAGE_INTENT_KEYWORDS = (
    "图片",
    "照片",
    "图像",
    "画一",
    "画个",
    "画张",
    "生成",
    "生图",
    "制作",
    "设计",
    "海报",
    "头像",
    "壁纸",
    "photo",
    "image",
    "picture",
    "draw",
    "render",
    "generate",
    "create",
)
MODEL_CHOICES = [
    {
        "key": "4.20-fast",
        "model": "grok-4.20-fast",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "chat-completions image path",
    },
    {
        "key": "4.3-fast",
        "model": "grok-4.3-fast",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "chat-completions image path",
    },
    {
        "key": "4.3-console",
        "model": "grok-4.3-console",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "user reasoning, default medium; chat-completions image path",
    },
    {
        "key": "4.3-low",
        "model": "grok-4.3-low",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "fixed low reasoning; chat-completions image path",
    },
    {
        "key": "4.3-medium",
        "model": "grok-4.3-medium",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "fixed medium reasoning; chat-completions image path",
    },
    {
        "key": "4.3-high",
        "model": "grok-4.3-high",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "fixed high reasoning; chat-completions image path",
    },
    {
        "key": "4.20-0309-console",
        "model": "grok-4.20-0309-console",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "default reasoning; chat-completions image path",
    },
    {
        "key": "4.20-0309-reasoning",
        "model": "grok-4.20-0309-reasoning-console",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "fixed reasoning; chat-completions image path",
    },
    {
        "key": "4.20-0309-non-reasoning",
        "model": "grok-4.20-0309-non-reasoning-console",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "no reasoning; chat-completions image path",
    },
    {
        "key": "multi-agent",
        "model": "grok-4.20-multi-agent-console",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "user reasoning, default medium; chat-completions image path",
    },
    {
        "key": "multi-agent-low",
        "model": "grok-4.20-multi-agent-low",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "fixed low reasoning, 4 agents; chat-completions image path",
    },
    {
        "key": "multi-agent-medium",
        "model": "grok-4.20-multi-agent-medium",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "fixed medium reasoning, 4 agents; chat-completions image path",
    },
    {
        "key": "multi-agent-high",
        "model": "grok-4.20-multi-agent-high",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "fixed high reasoning, 16 agents; chat-completions image path",
    },
    {
        "key": "multi-agent-xhigh",
        "model": "grok-4.20-multi-agent-xhigh",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "fixed xhigh reasoning, 16 agents; chat-completions image path",
    },
    {
        "key": "build",
        "model": "grok-build-console",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "chat",
        "note": "build console model; chat-completions image path",
    },
    {
        "key": "image",
        "model": "grok-imagine-image-lite",
        "where": "local grok2api",
        "tier": "basic",
        "capability": "image",
        "note": "images/generations path",
    },
]
MODEL_ALIASES = {
    choice["key"]: choice["model"]
    for choice in MODEL_CHOICES
}
MODEL_ALIASES.update({
    str(index): choice["model"]
    for index, choice in enumerate(MODEL_CHOICES, start=1)
})
MODEL_ALIASES.update({
    "fast": "grok-4.20-fast",
    "lite": "grok-imagine-image-lite",
    "image-lite": "grok-imagine-image-lite",
    "grok-4.20-0309-reasoning-console": "grok-4.20-0309-reasoning-console",
    "grok-4.20-0309-non-reasoning-console": "grok-4.20-0309-non-reasoning-console",
})
ALLOWED_MODELS = {choice["model"] for choice in MODEL_CHOICES}
IMAGE_MODELS = {choice["model"] for choice in MODEL_CHOICES if choice["capability"] == "image"}
CHAT_MODELS = {choice["model"] for choice in MODEL_CHOICES if choice["capability"] == "chat"}
ASPECT_RATIO_TO_SIZE = {
    "1:1": "1024x1024",
    "16:9": "1280x720",
    "9:16": "720x1280",
    "3:2": "1792x1024",
    "2:3": "1024x1792",
}
Message = dict[str, Any]


class GeneratedImage(NamedTuple):
    label: str
    link: str
    saved_path: Path


def read_api_key(config_path: Path) -> str:
    env_key = (
        os.environ.get("GROK2API_API_KEY")
        or os.environ.get("XAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if env_key:
        return env_key.strip()

    if not config_path.exists():
        return ""

    try:
        import tomllib

        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return str(data.get("app", {}).get("api_key", "") or "").strip()
    except Exception:
        match = re.search(
            r'(?m)^\s*api_key\s*=\s*"([^"]*)"',
            config_path.read_text(encoding="utf-8", errors="replace"),
        )
        return match.group(1).strip() if match else ""


def request_json(
    url: str,
    api_key: str,
    payload: dict[str, object],
    *,
    retries: int,
    timeout: float,
) -> dict[str, object]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    last_error: str | None = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}\n{detail}"
            if exc.code < 500 or attempt >= attempts:
                raise SystemExit(f"request failed: {last_error}") from exc
            time.sleep(min(2 * attempt, 6))
        except URLError as exc:
            last_error = str(exc)
            if attempt >= attempts:
                raise SystemExit(f"request failed: {last_error}") from exc
            time.sleep(min(2 * attempt, 6))
    else:
        raise SystemExit(f"request failed: {last_error or 'unknown error'}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"response was not JSON:\n{raw[:500]!r}") from exc

    if isinstance(data, dict) and data.get("error"):
        raise SystemExit(f"api error: {json.dumps(data['error'], ensure_ascii=False)}")
    return data


def extension_from_content_type(content_type: str | None, fallback: str = ".png") -> str:
    if content_type:
        content_type = content_type.split(";", 1)[0].strip().lower()
        ext = mimetypes.guess_extension(content_type)
        if ext in {".jpe", ".jpeg"}:
            return ".jpg"
        if ext:
            return ext
    return fallback


def extension_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".png"


def download_image(url: str, base_url: str, api_key: str) -> tuple[bytes, str]:
    full_url = urljoin(base_url.rstrip("/") + "/", url)
    base_host = urlparse(base_url).netloc
    target_host = urlparse(full_url).netloc
    headers = {
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
    }
    if target_host == base_host:
        headers["Authorization"] = f"Bearer {api_key}"

    req = Request(full_url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=180) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"image download failed: HTTP {exc.code}\n{detail}") from exc
    except URLError as exc:
        raise SystemExit(f"image download failed: {exc}") from exc

    return raw, extension_from_content_type(content_type, extension_from_url(full_url))


def full_image_url(url: str, base_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", url)


def is_grok_cdn_url(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() == "assets.grok.com"


def is_local_proxy_image_url(url: str, base_url: str) -> bool:
    full_url = full_image_url(url, base_url)
    parsed = urlparse(full_url)
    base = urlparse(base_url)
    return parsed.netloc == base.netloc and parsed.path.rstrip("/") == "/v1/files/image"


def should_download_url(
    url: str,
    *,
    base_url: str,
    download_images: bool,
    links_only: bool,
) -> bool:
    if links_only:
        return False
    return download_images or is_local_proxy_image_url(url, base_url)


def write_url_file(output_dir: Path, stamp: str, index: int, stem: str, url: str) -> Path:
    path = output_dir / f"{stamp}_{index:02d}_{stem}.url"
    path.write_text(url + "\n", encoding="utf-8")
    return path


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())[:48].strip("-._")
    return stem or "image"


def resolve_model(model: str) -> str:
    key = (model or DEFAULT_MODEL).strip()
    resolved = MODEL_ALIASES.get(key.lower(), key)
    if resolved not in ALLOWED_MODELS:
        supported = ", ".join(sorted(ALLOWED_MODELS))
        raise SystemExit(f"unsupported model {key!r}; allowed models: {supported}")
    return resolved


def print_model_choices() -> None:
    print("Available model selectors:")
    for index, choice in enumerate(MODEL_CHOICES, start=1):
        print(
            f"  {index:>2}. {choice['key']:<26} -> {choice['model']:<42} "
            f"[{choice['where']}, {choice['tier']}, {choice['capability']}] {choice['note']}"
        )
    print("\nYou can also pass a full model id with --model.")


def direct_image_prompt_from_command(text: str) -> str | None:
    stripped = (text or "").strip()
    lowered = stripped.lower()
    for command in DIRECT_IMAGE_COMMANDS:
        if lowered == command:
            return ""
        if lowered.startswith(command + " "):
            return stripped[len(command):].strip()
    return None


def looks_like_image_request(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if direct_image_prompt_from_command(text) is not None:
        return True
    return any(keyword in lowered for keyword in IMAGE_INTENT_KEYWORDS)


def build_image_tool_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": IMAGE_TOOL_NAME,
            "description": (
                "Generate a photo or image with grok-imagine-image-lite. "
                "Call this when the user asks to create, draw, render, or generate a picture."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Detailed image prompt. Include subject, scene, style, lighting, "
                            "composition, and any constraints from the user."
                        ),
                    },
                    "n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4,
                        "default": 1,
                        "description": "Number of images to generate.",
                    },
                    "size": {
                        "type": "string",
                        "enum": [
                            "1024x1024",
                            "1280x720",
                            "720x1280",
                            "1792x1024",
                            "1024x1792",
                        ],
                        "default": "1024x1024",
                        "description": "Output image size.",
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    }


def resolve_size(size: str, aspect_ratio: str | None) -> str:
    if not aspect_ratio:
        return size
    normalized = aspect_ratio.strip().lower()
    if normalized == "auto":
        return size
    if normalized not in ASPECT_RATIO_TO_SIZE:
        supported = ", ".join([*ASPECT_RATIO_TO_SIZE, "auto"])
        raise SystemExit(f"unsupported local aspect ratio {aspect_ratio!r}; use one of: {supported}")
    return ASPECT_RATIO_TO_SIZE[normalized]


def is_local_grok2api(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    return host in {"127.0.0.1", "localhost", "::1"}


def save_images(
    data: dict[str, object],
    *,
    output_dir: Path,
    prompt: str,
    base_url: str,
    api_key: str,
    download_images: bool,
    links_only: bool,
) -> list[GeneratedImage]:
    items = data.get("data")
    if not isinstance(items, list):
        raise SystemExit(f"unexpected response shape:\n{json.dumps(data, ensure_ascii=False)[:1000]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = safe_stem(prompt)
    saved: list[GeneratedImage] = []

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        label = f"图片 {index}"
        if item.get("b64_json"):
            raw = base64.b64decode(str(item["b64_json"]))
            ext = ".png"
            path = output_dir / f"{stamp}_{index:02d}_{stem}{ext}"
            path.write_bytes(raw)
            saved.append(GeneratedImage(label=label, link=path.resolve().as_posix(), saved_path=path))
            continue

        if item.get("url"):
            url = full_image_url(str(item["url"]), base_url)
            if not should_download_url(
                url,
                base_url=base_url,
                download_images=download_images,
                links_only=links_only,
            ):
                path = write_url_file(output_dir, stamp, index, stem, url)
                saved.append(GeneratedImage(label=label, link=url, saved_path=path))
                continue
            try:
                raw, ext = download_image(url, base_url=base_url, api_key=api_key)
            except SystemExit as exc:
                path = write_url_file(output_dir, stamp, index, stem, url)
                failure = str(exc).splitlines()[0]
                print(f"Warning: saved URL only because image download failed: {failure}", file=sys.stderr)
                saved.append(GeneratedImage(label=label, link=url, saved_path=path))
                continue
        else:
            raise SystemExit(f"image item has no url or b64_json: {item}")

        path = output_dir / f"{stamp}_{index:02d}_{stem}{ext}"
        path.write_bytes(raw)
        saved.append(GeneratedImage(label=label, link=path.resolve().as_posix(), saved_path=path))

    if not saved:
        raise SystemExit("no images were returned")
    return saved


def extract_chat_content(data: dict[str, object]) -> str:
    chunks: list[str] = []
    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                chunks.append(text)
    return "\n".join(chunks)


def extract_tool_calls(data: dict[str, object]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    choices = data.get("choices")
    if not isinstance(choices, list):
        return calls

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if isinstance(function, dict) and function.get("name"):
                    calls.append(call)

        function_call = message.get("function_call")
        if isinstance(function_call, dict) and function_call.get("name"):
            calls.append(
                {
                    "id": f"call_{int(time.time() * 1000)}",
                    "type": "function",
                    "function": {
                        "name": function_call.get("name"),
                        "arguments": function_call.get("arguments", "{}"),
                    },
                }
            )
    return calls


def parse_tool_call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ValueError("tool call has no function payload")

    raw_args = function.get("arguments") or "{}"
    if isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str):
        args = json.loads(raw_args)
    else:
        raise ValueError("tool call arguments must be a JSON object")

    if not isinstance(args, dict):
        raise ValueError("tool call arguments must be a JSON object")
    return args


def extract_image_urls_from_text(text: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text or ""))
    candidates.extend(re.findall(r"https?://[^\s)>\"]+", text or ""))
    candidates.extend(re.findall(r"(?<!\w)(/v1/files/image\?id=[A-Za-z0-9_-]+)", text or ""))

    seen: set[str] = set()
    urls: list[str] = []
    for raw in candidates:
        url = raw.strip().strip("'\"")
        if not url:
            continue
        if url.startswith("data:image/"):
            if url not in seen:
                seen.add(url)
                urls.append(url)
            continue
        lower = url.lower()
        if (
            "/v1/files/image" not in lower
            and "image" not in lower
            and not lower.endswith((".png", ".jpg", ".jpeg", ".webp"))
        ):
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def save_chat_images(
    data: dict[str, object],
    *,
    output_dir: Path,
    prompt: str,
    base_url: str,
    api_key: str,
    download_images: bool,
    links_only: bool,
) -> list[GeneratedImage]:
    content = extract_chat_content(data)
    urls = extract_image_urls_from_text(content)
    if not urls:
        preview = content[:1000] if content else json.dumps(data, ensure_ascii=False)[:1000]
        raise SystemExit(f"chat response did not contain image URLs:\n{preview}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = safe_stem(prompt)
    saved: list[GeneratedImage] = []
    for index, url in enumerate(urls, start=1):
        label = f"图片 {index}"
        if url.startswith("data:image/"):
            header, b64_data = url.split(",", 1)
            mime = header.split(";", 1)[0].split(":", 1)[1]
            raw = base64.b64decode(b64_data)
            ext = extension_from_content_type(mime, ".png")
            path = output_dir / f"{stamp}_{index:02d}_{stem}{ext}"
            path.write_bytes(raw)
            link = path.resolve().as_posix()
        else:
            link = full_image_url(url, base_url)
            if not should_download_url(
                link,
                base_url=base_url,
                download_images=download_images,
                links_only=links_only,
            ):
                path = write_url_file(output_dir, stamp, index, stem, link)
            else:
                try:
                    raw, ext = download_image(link, base_url=base_url, api_key=api_key)
                    path = output_dir / f"{stamp}_{index:02d}_{stem}{ext}"
                    path.write_bytes(raw)
                    link = path.resolve().as_posix()
                except SystemExit as exc:
                    path = write_url_file(output_dir, stamp, index, stem, link)
                    failure = str(exc).splitlines()[0]
                    print(f"Warning: saved URL only because image download failed: {failure}", file=sys.stderr)
        saved.append(GeneratedImage(label=label, link=link, saved_path=path))
    return saved


def print_dialogue(
    prompt: str,
    model: str,
    assets: list[GeneratedImage],
    *,
    assistant_text: str = "",
) -> None:
    print(f"用户：{prompt}")
    if assets:
        print(f"助手：{assistant_text or f'已使用 `{model}` 生成图片。'}")
    else:
        print(f"助手：{assistant_text or '没有返回图片。'}")
    print()
    for asset in assets:
        if is_grok_cdn_url(asset.link):
            print(f"- {asset.label}：[Grok 原始图片链接，需要对应账号]({asset.link})")
            print("  注意：这个链接通常需要生成它的 Grok 账号会话；请优先使用 local_url/base64 输出。")
        else:
            print(f"- {asset.label}：[打开图片]({asset.link})")
        print(f"  记录文件：{asset.saved_path}")


def build_chat_payload(
    *,
    model: str,
    messages: list[Message],
    n: int,
    size: str,
    response_format: str,
    tools: list[dict[str, object]] | None = None,
    tool_choice: Any = None,
) -> tuple[str, dict[str, object]]:
    payload: dict[str, object] = {
        "model": model,
        "stream": False,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    if n != 1 or size != "1024x1024" or response_format != "url":
        payload["image_config"] = {
            "n": n,
            "size": size,
            "response_format": response_format,
        }
    return "chat", payload


def build_payload(
    *,
    model: str,
    prompt: str,
    n: int,
    size: str,
    response_format: str,
    aspect_ratio: str | None,
    resolution: str | None,
    base_url: str,
) -> tuple[str, dict[str, object]]:
    if model in IMAGE_MODELS:
        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "response_format": response_format,
        }
        if is_local_grok2api(base_url):
            payload["size"] = size
        else:
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
            if resolution:
                payload["resolution"] = resolution
        return "images", payload

    return build_chat_payload(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        n=n,
        size=size,
        response_format=response_format,
    )


def append_assistant_response(
    messages: list[Message],
    data: dict[str, object],
) -> str:
    content = extract_chat_content(data)
    tool_calls = extract_tool_calls(data)
    if content or tool_calls:
        message: Message = {"role": "assistant"}
        if content:
            message["content"] = content
        elif tool_calls:
            message["content"] = None
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)
    return content


def load_session_messages(path: Path) -> list[Message]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        messages = data.get("messages", [])
    else:
        raise SystemExit(f"session file has unsupported shape: {path}")
    if not isinstance(messages, list):
        raise SystemExit(f"session file messages must be a list: {path}")
    normalized: list[Message] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if not isinstance(role, str):
            continue

        message: Message = {"role": role}
        content = item.get("content")
        if isinstance(content, str) or isinstance(content, list) or content is None:
            if content is not None or role in {"assistant", "tool"}:
                message["content"] = content

        tool_calls = item.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list):
            message["tool_calls"] = tool_calls

        for key in ("tool_call_id", "name"):
            value = item.get(key)
            if isinstance(value, str):
                message[key] = value

        if "content" in message or "tool_calls" in message:
            normalized.append(message)
    return normalized


def save_session(path: Path, *, model: str, messages: list[Message]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "messages": messages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_numbered_response(path_template: str, turn: int, data: dict[str, object]) -> None:
    path = Path(path_template)
    if turn <= 1:
        target = path
    else:
        target = path.with_name(f"{path.stem}_{turn:03d}{path.suffix or '.json'}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_chat_assets_if_present(
    data: dict[str, object],
    *,
    output_dir: Path,
    prompt: str,
    base_url: str,
    api_key: str,
    download_images: bool,
    links_only: bool,
) -> list[GeneratedImage]:
    content = extract_chat_content(data)
    if not extract_image_urls_from_text(content):
        return []
    return save_chat_images(
        data,
        output_dir=output_dir,
        prompt=prompt,
        base_url=base_url,
        api_key=api_key,
        download_images=download_images,
        links_only=links_only,
    )


def bounded_tool_image_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 1
    return min(max(count, 1), 4)


def tool_call_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def tool_call_id(tool_call: dict[str, Any]) -> str:
    call_id = str(tool_call.get("id") or "").strip()
    if call_id:
        return call_id
    return f"call_{int(time.time() * 1000)}"


def execute_image_tool_call(
    tool_call: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    output_dir: Path,
    default_size: str,
    response_format: str,
    download_images: bool,
    links_only: bool,
    retries: int,
    timeout: float,
) -> tuple[dict[str, Any], list[GeneratedImage]]:
    if tool_call_name(tool_call) != IMAGE_TOOL_NAME:
        return (
            {
                "ok": False,
                "error": f"unsupported tool: {tool_call_name(tool_call) or '<missing>'}",
            },
            [],
        )

    try:
        args = parse_tool_call_arguments(tool_call)
    except (json.JSONDecodeError, ValueError) as exc:
        return {"ok": False, "error": f"invalid tool arguments: {exc}"}, []

    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "missing required argument: prompt"}, []

    n = bounded_tool_image_count(args.get("n", 1))
    size = str(args.get("size") or default_size or "1024x1024").strip()
    _, payload = build_payload(
        model=IMAGE_TOOL_MODEL,
        prompt=prompt,
        n=n,
        size=size,
        response_format=response_format,
        aspect_ratio=None,
        resolution=None,
        base_url=base_url,
    )
    data = request_json(
        f"{base_url}/images/generations",
        api_key,
        payload,
        retries=retries,
        timeout=timeout,
    )
    saved = save_images(
        data,
        output_dir=output_dir,
        prompt=prompt,
        base_url=base_url,
        api_key=api_key,
        download_images=download_images,
        links_only=links_only,
    )
    images = [
        {
            "label": asset.label,
            "link": asset.link,
            "saved_path": asset.saved_path.resolve().as_posix(),
        }
        for asset in saved
    ]
    return (
        {
            "ok": True,
            "tool": IMAGE_TOOL_NAME,
            "model": IMAGE_TOOL_MODEL,
            "prompt": prompt,
            "images": images,
            "markdown": "\n".join(f"- [{item['label']}]({item['link']})" for item in images),
        },
        saved,
    )


def append_tool_result(
    messages: list[Message],
    tool_call: dict[str, Any],
    result: dict[str, Any],
) -> None:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id(tool_call),
            "name": tool_call_name(tool_call),
            "content": json.dumps(result, ensure_ascii=False),
        }
    )


def direct_image_tool_call(prompt: str, *, n: int, size: str) -> dict[str, Any]:
    return {
        "id": f"call_direct_{int(time.time() * 1000)}",
        "type": "function",
        "function": {
            "name": IMAGE_TOOL_NAME,
            "arguments": json.dumps(
                {
                    "prompt": prompt,
                    "n": n,
                    "size": size,
                },
                ensure_ascii=False,
            ),
        },
    }


def generate_direct_image(
    prompt: str,
    *,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
    output_dir: Path,
) -> tuple[dict[str, Any], list[GeneratedImage]]:
    size = resolve_size(args.size, args.aspect_ratio)
    return execute_image_tool_call(
        direct_image_tool_call(prompt, n=args.n, size=size),
        base_url=base_url,
        api_key=api_key,
        output_dir=output_dir,
        default_size=size,
        response_format=args.response_format,
        download_images=args.download_images,
        links_only=args.links_only,
        retries=args.retries,
        timeout=args.timeout,
    )


def configure_interactive_agent_flags(args: argparse.Namespace) -> None:
    if getattr(args, "agent", False):
        args.interactive = True
    if getattr(args, "interactive", False):
        args.enable_image_tool = not getattr(args, "disable_image_tool", False)
        args.auto_image_fallback = not getattr(args, "disable_auto_image_fallback", False)


def run_interactive(args: argparse.Namespace, *, api_key: str, base_url: str, model: str) -> int:
    if model not in CHAT_MODELS:
        raise SystemExit("--interactive only supports chat-capable models")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_path = (
        Path(args.session_file)
        if args.session_file
        else output_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    messages = load_session_messages(session_path)
    if not messages and args.system:
        messages.append({"role": "system", "content": args.system.strip()})
        save_session(session_path, model=model, messages=messages)

    image_tools = [build_image_tool_schema()] if args.enable_image_tool else None
    print(f"持续对话模式：model={model}")
    print(f"会话文件：{session_path}")
    if image_tools:
        print(f"工具：{IMAGE_TOOL_NAME} -> {IMAGE_TOOL_MODEL}")
    if args.auto_image_fallback:
        print("图片 fallback：开启")
    print("输入 /image <提示词> 直接生图，/exit 退出，/reset 清空当前会话历史。")

    turn = sum(1 for msg in messages if msg.get("role") == "user")
    while True:
        try:
            user_text = input("你> ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            break
        if user_text == "/reset":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system.strip()})
            save_session(session_path, model=model, messages=messages)
            turn = 0
            print("助手：已清空当前会话历史。")
            continue

        direct_prompt = direct_image_prompt_from_command(user_text)
        if direct_prompt is not None:
            if not direct_prompt:
                print("助手：请在 /image 后输入图片提示词。")
                continue
            turn += 1
            messages.append({"role": "user", "content": user_text})
            print(f"助手：正在直接调用图片模型 {IMAGE_TOOL_MODEL} ...")
            result, assets = generate_direct_image(
                direct_prompt,
                args=args,
                base_url=base_url,
                api_key=api_key,
                output_dir=output_dir,
            )
            if result.get("ok"):
                assistant_text = "已直接调用图片模型生成。"
                messages.append(
                    {
                        "role": "assistant",
                        "content": str(result.get("markdown") or assistant_text),
                    }
                )
            else:
                assistant_text = f"图片工具调用失败：{result.get('error')}"
                messages.append({"role": "assistant", "content": assistant_text})
            save_session(session_path, model=model, messages=messages)
            print_dialogue(direct_prompt, IMAGE_TOOL_MODEL, assets, assistant_text=assistant_text)
            continue

        turn += 1
        request_messages = [*messages, {"role": "user", "content": user_text}]
        _, payload = build_chat_payload(
            model=model,
            messages=request_messages,
            n=args.n,
            size=resolve_size(args.size, args.aspect_ratio),
            response_format=args.response_format,
            tools=image_tools,
            tool_choice=args.tool_choice if image_tools else None,
        )
        data = request_json(
            f"{base_url}/chat/completions",
            api_key,
            payload,
            retries=args.retries,
            timeout=args.timeout,
        )
        if args.save_response:
            save_numbered_response(args.save_response, turn, data)

        messages = request_messages
        assistant_text = append_assistant_response(messages, data)
        tool_calls = extract_tool_calls(data)
        assets: list[GeneratedImage] = []
        used_image_model = False

        if tool_calls:
            used_image_model = True
            print(f"助手：正在调用工具 {IMAGE_TOOL_NAME} ...")
            for tool_call in tool_calls:
                result, saved_assets = execute_image_tool_call(
                    tool_call,
                    base_url=base_url,
                    api_key=api_key,
                    output_dir=output_dir,
                    default_size=resolve_size(args.size, args.aspect_ratio),
                    response_format=args.response_format,
                    download_images=args.download_images,
                    links_only=args.links_only,
                    retries=args.retries,
                    timeout=args.timeout,
                )
                assets.extend(saved_assets)
                append_tool_result(messages, tool_call, result)

            _, followup_payload = build_chat_payload(
                model=model,
                messages=messages,
                n=args.n,
                size=resolve_size(args.size, args.aspect_ratio),
                response_format=args.response_format,
            )
            followup_data = request_json(
                f"{base_url}/chat/completions",
                api_key,
                followup_payload,
                retries=args.retries,
                timeout=args.timeout,
            )
            if args.save_response:
                save_numbered_response(args.save_response, turn + 1000, followup_data)
            assistant_text = append_assistant_response(messages, followup_data)
            assets.extend(
                save_chat_assets_if_present(
                    followup_data,
                    output_dir=output_dir,
                    prompt=user_text,
                    base_url=base_url,
                    api_key=api_key,
                    download_images=args.download_images,
                    links_only=args.links_only,
                )
            )
        else:
            assets = save_chat_assets_if_present(
                data,
                output_dir=output_dir,
                prompt=user_text,
                base_url=base_url,
                api_key=api_key,
                download_images=args.download_images,
                links_only=args.links_only,
            )
            if (
                not assets
                and args.auto_image_fallback
                and looks_like_image_request(user_text)
            ):
                used_image_model = True
                print(f"助手：聊天模型没有返回图片，正在直接调用图片模型 {IMAGE_TOOL_MODEL} ...")
                result, assets = generate_direct_image(
                    user_text,
                    args=args,
                    base_url=base_url,
                    api_key=api_key,
                    output_dir=output_dir,
                )
                if result.get("ok"):
                    assistant_text = "聊天模型没有返回图片，已直接调用图片模型生成。"
                    messages.append(
                        {
                            "role": "assistant",
                            "content": str(result.get("markdown") or assistant_text),
                        }
                    )
                else:
                    assistant_text = (
                        f"{assistant_text}\n\n图片 fallback 失败：{result.get('error')}"
                        if assistant_text
                        else f"图片 fallback 失败：{result.get('error')}"
                    )

        save_session(session_path, model=model, messages=messages)
        display_model = f"{model} + {IMAGE_TOOL_MODEL}" if used_image_model else model
        print_dialogue(user_text, display_model, assets, assistant_text=assistant_text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an image through grok2api.")
    parser.add_argument("prompt", nargs="?", help="Image prompt. Can also use --prompt.")
    parser.add_argument("--prompt", dest="prompt_flag", help="Image prompt.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Default: {DEFAULT_BASE_URL}")
    parser.add_argument(
        "--model",
        default="fast",
        help=(
            "Model selector or full model id. Use --list-models to show selectors. "
            "Default: fast"
        ),
    )
    parser.add_argument("--list-models", action="store_true", help="Print model selectors and exit.")
    parser.add_argument("--size", default="1024x1024", help="Example: 1024x1024 or 1792x1024")
    parser.add_argument(
        "--aspect-ratio",
        choices=("1:1", "16:9", "9:16", "3:2", "2:3", "auto"),
        help=(
            "Official xAI parameter. For local grok2api it is mapped to size "
            "(1:1, 16:9, 9:16, 3:2, 2:3); lite ignores it upstream."
        ),
    )
    parser.add_argument(
        "--resolution",
        choices=("1k", "2k"),
        help="Official xAI parameter. Local grok2api may ignore it.",
    )
    parser.add_argument("--n", type=int, default=1, help="Number of images, default: 1")
    parser.add_argument(
        "--response-format",
        choices=("url", "b64_json"),
        default="url",
        help="Default: url",
    )
    parser.add_argument("--output-dir", default="generated_images", help="Default: generated_images")
    parser.add_argument("--api-key", default="", help="API key. Prefer GROK2API_API_KEY or XAI_API_KEY env var.")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start a persistent terminal chat loop and keep message history.",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Start interactive mode with the local image generation tool enabled.",
    )
    parser.add_argument(
        "--enable-image-tool",
        action="store_true",
        help="Expose generate_photo to the chat model in interactive mode. Enabled by default with --interactive.",
    )
    parser.add_argument(
        "--disable-image-tool",
        action="store_true",
        help="Use --interactive as plain chat without the local generate_photo tool.",
    )
    parser.add_argument(
        "--disable-auto-image-fallback",
        action="store_true",
        help="Do not directly call the image model when an image-like interactive request returns no image.",
    )
    parser.add_argument(
        "--tool-choice",
        default="auto",
        choices=("auto", "required", "none"),
        help="Tool choice for interactive image tools, default: auto.",
    )
    parser.add_argument("--session-file", help="Interactive mode session JSON path.")
    parser.add_argument("--system", default="", help="Optional system message for a new interactive session.")
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Also try to download Grok/external image URLs. Local proxy image URLs are downloaded by default.",
    )
    parser.add_argument(
        "--links-only",
        action="store_true",
        help="Do not download image URLs; save URL files and print links only.",
    )
    parser.add_argument(
        "--plain-output",
        action="store_true",
        help="Print only saved file paths instead of dialogue-style Markdown.",
    )
    parser.add_argument("--retries", type=int, default=2, help="Retry transient HTTP 5xx/network failures, default: 2")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout seconds, default: 300")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "data" / "config.toml"),
        help="Fallback config.toml path for app.api_key.",
    )
    parser.add_argument("--save-response", help="Optional path to save the raw JSON response.")
    args = parser.parse_args()

    configure_interactive_agent_flags(args)

    if args.list_models:
        print_model_choices()
        return 0

    api_key = args.api_key.strip() or read_api_key(Path(args.config))
    if not api_key:
        raise SystemExit("missing API key: set GROK2API_API_KEY or pass --api-key")

    base_url = args.base_url.rstrip("/")
    model = resolve_model(args.model)
    if args.model != model:
        print(f"Using model: {model} ({args.model})", file=sys.stderr)
    else:
        print(f"Using model: {model}", file=sys.stderr)

    if args.interactive:
        return run_interactive(args, api_key=api_key, base_url=base_url, model=model)

    prompt = (args.prompt_flag or args.prompt or "").strip()
    if not prompt:
        parser.error("prompt is required")

    size = resolve_size(args.size, args.aspect_ratio)
    endpoint, payload = build_payload(
        model=model,
        prompt=prompt,
        n=args.n,
        size=size,
        response_format=args.response_format,
        aspect_ratio=args.aspect_ratio,
        resolution=args.resolution,
        base_url=base_url,
    )
    path = "/images/generations" if endpoint == "images" else "/chat/completions"
    data = request_json(f"{base_url}{path}", api_key, payload, retries=args.retries, timeout=args.timeout)

    if args.save_response:
        Path(args.save_response).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    save_fn = save_images if endpoint == "images" else save_chat_images
    saved = save_fn(
        data,
        output_dir=Path(args.output_dir),
        prompt=prompt,
        base_url=base_url,
        api_key=api_key,
        download_images=args.download_images,
        links_only=args.links_only,
    )
    if args.plain_output:
        for asset in saved:
            print(asset.saved_path)
    else:
        print_dialogue(prompt, model, saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
