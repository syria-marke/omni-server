import asyncio
import base64
import hmac
import json
import logging
import os
import platform
import shutil
import socket
import time
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import mimetypes
import re

import jwt
import psutil
import requests
from openai import AsyncOpenAI
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from pymongo import ASCENDING, DESCENDING, IndexModel
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

try:  # self-hosted builds do not ship the platform-only package
    from emergentintegrations.llm.chat import (
        ChatError,
        ImageContent,
        LlmChat,
        StreamDone,
        TextDelta,
        ToolCallStart,
        UserMessage,
    )
    HAVE_EMERGENT = True
except Exception:  # pragma: no cover
    HAVE_EMERGENT = False

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("emergentintegrations is not installed (self-hosted build)")

    ChatError = LlmChat = StreamDone = TextDelta = ToolCallStart = UserMessage = _Unavailable

    class ImageContent:  # minimal shim: enough for the OpenAI-compatible path
        def __init__(self, image_base64: str = "", mime_type: str = "", **kwargs):
            self.image_base64 = image_base64
            self.mime_type = mime_type

        @staticmethod
        def get_mime_type(b64: str) -> str:
            raw = b""
            for cut in (32, 24, 16, 8):
                try:
                    raw = base64.b64decode(b64[:cut] + "=" * (-len(b64[:cut]) % 4))
                    break
                except Exception:
                    continue
            if raw[:3] == b"\xff\xd8\xff":
                return "image/jpeg"
            if raw[:8] == b"\x89PNG\r\n\x1a\n":
                return "image/png"
            if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
                return "image/webp"
            if raw[:6] in (b"GIF87a", b"GIF89a"):
                return "image/gif"
            return "image/png"


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")
APP_PASSCODE = os.environ["APP_PASSCODE"]
JWT_SECRET = os.environ["JWT_SECRET"]
APP_NAME = os.environ["APP_NAME"]
JWT_ISSUER = "private-server-ai"
JWT_AUDIENCE = "owner-app"
TOKEN_DAYS = 7
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DS_BASE = "https://api.deepseek.com"


def _em(llm: str, model: str, label: str) -> dict[str, Any]:
    return {"kind": "emergent", "llm": llm, "model": model, "label": label, "vision": True, "group": "Emergent"}


BUILTIN_MODELS: dict[str, dict[str, Any]] = {
    "claude": _em("anthropic", "claude-sonnet-5-5", "Claude Sonnet 5.5"),
    "claude-opus": _em("anthropic", "claude-opus-5-5", "Claude Opus 5.5"),
    "claude-haiku": _em("anthropic", "claude-haiku-5-5", "Claude Haiku 5.5"),
    "gpt": _em("openai", "gpt-5.6-terra", "GPT-5.6 Terra"),
    "gemini": _em("gemini", "gemini-3.1-pro-preview", "Gemini 3.1 Pro"),
    "deepseek-flash": {"kind": "compat", "base_url": DS_BASE, "api_key": DEEPSEEK_API_KEY, "model": "deepseek-flash", "label": "DeepSeek Flash", "vision": True, "group": "DeepSeek"},
    "deepseek-pro": {"kind": "compat", "base_url": DS_BASE, "api_key": DEEPSEEK_API_KEY, "model": "deepseek-v4-pro", "label": "DeepSeek V4 Pro", "vision": False, "group": "DeepSeek"},
}
if not HAVE_EMERGENT:
    BUILTIN_MODELS = {k: v for k, v in BUILTIN_MODELS.items() if v.get("kind") != "emergent"}

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL") or (
    "claude" if HAVE_EMERGENT and "claude" in BUILTIN_MODELS else next(iter(BUILTIN_MODELS), "deepseek-flash")
)
VISION_HINTS = ("vision", "-vl", "vl-", "gpt-4o", "gpt-4.1", "gpt-5", "gpt-6", "claude", "gemini", "flash", "pixtral", "llava", "grok-4", "omni")
UPLOAD_DIR = Path.home() / "uploads"
MAX_UPLOAD = 100 * 1024 * 1024
MAX_AGENT_STEPS = 25
WORKDIR = str(Path.home())

STORAGE_BASE = (os.environ.get("INTEGRATION_PROXY_URL") or "").strip() or "https://integrations.emergentagent.com"
STORAGE_URL = STORAGE_BASE.rstrip("/") + "/objstore/api/v1/storage"
storage_key: Optional[str] = None

INDEXES: dict[str, list[IndexModel]] = {
    "conversations": [IndexModel([("updated_at", DESCENDING)], name="updated_desc")],
    "messages": [IndexModel([("conversation_id", ASCENDING), ("created_at", ASCENDING)], name="conv_created")],
    "files": [IndexModel([("storage_path", ASCENDING)], name="storage_path", unique=True)],
    "login_limits": [IndexModel([("created_at", ASCENDING)], name="ttl", expireAfterSeconds=3600)],
    "providers": [IndexModel([("created_at", ASCENDING)], name="created")],
}


async def ensure_indexes() -> None:
    for collection, models in INDEXES.items():
        for model in models:
            try:
                await db[collection].create_indexes([model])
            except Exception as exc:
                logger.error("ensure_indexes(%s.%s): %s", collection, model.document["name"], exc)


# ---------------------------------------------------------------- models
PyObjectId = Annotated[str, BeforeValidator(lambda v: str(v) if isinstance(v, ObjectId) else v)]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class BaseDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[PyObjectId] = Field(default=None, alias="_id")

    def to_mongo(self) -> dict:
        data = self.model_dump(by_alias=True, exclude={"id"})
        if self.id:
            data["_id"] = ObjectId(self.id)
        return data

    @classmethod
    def from_mongo(cls, doc: dict):
        return cls.model_validate(doc)

    def public(self, exclude: Optional[set] = None) -> dict:
        data = self.model_dump(exclude=exclude)
        for k, v in data.items():
            if isinstance(v, datetime):
                data[k] = v.replace(tzinfo=v.tzinfo or timezone.utc).isoformat()
        return data


class Conversation(BaseDocument):
    title: str = "محادثة جديدة"
    model: str = DEFAULT_MODEL
    preview: str = ""
    llm_history: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class Message(BaseDocument):
    conversation_id: str
    role: str
    text: str = ""
    image_path: Optional[str] = None
    image_paths: list[str] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    model_label: Optional[str] = None
    status: str = "done"
    parts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)


class StoredFile(BaseDocument):
    storage_path: str
    content_type: str
    size: int
    name: str = "file"
    server_path: Optional[str] = None
    created_at: datetime = Field(default_factory=now_utc)

    def attachment(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "storage_path": self.storage_path,
            "mime": self.content_type,
            "size": self.size,
            "server_path": self.server_path,
            "is_image": self.content_type.startswith("image/"),
        }


class Provider(BaseDocument):
    name: str
    base_url: str
    api_key: str
    models: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)

    def safe(self) -> dict:
        data = self.public(exclude={"api_key"})
        data["api_key_hint"] = "••••" + self.api_key[-4:]
        return data


class LoginBody(BaseModel):
    passcode: str = Field(min_length=1, max_length=64)


class ChatBody(BaseModel):
    text: str = Field(default="", max_length=50000)
    attachments: list[str] = Field(default_factory=list, max_length=10)
    image_paths: list[str] = Field(default_factory=list, max_length=4)


class ProviderBody(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    base_url: str = Field(min_length=4, max_length=300)
    api_key: str = Field(min_length=1, max_length=500)
    models: Optional[list[str]] = None


# ---------------------------------------------------------------- auth
bearer = HTTPBearer(auto_error=False)
LOGIN_WINDOW = 15 * 60
IP_LIMIT = 5
GLOBAL_LIMIT = 20


def decode_token(token: str) -> dict:
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], issuer=JWT_ISSUER, audience=JWT_AUDIENCE)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if claims.get("sub") != "owner":
        raise HTTPException(status_code=401, detail="Invalid token")
    return claims


async def require_owner(credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer)]) -> dict:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")
    return decode_token(credentials.credentials)


async def consume_login_limit(request: Request) -> bool:
    now = now_utc()
    bucket = int(now.timestamp()) // LOGIN_WINDOW
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    ip = forwarded or (request.client.host if request.client else "unknown")
    ok = True
    for key, limit in ((f"ip:{ip}:{bucket}", IP_LIMIT), (f"global:{bucket}", GLOBAL_LIMIT)):
        doc = await db.login_limits.find_one_and_update(
            {"_id": key},
            {"$inc": {"count": 1}, "$setOnInsert": {"created_at": now}},
            upsert=True,
            return_document=True,
        )
        if doc["count"] > limit:
            ok = False
    return ok


# ---------------------------------------------------------------- object storage (local disk)
LOCAL_STORE = Path(os.environ.get("STORAGE_DIR") or (Path.home() / "objstore"))
storage_key: Optional[str] = None


def _safe_obj_path(path: str) -> Path:
    clean = re.sub(r"[^\w./-]+", "_", path).lstrip("/")
    parts = [p for p in clean.split("/") if p not in ("", ".", "..")]
    dest = LOCAL_STORE.joinpath(*parts)
    dest.resolve().relative_to(LOCAL_STORE.resolve())
    return dest


def init_storage() -> str:
    global storage_key
    LOCAL_STORE.mkdir(parents=True, exist_ok=True)
    storage_key = "local"
    return storage_key


def put_object(path: str, data: bytes, content_type: str) -> dict:
    dest = _safe_obj_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return {"path": path, "size": len(data), "content_type": content_type}


def get_object(path: str) -> tuple[bytes, str]:
    dest = _safe_obj_path(path)
    if not dest.is_file():
        raise FileNotFoundError(path)
    return dest.read_bytes(), (mimetypes.guess_type(dest.name)[0] or "application/octet-stream")


# ---------------------------------------------------------------- agent tools
SYSTEM_PROMPT = f"""You are the owner's private AI agent, running inside their own app with full access to their Linux server through tools.
You can run shell commands, read and write files, install packages, inspect processes, manage services, write and run code — just like a senior engineer with a terminal.

Rules:
- Always reply in the same language the owner writes in (usually Arabic). Keep replies clear and concise; use markdown code fences for code.
- Use tools proactively to actually do the work instead of only explaining. Verify results after making changes.
- Commands are non-interactive (no prompts, no editors like vim/nano). Use flags like -y. Long-running servers must be started in the background with nohup and output redirected.
- Before anything destructive or irreversible (deleting data, killing important processes, rebooting, overwriting config), explain what you'll do and ask the owner to confirm first.
- Never reveal secrets from .env files unless the owner explicitly asks for them.
- Default working directory: {WORKDIR}. Server OS: {platform.platform()}.

UI design review (the owner develops a native Android app in Kotlin):
When the owner sends a design image and a screenshot of their app (image 1 = target design, image 2 = current app screenshot, unless they say otherwise), compare them precisely:
1. Start with an approximate match score (0-100%).
2. List every difference ordered by importance: layout & alignment, spacing/padding/margins (estimate in dp), sizes, colors (give hex values), typography (font, size in sp, weight), icons, corner radius, shadows/elevation, texts/copy, missing or extra elements.
3. Point out visible bugs (clipped/overflowing text, misalignment, wrong RTL direction, poor contrast, inconsistent states).
4. Give ready-to-paste Kotlin fixes (Jetpack Compose by default, or XML layouts if the screenshot/owner indicates Views), grouped per element.
When only one image is sent, review it for UI problems and missing elements. Be specific — never vague advice.

Files:
- The owner can attach any file (zip, apk, pdf, code, images...). Attached files are saved on the server under ~/uploads and their exact paths are listed in the message — work on them with your tools (unzip, read, edit, build, analyse).
- To hand a file back to the owner (built APK, zip, report, generated file), call share_file with its server path. It appears in chat as a downloadable file.

Model providers:
- If the owner gives you a model provider / proxy (base URL + API key), call add_model_provider so its models appear in their model picker. Use list_model_providers / remove_model_provider to manage them. Never repeat API keys back in your replies.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a non-interactive bash command on the server. Returns combined stdout/stderr and the exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run"},
                    "timeout_seconds": {"type": "integer", "description": "Timeout, default 60, max 300"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the server.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file on the server (parent folders are created).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "share_file",
            "description": "Send a file from the server to the owner as a downloadable attachment in the chat.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute path of the file on the server"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_model_provider",
            "description": "Register an OpenAI-compatible model provider or proxy so its models appear in the owner's model picker. Models are fetched automatically from {base_url}/models unless given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short display name"},
                    "base_url": {"type": "string", "description": "Base URL before /chat/completions, e.g. https://openrouter.ai/api/v1"},
                    "api_key": {"type": "string"},
                    "models": {"type": "array", "items": {"type": "string"}, "description": "Optional explicit model ids"},
                },
                "required": ["name", "base_url", "api_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_model_providers",
            "description": "List the custom model providers already added (without keys).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_model_provider",
            "description": "Remove a custom model provider by its name.",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
    },
]

MAX_OUTPUT = 12000


def clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    half = MAX_OUTPUT // 2
    return text[:half] + f"\n... [{len(text) - MAX_OUTPUT} chars truncated] ...\n" + text[-half:]


async def run_command(command: str, timeout_seconds: int = 60) -> tuple[str, int]:
    timeout = max(1, min(int(timeout_seconds or 60), 300))
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=WORKDIR,
        executable="/bin/bash",
        start_new_session=True,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, 9)
        return f"Command timed out after {timeout}s", 124
    return clip(out.decode("utf-8", errors="replace")), proc.returncode


def normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith("http"):
        url = "https://" + url
    for suffix in ("/chat/completions", "/models"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def guess_vision(model_id: str) -> bool:
    low = model_id.lower()
    return any(h in low for h in VISION_HINTS)


async def fetch_models(base_url: str, api_key: str) -> list[str]:
    def _get() -> list[str]:
        r = requests.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=25)
        r.raise_for_status()
        data = r.json()
        items = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
        return [m.get("id") or m.get("name") if isinstance(m, dict) else str(m) for m in items][:300]

    return [m for m in await run_in_threadpool(_get) if m]


async def create_provider(name: str, base_url: str, api_key: str, models: Optional[list[str]] = None) -> Provider:
    base_url = normalize_base_url(base_url)
    ids = [m.strip() for m in (models or []) if m.strip()] or await fetch_models(base_url, api_key.strip())
    if not ids:
        raise ValueError("المزود لم يرجع أي نموذج")
    provider = Provider(
        name=name.strip()[:40],
        base_url=base_url,
        api_key=api_key.strip(),
        models=[{"id": i, "vision": guess_vision(i)} for i in ids],
    )
    res = await db.providers.insert_one(provider.to_mongo())
    provider.id = str(res.inserted_id)
    return provider


async def resolve_model(key: str) -> Optional[dict[str, Any]]:
    if key in BUILTIN_MODELS:
        return {**BUILTIN_MODELS[key], "key": key}
    if key.startswith("p:"):
        try:
            _, pid, model_id = key.split(":", 2)
        except ValueError:
            return None
        if not ObjectId.is_valid(pid):
            return None
        doc = await db.providers.find_one({"_id": ObjectId(pid)})
        if not doc:
            return None
        p = Provider.from_mongo(doc)
        m = next((m for m in p.models if m["id"] == model_id), None)
        if not m:
            return None
        return {"key": key, "kind": "compat", "base_url": p.base_url, "api_key": p.api_key, "model": model_id, "label": model_id, "vision": m.get("vision", False), "group": p.name}
    return None


async def list_all_models() -> list[dict[str, Any]]:
    out = [
        {"key": k, "label": v["label"], "vision": v["vision"], "group": v["group"], "available": v["kind"] != "compat" or bool(v["api_key"])}
        for k, v in BUILTIN_MODELS.items()
    ]
    for doc in await db.providers.find().sort("created_at", 1).to_list(100):
        p = Provider.from_mongo(doc)
        out += [{"key": f"p:{p.id}:{m['id']}", "label": m["id"], "vision": m.get("vision", False), "group": p.name, "available": True} for m in p.models]
    return out


_clients: dict[tuple[str, str], AsyncOpenAI] = {}


def compat_client(base_url: str, api_key: str) -> AsyncOpenAI:
    key = (base_url, api_key)
    if key not in _clients:
        _clients[key] = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=600)
    return _clients[key]


async def share_file(path_str: str) -> tuple[str, int, Optional[dict]]:
    p = Path(path_str).expanduser()
    if not p.is_file():
        return f"File not found: {p}", 1, None
    size = p.stat().st_size
    if size > MAX_UPLOAD:
        return f"File too large ({size} bytes, max {MAX_UPLOAD})", 1, None
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    if p.suffix.lower() == ".apk":
        mime = "application/vnd.android.package-archive"
    data = await run_in_threadpool(p.read_bytes)
    result = await run_in_threadpool(put_object, f"{APP_NAME}/shared/{uuid.uuid4().hex[:12]}/{safe_name(p.name)}", data, mime)
    stored = StoredFile(storage_path=result["path"], content_type=mime, size=size, name=p.name, server_path=str(p))
    await db.files.insert_one(stored.to_mongo())
    return f"Shared {p.name} ({size} bytes) with the owner.", 0, {"type": "file", **stored.attachment()}


def safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name)[-80:] or "file"


async def execute_tool(name: str, args: dict) -> tuple[str, int, Optional[dict]]:
    try:
        if name == "run_command":
            out, code = await run_command(args.get("command", ""), args.get("timeout_seconds", 60))
            return out, code, None
        if name == "read_file":
            p = Path(args["path"]).expanduser()
            return clip(await run_in_threadpool(p.read_text, "utf-8", "replace")), 0, None
        if name == "write_file":
            p = Path(args["path"]).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            await run_in_threadpool(p.write_text, args.get("content", ""), "utf-8")
            return f"Wrote {len(args.get('content', ''))} chars to {p}", 0, None
        if name == "share_file":
            return await share_file(args.get("path", ""))
        if name == "add_model_provider":
            prov = await create_provider(args.get("name", "Provider"), args.get("base_url", ""), args.get("api_key", ""), args.get("models"))
            sample = ", ".join(m["id"] for m in prov.models[:15])
            return f"Added provider '{prov.name}' with {len(prov.models)} models: {sample}", 0, None
        if name == "list_model_providers":
            docs = await db.providers.find().to_list(100)
            if not docs:
                return "No custom providers yet.", 0, None
            lines = [f"- {d['name']} ({d['base_url']}): {len(d.get('models', []))} models" for d in docs]
            return "\n".join(lines), 0, None
        if name == "remove_model_provider":
            res = await db.providers.delete_one({"name": args.get("name", "")})
            return ("Removed." if res.deleted_count else "No provider with that name."), 0 if res.deleted_count else 1, None
        return f"Unknown tool {name}", 1, None
    except Exception as exc:
        return f"Error: {exc}", 1, None


def sanitize_history(messages: list[dict]) -> list[dict]:
    """Strip base64 images and close any dangling tool calls so history stays valid."""
    out: list[dict] = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = [
                {"type": "text", "text": "[المالك أرسل صورة هنا]"} if c.get("type") == "image_url" else c
                for c in m["content"]
            ]
        out.append(m)
    # find last assistant with tool calls and add missing results
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "assistant":
            ids = [tc["id"] for tc in (out[i].get("tool_calls") or [])]
            answered = {m.get("tool_call_id") for m in out[i + 1:] if m.get("role") == "tool"}
            for tid in ids:
                if tid not in answered:
                    out.append({"role": "tool", "tool_call_id": tid, "content": "Interrupted by owner."})
            break
    if out and out[-1].get("role") in ("tool", "user"):
        out.append({"role": "assistant", "content": "(توقفت هنا)"})
    return out


def strip_reasoning(messages: list[dict]) -> list[dict]:
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]


async def compat_turn(client_: AsyncOpenAI, messages: list[dict], model: str):
    """One streamed OpenAI-compatible completion. Yields ("text", delta) then ("tools", [(id, name, args)]).
    Appends the assistant message to `messages`."""
    stream = await client_.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, stream=True, max_tokens=8000
    )
    content, reasoning = "", ""
    calls: dict[int, dict[str, str]] = {}
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        r = getattr(delta, "reasoning_content", None)
        if r:
            reasoning += r
        if delta.content:
            content += delta.content
            yield "text", delta.content
        for tc in delta.tool_calls or []:
            c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
            if tc.id:
                c["id"] = tc.id
            if tc.function and tc.function.name:
                c["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                c["args"] += tc.function.arguments
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
            for c in calls.values()
        ]
    messages.append(msg)
    parsed = []
    for c in calls.values():
        try:
            args = json.loads(c["args"] or "{}")
        except json.JSONDecodeError:
            args = {}
        parsed.append((c["id"], c["name"], args))
    yield "tools", parsed


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------- app
app = FastAPI()


@app.get("/health", include_in_schema=False)
@app.head("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


api = APIRouter(prefix="/api")


@api.get("/")
async def root():
    return {"message": "ok"}


@api.post("/auth/login")
async def login(body: LoginBody, request: Request):
    allowed = await consume_login_limit(request)
    valid = allowed and hmac.compare_digest(body.passcode.encode(), APP_PASSCODE.encode())
    if not valid:
        if not allowed:
            raise HTTPException(status_code=429, detail="محاولات كثيرة، حاول بعد 15 دقيقة")
        raise HTTPException(status_code=401, detail="الرمز غير صحيح")
    now = now_utc()
    token = jwt.encode(
        {"sub": "owner", "iss": JWT_ISSUER, "aud": JWT_AUDIENCE, "iat": now, "exp": now + timedelta(days=TOKEN_DAYS)},
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"access_token": token, "token_type": "bearer"}


@api.get("/auth/me")
async def me(_: Annotated[dict, Depends(require_owner)]):
    return {"authenticated": True}


async def get_conversation(cid: str) -> Conversation:
    if not ObjectId.is_valid(cid):
        raise HTTPException(status_code=404, detail="Conversation not found")
    doc = await db.conversations.find_one({"_id": ObjectId(cid), "deleted_at": None})
    if not doc:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Conversation.from_mongo(doc)


@api.get("/conversations")
async def list_conversations(_: Annotated[dict, Depends(require_owner)]):
    docs = await db.conversations.find({"deleted_at": None}, {"llm_history": 0}).sort("updated_at", -1).to_list(200)
    return [{**Conversation.from_mongo(d).public(exclude={"llm_history"}), "running": str(d["_id"]) in RUNS} for d in docs]


@api.post("/conversations")
async def create_conversation(_: Annotated[dict, Depends(require_owner)]):
    last = await db.conversations.find_one({"deleted_at": None}, {"model": 1}, sort=[("updated_at", -1)])
    conv = Conversation(model=(last or {}).get("model") or DEFAULT_MODEL)
    res = await db.conversations.insert_one(conv.to_mongo())
    conv.id = str(res.inserted_id)
    return conv.public(exclude={"llm_history"})


@api.get("/conversations/{cid}")
async def get_conversation_route(cid: str, _: Annotated[dict, Depends(require_owner)]):
    return (await get_conversation(cid)).public(exclude={"llm_history"})


@api.get("/models")
async def list_models(_: Annotated[dict, Depends(require_owner)]):
    return await list_all_models()


class ConversationPatch(BaseModel):
    model: str


@api.patch("/conversations/{cid}")
async def update_conversation(cid: str, body: ConversationPatch, _: Annotated[dict, Depends(require_owner)]):
    if not await resolve_model(body.model):
        raise HTTPException(status_code=400, detail="Unknown model")
    conv = await get_conversation(cid)
    await db.conversations.update_one({"_id": ObjectId(conv.id)}, {"$set": {"model": body.model}})
    conv.model = body.model
    return conv.public(exclude={"llm_history"})


@api.get("/providers")
async def list_providers(_: Annotated[dict, Depends(require_owner)]):
    docs = await db.providers.find().sort("created_at", 1).to_list(100)
    return [Provider.from_mongo(d).safe() for d in docs]


@api.post("/providers")
async def add_provider(body: ProviderBody, _: Annotated[dict, Depends(require_owner)]):
    try:
        prov = await create_provider(body.name, body.base_url, body.api_key, body.models)
    except Exception as exc:
        err = str(exc).replace(body.api_key.strip(), "••••")
        raise HTTPException(status_code=400, detail=f"تعذّر جلب النماذج من المزود: {err[:160]}")
    return prov.safe()


async def get_provider(pid: str) -> Provider:
    if not ObjectId.is_valid(pid):
        raise HTTPException(status_code=404, detail="Provider not found")
    doc = await db.providers.find_one({"_id": ObjectId(pid)})
    if not doc:
        raise HTTPException(status_code=404, detail="Provider not found")
    return Provider.from_mongo(doc)


@api.post("/providers/{pid}/refresh")
async def refresh_provider(pid: str, _: Annotated[dict, Depends(require_owner)]):
    prov = await get_provider(pid)
    try:
        ids = await fetch_models(prov.base_url, prov.api_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"تعذّر تحديث النماذج: {str(exc)[:160]}")
    models = [{"id": i, "vision": guess_vision(i)} for i in ids]
    await db.providers.update_one({"_id": ObjectId(prov.id)}, {"$set": {"models": models}})
    prov.models = models
    return prov.safe()


@api.delete("/providers/{pid}")
async def delete_provider(pid: str, _: Annotated[dict, Depends(require_owner)]):
    prov = await get_provider(pid)
    # Hard delete on explicit owner action: the record holds an API key that should not linger.
    await db.providers.delete_one({"_id": ObjectId(prov.id)})
    return {"deleted": True}


@api.delete("/conversations/{cid}")
async def delete_conversation(cid: str, _: Annotated[dict, Depends(require_owner)]):
    conv = await get_conversation(cid)
    await db.conversations.update_one({"_id": ObjectId(conv.id)}, {"$set": {"deleted_at": now_utc()}})
    return {"deleted": True}


@api.get("/conversations/{cid}/messages")
async def list_messages(cid: str, _: Annotated[dict, Depends(require_owner)]):
    conv = await get_conversation(cid)
    docs = await db.messages.find({"conversation_id": conv.id}).sort("created_at", 1).to_list(1000)
    return [Message.from_mongo(d).public() for d in docs]


@api.post("/upload")
async def upload(_: Annotated[dict, Depends(require_owner)], file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="الملف أكبر من 100 ميغابايت")
    name = file.filename or "file"
    content_type = file.content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
    if name.lower().endswith(".apk"):
        content_type = "application/vnd.android.package-archive"
    clean = safe_name(name)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    server_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{clean}"
    await run_in_threadpool(server_path.write_bytes, data)
    try:
        result = await run_in_threadpool(put_object, f"{APP_NAME}/uploads/owner/{uuid.uuid4()}/{clean}", data, content_type)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 402:
            raise HTTPException(status_code=402, detail="الرصيد غير كافٍ لرفع الملفات")
        raise HTTPException(status_code=502, detail="تعذّر رفع الملف")
    stored = StoredFile(storage_path=result["path"], content_type=content_type, size=len(data), name=name, server_path=str(server_path))
    await db.files.insert_one(stored.to_mongo())
    return {"path": result["path"], **stored.attachment()}


@api.get("/files/{path:path}")
async def download(
    path: str,
    token: Optional[str] = Query(None),
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer)] = None,
):
    raw = credentials.credentials if credentials else token
    if not raw:
        raise HTTPException(status_code=401, detail="Not authenticated")
    decode_token(raw)
    doc = await db.files.find_one({"storage_path": path})
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")
    record = StoredFile.from_mongo(doc)
    data, _ = await run_in_threadpool(get_object, path)
    disposition = "inline" if record.content_type.startswith("image/") else "attachment"
    return Response(
        content=data,
        media_type=record.content_type,
        headers={"Cache-Control": "private, max-age=86400", "Content-Disposition": f'{disposition}; filename="{safe_name(record.name)}"'},
    )


@api.get("/server/status")
async def server_status(_: Annotated[dict, Depends(require_owner)]):
    cpu = await run_in_threadpool(psutil.cpu_percent, 0.4)
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage("/")
    load = os.getloadavg()
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": cpu,
        "load": [round(x, 2) for x in load],
        "memory": {"total": mem.total, "used": mem.total - mem.available, "percent": mem.percent},
        "disk": {"total": disk.total, "used": disk.used, "percent": round(disk.used / disk.total * 100, 1)},
        "uptime_seconds": int(time.time() - psutil.boot_time()),
        "model": BUILTIN_MODELS[DEFAULT_MODEL]["label"],
    }


class Run:
    """A background agent run. Survives client disconnects; clients subscribe for live events."""

    def __init__(self) -> None:
        self.queues: list[asyncio.Queue] = []
        self.task: Optional[asyncio.Task] = None

    def publish(self, ev: dict) -> None:
        for q in list(self.queues):
            q.put_nowait(ev)


RUNS: dict[str, Run] = {}


def friendly_error(exc: Exception, spec: dict) -> str:
    low = str(exc).lower()
    if "budget" in low:
        return "\n\nانتهى رصيد مفتاح Emergent. اشحنه من: الملف الشخصي ← Manage plan ← Universal Key ← Add Balance، أو بدّل النموذج."
    if "insufficient" in low or "402" in low or "quota" in low:
        return f"\n\nانتهى رصيد حسابك عند {spec['group']}. اشحنه أو بدّل النموذج."
    if "401" in low or "authentication" in low or "invalid api key" in low:
        return f"\n\nمفتاح {spec['group']} غير صالح. حدّثه من الإعدادات."
    if "404" in low or "not found" in low or "does not exist" in low:
        return f"\n\nالنموذج {spec['label']} غير متاح عند {spec['group']}. جرّب نموذجاً آخر."
    return "\n\nحدث خطأ أثناء التواصل مع الذكاء الاصطناعي، حاول مرة أخرى."


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


async def run_agent(
    run: Run,
    conv: Conversation,
    spec: dict,
    text: str,
    files: list[StoredFile],
    images_b64: list[str],
    assistant: Message,
    preview: str,
) -> None:
    default_text = (
        "الصورة 1 هي التصميم المطلوب، والصورة 2 لقطة شاشة من تطبيقي. قارن بينهما."
        if len(images_b64) >= 2
        else ("راجع هذه الشاشة" if images_b64 else "اطّلع على الملفات المرفقة")
    )
    prompt = text or default_text
    if len(images_b64) > 1:
        prompt += f"\n[{len(images_b64)} صور مرفقة بالترتيب]"
    if files:
        lines = [f"- {f.name} ({human_size(f.size)}, {f.content_type}): {f.server_path}" for f in files]
        prompt += "\n\n[Attached files saved on the server]\n" + "\n".join(lines)

    history = strip_reasoning(conv.llm_history) if spec["kind"] == "emergent" else conv.llm_history
    base_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    chat_session: Optional[LlmChat] = None
    ds_messages: list[dict] = []
    next_msg: Optional[UserMessage] = None
    if spec["kind"] == "emergent":
        chat_session = (
            LlmChat(api_key=EMERGENT_LLM_KEY, session_id=conv.id, system_message=SYSTEM_PROMPT, initial_messages=base_messages)
            .with_model(spec["llm"], spec["model"])
            .with_tools(TOOLS)
            .with_params(max_tokens=8000)
        )
        next_msg = UserMessage(text=prompt, file_contents=[ImageContent(image_base64=b) for b in images_b64])
    else:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b in images_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:{ImageContent.get_mime_type(b)};base64,{b}"}})
        ds_messages = base_messages + [{"role": "user", "content": content if images_b64 else prompt}]
        llm_client = compat_client(spec["base_url"], spec["api_key"])

    parts: list[dict[str, Any]] = []
    status = "done"
    last_flush = time.monotonic()

    def joined_text() -> str:
        return "".join(p["text"] for p in parts if p["type"] == "text")

    async def flush(final_status: str = "running") -> None:
        await db.messages.update_one(
            {"_id": ObjectId(assistant.id)},
            {"$set": {"parts": parts, "text": joined_text(), "status": final_status}},
        )

    async def add_text(delta: str) -> None:
        nonlocal last_flush
        if parts and parts[-1]["type"] == "text":
            parts[-1]["text"] += delta
        else:
            parts.append({"type": "text", "text": delta})
        run.publish({"type": "text", "content": delta})
        if time.monotonic() - last_flush > 1.2:
            last_flush = time.monotonic()
            await flush()

    try:
        for _step in range(MAX_AGENT_STEPS):
            tool_calls: list[tuple[str, str, dict]] = []
            if chat_session is not None:
                async for ev in chat_session.stream_message(next_msg):
                    if isinstance(ev, TextDelta):
                        await add_text(ev.content)
                    elif isinstance(ev, ToolCallStart):
                        run.publish({"type": "tool_pending", "id": ev.id, "name": ev.name})
                    elif isinstance(ev, StreamDone):
                        tool_calls = [(tc.id, tc.name, tc.arguments) for tc in (ev.tool_calls or [])]
                next_msg = None
            else:
                async for kind, payload in compat_turn(llm_client, ds_messages, spec["model"]):
                    if kind == "text":
                        await add_text(payload)
                    elif kind == "tools":
                        tool_calls = payload
            if not tool_calls:
                break
            for tc_id, tc_name, tc_args in tool_calls:
                safe_args = {k: ("••••" if k == "api_key" else v) for k, v in (tc_args or {}).items()}
                part = {"type": "tool", "id": tc_id, "name": tc_name, "input": safe_args, "output": None, "exit_code": None}
                parts.append(part)
                run.publish({"type": "tool_start", "part": part})
                await flush()
                output, code, extra = await execute_tool(tc_name, tc_args or {})
                part["output"], part["exit_code"] = output, code
                run.publish({"type": "tool_result", "part": part})
                if extra:
                    parts.append(extra)
                    run.publish({"type": "file", "part": extra})
                await flush()
                result = f"exit_code={code}\n{output}"
                if chat_session is not None:
                    chat_session.add_tool_result(tc_id, result)
                else:
                    ds_messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
        else:
            parts.append({"type": "text", "text": "\n\n(وصلت إلى الحد الأقصى من الخطوات، اكتب «تابع» لأكمل)"})
    except asyncio.CancelledError:
        status = "stopped"
        parts.append({"type": "text", "text": "\n\n(تم الإيقاف)"})
    except Exception as exc:
        logger.exception("LLM error")
        status = "error"
        msg = friendly_error(exc, spec)
        parts.append({"type": "text", "text": msg})
        run.publish({"type": "error", "message": msg.strip()})
    finally:
        RUNS.pop(conv.id, None)
        raw = chat_session.messages[1:] if chat_session is not None else ds_messages[1:]
        assistant.parts = parts
        assistant.text = joined_text()
        assistant.status = status
        await flush(status)
        await db.conversations.update_one(
            {"_id": ObjectId(conv.id)},
            {"$set": {
                "llm_history": sanitize_history(raw),
                "updated_at": now_utc(),
                "preview": (assistant.text or preview).replace("*", "").replace("`", "").replace("#", "").strip()[:120],
            }},
        )
        run.publish({"type": "done", "message": assistant.public()})


@api.post("/conversations/{cid}/chat")
async def chat(cid: str, body: ChatBody, _: Annotated[dict, Depends(require_owner)]):
    conv = await get_conversation(cid)
    if conv.id in RUNS:
        raise HTTPException(status_code=409, detail="المساعد ما زال يعمل في هذه المحادثة")
    text = body.text.strip()
    paths = body.attachments + body.image_paths
    if not text and not paths:
        raise HTTPException(status_code=400, detail="الرسالة فارغة")

    spec = await resolve_model(conv.model) or await resolve_model(DEFAULT_MODEL)
    if spec["kind"] == "compat" and not spec["api_key"]:
        raise HTTPException(status_code=400, detail=f"مفتاح {spec['group']} غير مضاف")

    files: list[StoredFile] = []
    for p in paths:
        doc = await db.files.find_one({"storage_path": p})
        if not doc:
            raise HTTPException(status_code=404, detail="File not found")
        files.append(StoredFile.from_mongo(doc))
    images = [f for f in files if f.content_type.startswith("image/")]
    if images and not spec["vision"]:
        raise HTTPException(status_code=400, detail=f"{spec['label']} لا يدعم الصور، بدّل النموذج")
    images_b64: list[str] = []
    for f in images:
        data, _ct = await run_in_threadpool(get_object, f.storage_path)
        images_b64.append(base64.b64encode(data).decode())

    user_msg = Message(conversation_id=conv.id, role="user", text=text, attachments=[f.attachment() for f in files])
    res = await db.messages.insert_one(user_msg.to_mongo())
    user_msg.id = str(res.inserted_id)
    assistant = Message(conversation_id=conv.id, role="assistant", model_label=spec["label"], status="running")
    res = await db.messages.insert_one(assistant.to_mongo())
    assistant.id = str(res.inserted_id)

    preview = (text or (files[0].name if files else "ملف"))[:120]
    updates: dict[str, Any] = {"updated_at": now_utc(), "preview": preview}
    if conv.title == "محادثة جديدة":
        updates["title"] = preview[:48]
    await db.conversations.update_one({"_id": ObjectId(conv.id)}, {"$set": updates})

    run = Run()
    queue: asyncio.Queue = asyncio.Queue()
    run.queues.append(queue)
    RUNS[conv.id] = run
    run.task = asyncio.create_task(run_agent(run, conv, spec, text, files, images_b64, assistant, preview))

    async def stream():
        try:
            yield sse({"type": "user_message", "message": user_msg.public()})
            yield sse({"type": "assistant_started", "message": assistant.public()})
            while True:
                ev = await queue.get()
                yield sse(ev)
                if ev["type"] == "done":
                    break
        finally:
            with suppress(ValueError):
                run.queues.remove(queue)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@api.post("/conversations/{cid}/stop")
async def stop_run(cid: str, _: Annotated[dict, Depends(require_owner)]):
    run = RUNS.get(cid)
    if run and run.task:
        run.task.cancel()
        return {"stopped": True}
    return {"stopped": False}


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    app.state.index_task = asyncio.create_task(ensure_indexes())
    await db.messages.update_many({"status": "running"}, {"$set": {"status": "interrupted"}})
    try:
        await run_in_threadpool(init_storage)
    except Exception as exc:
        logger.error("Storage init failed: %s", exc)


@app.on_event("shutdown")
async def shutdown_db_client():
    app.state.index_task.cancel()
    with suppress(asyncio.CancelledError):
        await app.state.index_task
    client.close()
