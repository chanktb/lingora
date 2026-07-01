"""AI image generation via Cloudflare Workers AI (FLUX-1-schnell, FREE).

Used by the `whats_this` layout to generate flat-cartoon illustrations for
each vocabulary item. 10 images per video, concurrent download via async.

API:
  POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell
  Authorization: Bearer {API_TOKEN}
  Body: {"prompt": "...", "steps": 4}
  Response: {"success": true, "result": {"image": "<base64-png>"}}

Free tier: 10,000 neurons/day (~100 images/day). ~2.7s/image.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("image_gen")

# ── Style suffixes appended to every prompt for visual consistency ──────────
# Selected at request time via env IMAGE_STYLE (default "cartoon"). The whole
# style experiment lives HERE — generator.py is untouched, so rollback is just
# flipping IMAGE_STYLE back to "cartoon" (or removing the env var).

# Default: flat-cartoon vocab cards — readable, friendly, no text-in-image.
STYLE_SUFFIX_CARTOON = (
    "Cute cartoon illustration, flat 2D vector art, vibrant colors, "
    "kawaii style, simple plain white background, centered subject, "
    "clear iconography, kids book style. NO TEXT, no letters, no words, "
    "no captions, no watermark, no signage."
)

# Experiment (IMAGE_STYLE=cinematic): realistic cinematic look.
STYLE_SUFFIX_CINEMATIC = (
    "Cinematic photograph, photorealistic, shot on 35mm film, shallow depth "
    "of field, dramatic natural lighting, rich cinematic color grading, "
    "subtle film grain, ultra-detailed, high dynamic range. "
    "NO TEXT, no letters, no words, no captions, no watermark, no signage."
)

# Cartoon descriptors that the generator's Gemini system prompts hard-code into
# the subject (e.g. scene prompts end with "illustration style", dialogue
# portraits say "kawaii cartoon portrait"). Left in cinematic mode they fight
# the cinematic suffix, so we strip them. Longest phrases first.
_CARTOON_TOKENS = [
    "flat 2d vector art", "flat illustration", "illustration style",
    "cartoon portrait", "kawaii cartoon", "vector art", "flat cartoon",
    "illustration", "cartoon", "kawaii", "flat 2d", "anime",
]

CF_MODEL = "@cf/black-forest-labs/flux-1-schnell"


def _style_mode(override: str | None = None) -> str:
    """Current style: 'cinematic' or 'cartoon' (default). Driven by IMAGE_STYLE.

    `override` — per-call opt-in ("cinematic" / "cartoon"). Used by layouts
    that want a specific style regardless of global env (e.g. vocab_card
    always cinematic — the whole point of the card is a real-looking scene
    behind the focal word).
    """
    if override:
        return override.strip().lower()
    return (os.environ.get("IMAGE_STYLE") or "cartoon").strip().lower()


def _style_suffix(override: str | None = None) -> str:
    return (
        STYLE_SUFFIX_CINEMATIC if _style_mode(override) == "cinematic"
        else STYLE_SUFFIX_CARTOON
    )


def _sanitize_subject(prompt: str) -> str:
    """Strip cartoon descriptors baked into the Gemini subject prompt.

    Only applied in cinematic mode. Removes tokens like "illustration style"
    so they don't conflict with the cinematic suffix, then tidies leftover
    punctuation/whitespace.
    """
    out = prompt
    for tok in _CARTOON_TOKENS:
        out = re.sub(re.escape(tok), "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out)              # collapse whitespace
    out = re.sub(r"\s*,(\s*,)+", ",", out)      # ", ," → ","
    out = re.sub(r"[,\s]+\.", ".", out)         # " ." / ", ." → "."
    return out.strip(" ,.;")


def build_full_prompt(subject_prompt: str, style: str | None = None) -> str:
    """Compose the final FLUX prompt = (sanitized) subject + style suffix.

    `style` — per-call override ("cinematic" / "cartoon"), takes precedence
    over the IMAGE_STYLE env var. Layouts pass their fixed style here.
    """
    subject = subject_prompt
    if _style_mode(style) == "cinematic":
        subject = _sanitize_subject(subject_prompt)
    return f"{subject}. {_style_suffix(style)}"


class CFQuotaExhaustedError(RuntimeError):
    """Raised when ALL configured CF accounts return 429 in a single pass.

    Signals a hard daily-quota wall (not a transient burst) so callers can
    SKIP the current turn instead of burning retries + long back-off waits.
    """


def _accounts() -> list[tuple[str, str]]:
    """Read CF account credentials, supporting rotation fallback.

    Preferred env var (v10+): CLOUDFLARE_ACCOUNTS = "id1:token1,id2:token2,..."
    Legacy env vars (fallback): CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN (single account).

    Returns ordered list of (account_id, token) to try in sequence on 429.
    Raises if no credentials configured.
    """
    accs_raw = (os.environ.get("CLOUDFLARE_ACCOUNTS") or "").strip()
    accounts: list[tuple[str, str]] = []
    if accs_raw:
        for pair in accs_raw.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            acc_id, token = pair.split(":", 1)
            acc_id = acc_id.strip()
            token = token.strip()
            if acc_id and token:
                accounts.append((acc_id, token))
    # Legacy single-account fallback (always APPENDED so rotation primary is preferred)
    legacy_id = (os.environ.get("CLOUDFLARE_ACCOUNT_ID") or "").strip()
    legacy_token = (os.environ.get("CLOUDFLARE_API_TOKEN") or "").strip()
    if legacy_id and legacy_token and (legacy_id, legacy_token) not in accounts:
        accounts.append((legacy_id, legacy_token))
    if not accounts:
        raise RuntimeError(
            "No CF credentials. Set CLOUDFLARE_ACCOUNTS=id1:token1,id2:token2 "
            "or legacy CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN.",
        )
    return accounts


def _request_one(account_id: str, api_token: str, full_prompt: str, dest: Path, *,
                 steps: int, timeout: float) -> int:
    """Single CF Workers AI request. Raises HTTPError on non-2xx."""
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/{CF_MODEL}"
    )
    payload = {"prompt": full_prompt, "steps": steps}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    if not body.get("success"):
        raise RuntimeError(f"Cloudflare AI returned error: {body}")
    b64_png = body["result"]["image"]
    png_bytes = base64.b64decode(b64_png)
    dest.write_bytes(png_bytes)
    return len(png_bytes)


def _fetch_blocking(subject_prompt: str, dest: Path, *,
                    steps: int = 4, timeout: float = 90.0,
                    style: str | None = None) -> int:
    """Blocking call with multi-account rotation on 429 (daily-quota exhausted).

    Tries each configured account in order until one succeeds. On HTTP 429 from
    an account, moves to next. Other errors propagate.
    """
    accounts = _accounts()
    full_prompt = build_full_prompt(subject_prompt, style=style)
    last_429: urllib.error.HTTPError | None = None
    for acc_id, token in accounts:
        try:
            return _request_one(acc_id, token, full_prompt, dest, steps=steps, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Quota exhausted on this account — try next.
                log.info("CF acct %s quota exhausted (429), trying next...", acc_id[:8])
                last_429 = exc
                continue
            raise
    # All accounts hit 429 — daily quota wall, not a transient burst. Raise a
    # distinct error so the caller can SKIP this turn instead of retrying.
    if last_429 is not None:
        raise CFQuotaExhaustedError(
            f"All {len(accounts)} CF account(s) quota-exhausted (429)"
        ) from last_429
    raise RuntimeError("No CF accounts available")


async def gen_image(subject_prompt: str, dest: Path, *, steps: int = 4,
                    timeout: float = 90.0, retries: int = 4,
                    style: str | None = None) -> Path:
    """Generate 1 image → save to dest. Async wrapper, with retry on transient err.

    Backoff is gentler for 429 (rate limit) since CF Workers AI shares burst quota
    across all concurrent requests: 5s, 10s, 20s, 40s. Other errors: 2s, 4s, 8s, 16s.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            size = await asyncio.to_thread(
                _fetch_blocking, subject_prompt, dest,
                steps=steps, timeout=timeout, style=style,
            )
            log.info("  ✓ %s (%d bytes, attempt %d)", dest.name, size, attempt + 1)
            return dest
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as exc:
            last_exc = exc
            # Hard quota wall (all accounts 429) — don't retry; bubble up fast
            # so the caller skips this turn rather than waiting 5+10+20+40s.
            if isinstance(exc, CFQuotaExhaustedError):
                log.warning("  ⏭ %s: tất cả CF account hết quota — bỏ qua (no retry)", dest.name)
                raise
            is_429 = isinstance(exc, urllib.error.HTTPError) and exc.code == 429
            log.warning("  ⚠ %s attempt %d failed: %s", dest.name, attempt + 1, exc)
            if attempt < retries:
                base = 5 if is_429 else 2
                wait = base * (2 ** attempt)  # exponential
                log.info("  ↻ %s retrying in %ds (%s)", dest.name, wait,
                         "rate-limited" if is_429 else "transient")
                await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


async def gen_images_batch(items: list[tuple[str, Path]],
                           max_concurrent: int = 3,
                           steps: int = 4) -> list[Path]:
    """Generate N images concurrently.

    Each `item` = (subject_prompt, dest_path). Returns list of dest paths
    in input order. Per-item failures DO raise — caller catches and falls back.

    `max_concurrent` caps parallel HTTP requests (CF allows burst; 5 is safe).
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded(prompt: str, dest: Path) -> Path:
        async with sem:
            return await gen_image(prompt, dest, steps=steps)

    tasks = [_bounded(prompt, dest) for prompt, dest in items]
    return await asyncio.gather(*tasks)
