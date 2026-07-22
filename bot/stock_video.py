"""Pexels stock video adapter for the guess_word layout background.

Fetches a short vertical 9:16 clip matching a 2-3 word English query,
strips audio (TTS narration is the only audio track we want), and returns
the local MP4 path. Callers pass the path to the composer, which copies it
into the HyperFrames project's ``static/`` dir so the template's <video>
element can play it.

Rate: free tier 200/h, plenty for guess_word which fires ~1x/day/channel.
Env:  PEXELS_API_KEY (single) or PEXELS_API_KEYS (comma-separated pool).
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from pathlib import Path

import httpx

log = logging.getLogger("stock_video")

PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
HTTP_TIMEOUT = 60.0
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5
MIN_CLIP_DURATION_SECONDS = 6
DEFAULT_MAX_HEIGHT = 1920


def _resolve_key() -> str | None:
    keys_csv = (os.environ.get("PEXELS_API_KEYS") or "").strip()
    if keys_csv:
        keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        if keys:
            return random.choice(keys)
    key = (os.environ.get("PEXELS_API_KEY") or "").strip()
    return key or None


def _pick_best_portrait_file(clip: dict, *, max_height: int = DEFAULT_MAX_HEIGHT) -> dict | None:
    files = [
        f for f in (clip.get("video_files") or [])
        if (f.get("file_type") or "").startswith("video/") and f.get("link")
    ]
    if not files:
        return None
    portrait = [
        f for f in files
        if (f.get("height") or 0) > (f.get("width") or 0) * 1.15
        and (f.get("height") or 0) <= max_height
        and f.get("quality") in ("hd", "sd")
    ]
    portrait.sort(key=lambda f: (f.get("height") or 0))
    return portrait[-1] if portrait else files[0]


async def _download(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            async for chunk in resp.aiter_bytes(64 * 1024):
                fh.write(chunk)


async def _strip_audio(src: Path, dst: Path) -> None:
    """Strip audio from src MP4 to dst. Tries -c copy first, re-encode fallback."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src), "-an", "-c:v", "copy", str(dst),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            str(dst),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg strip-audio failed: {err.decode(errors='replace')[-300:]}"
            )


_SCENE_STYLE_HINTS_TO_STRIP = (
    "illustration style", "cartoon style", "cinematic style",
    "photorealistic", "photo realistic",
    ", warm lighting", ", soft lighting", ", cozy lighting",
    ", golden hour", ", natural lighting",
    ", cozy", ", warm", ", modern", ", elegant",
)


def scene_prompt_to_pexels_query(scene_prompt: str, *, max_len: int = 60) -> str:
    """Turn a Gemini scene_image_prompt into a short Pexels search query.

    Gemini emits English scene descriptors like
      "modern Berlin office interior with desks and computers, illustration style"
    Pexels treats the query as a bag of keywords, so we strip AI style hints
    that would poison stock-video matching (nothing on Pexels is tagged
    'illustration style'), collapse whitespace, and cap length. Falls back to
    'landscape' when the prompt is empty.
    """
    q = (scene_prompt or "").strip().lower()
    if not q:
        return "landscape"
    for hint in _SCENE_STYLE_HINTS_TO_STRIP:
        q = q.replace(hint, "")
    q = " ".join(q.split()).strip(" ,.;")
    if len(q) > max_len:
        q = q[:max_len].rsplit(" ", 1)[0].strip(" ,.;")
    return q or "landscape"


async def composite_bg(
    hf_output: Path,
    bg_video: Path,
    final_out: Path,
    *,
    dim: float = 0.55,
    chroma_similarity: float = 0.22,
    chroma_blend: float = 0.10,
) -> Path:
    """Composite HF output (with #00ff00 chromakey bg) over the looping bg video
    plus a black dim overlay. HF cannot advance a <video> element under its
    stepped virtual clock, so realtime playback is deferred to ffmpeg here.

    Filter graph:
      1. bg.mp4 looped, scaled+center-cropped to 1080x1920, trimmed to HF duration
      2. black rectangle at opacity `dim` overlaid on bg (readability dim)
      3. HF output green-keyed (color #00ff00, similarity, blend), then overlaid
         on the dimmed bg
      4. Audio from HF output copied to final
    """
    hf_output = hf_output.resolve()
    bg_video = bg_video.resolve()
    final_out = final_out.resolve()

    if not hf_output.exists():
        raise FileNotFoundError(f"hf_output missing: {hf_output}")
    if not bg_video.exists():
        raise FileNotFoundError(f"bg_video missing: {bg_video}")

    filter_complex = (
        f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,setsar=1[bgs];"
        f"[bgs]drawbox=x=0:y=0:w=1080:h=1920:color=black@{dim:.2f}:t=fill[bgd];"
        f"[1:v]chromakey=color=0x00ff00:similarity={chroma_similarity:.2f}:"
        f"blend={chroma_blend:.2f}[fg];"
        f"[bgd][fg]overlay=0:0:shortest=1[out]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-stream_loop", "-1", "-i", str(bg_video),
        "-i", str(hf_output),
        "-filter_complex", filter_complex,
        "-map", "[out]", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        str(final_out),
    ]
    log.info("composite_bg: %s", " ".join(cmd[-8:]))
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"composite_bg failed: {err.decode(errors='replace')[-400:]}"
        )
    return final_out


async def fetch_bg_video(
    query: str,
    dest: Path,
    *,
    per_page: int = 15,
    min_duration: int = MIN_CLIP_DURATION_SECONDS,
    max_height: int = DEFAULT_MAX_HEIGHT,
    api_key: str | None = None,
) -> Path | None:
    """Search Pexels for a portrait 9:16 clip and save (audio-stripped) to dest.

    Returns the on-disk path on success, or None on any failure (no key,
    network error, 4xx, no candidate long enough). Never raises: callers
    fall back to a gradient bg if this returns None.
    """
    key = api_key or _resolve_key()
    if not key:
        log.warning("stock_video: no PEXELS_API_KEY, skipping bg fetch")
        return None

    query = (query or "").strip() or "landscape"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_raw = dest.with_suffix(".raw.mp4")

    params = {"query": query, "orientation": "portrait", "size": "medium", "per_page": per_page}
    headers = {"Authorization": key}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                resp = await client.get(PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params)
                if resp.status_code != 200:
                    log.warning("pexels %s attempt %d -> HTTP %d", query, attempt, resp.status_code)
                    if resp.status_code in (401, 403, 429):
                        return None
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                clips = [c for c in (resp.json().get("videos") or []) if (c.get("duration") or 0) >= min_duration]
                if not clips:
                    log.warning("pexels %s: no clips longer than %ds", query, min_duration)
                    return None
                random.shuffle(clips)
                pick_url = None
                for c in clips:
                    f = _pick_best_portrait_file(c, max_height=max_height)
                    if f and f.get("link"):
                        pick_url = f["link"]
                        break
                if not pick_url:
                    log.warning("pexels %s: no portrait file in candidates", query)
                    return None
                await _download(client, pick_url, tmp_raw)
        except Exception as exc:
            log.warning("pexels %s attempt %d failed: %s", query, attempt, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            return None

        try:
            await _strip_audio(tmp_raw, dest)
        finally:
            tmp_raw.unlink(missing_ok=True)
        log.info("stock_video ok: %s -> %s (%d bytes)", query, dest.name, dest.stat().st_size)
        return dest
    return None
