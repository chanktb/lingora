"""Post finished videos to social platforms.

Phase 3 — Facebook Page upload via Graph API.
TikTok auto-post is planned for a later phase (user uploads manually for now).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

log = logging.getLogger("poster")

GRAPH_API_VERSION = os.environ.get("FB_GRAPH_API_VERSION", "v22.0")
GRAPH_VIDEO_BASE = f"https://graph-video.facebook.com/{GRAPH_API_VERSION}"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


class PosterError(RuntimeError):
    pass


async def post_to_facebook(
    video_path: Path,
    caption: str,
    page_id: str,
    access_token: str,
    *,
    thumbnail_path: Path | None = None,
    timeout: float = 300.0,
) -> dict:
    """Upload a video to a Facebook Page via Graph API.

    thumbnail_path: optional JPG/PNG to use as the video's poster image.
                    Recommended full-resolution (e.g. 1080×1920 for 9:16).
                    If None, Facebook picks an arbitrary frame.

    Returns the response JSON; includes "id" (video post id) on success.
    Raises PosterError on HTTP or FB API error.
    """
    if not video_path.exists():
        raise PosterError(f"video not found: {video_path}")
    size_mb = video_path.stat().st_size / 1_048_576
    log.info(
        "Uploading %s (%.1f MB) to FB Page %s%s",
        video_path.name, size_mb, page_id,
        f" with thumb {thumbnail_path.name}" if thumbnail_path else " (no custom thumb)",
    )

    url = f"{GRAPH_VIDEO_BASE}/{page_id}/videos"
    vf = video_path.open("rb")
    tf = (
        thumbnail_path.open("rb")
        if thumbnail_path and thumbnail_path.exists() else None
    )
    try:
        files: dict = {"source": (video_path.name, vf, "video/mp4")}
        if tf is not None:
            files["thumb"] = (thumbnail_path.name, tf, "image/jpeg")
        data = {
            "access_token": access_token,
            "description": caption,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, files=files, data=data)
    finally:
        vf.close()
        if tf is not None:
            tf.close()

    if r.status_code >= 400:
        try:
            err = r.json().get("error", {})
            msg = err.get("message", r.text[:300])
            etype = err.get("type", "")
            code = err.get("code", "")
            raise PosterError(f"FB {r.status_code} {etype}/{code}: {msg}")
        except ValueError:
            raise PosterError(f"FB HTTP {r.status_code}: {r.text[:300]}")

    body = r.json()
    if "id" not in body:
        raise PosterError(f"FB response missing 'id': {body}")
    log.info("FB upload OK: id=%s", body["id"])
    return body


def fb_post_url(video_id: str) -> str:
    """Public URL of an uploaded video post."""
    return f"https://www.facebook.com/watch/?v={video_id}"


def fb_photo_url(page_id: str, photo_id: str) -> str:
    """Public URL of an uploaded photo post."""
    return f"https://www.facebook.com/{page_id}/photos/{photo_id}"


async def _fb_resumable_upload(
    *,
    page_id: str,
    access_token: str,
    video_path: Path,
    container: str,
    finish_extras: dict,
    timeout: float = 300.0,
) -> dict:
    """3-step resumable upload to /video_reels or /video_stories.

    Step 1: POST /{page_id}/{container}?upload_phase=start → video_id + upload_url
    Step 2: POST https://rupload.facebook.com/video-upload/v25.0/{video_id}
            with Authorization: OAuth + offset + file_size headers, body = binary
    Step 3: POST /{page_id}/{container}?upload_phase=finish&video_id=... → publish

    container = "video_reels" | "video_stories"
    finish_extras = extra form fields passed to the finish call (e.g.
                    {"video_state": "PUBLISHED", "description": "..."} for Reels)

    Raises PosterError on any non-2xx in steps 1-3. Caller catches and decides
    whether to fall back to manual button or skip.
    """
    if not video_path.exists():
        raise PosterError(f"video not found: {video_path}")
    video_size = video_path.stat().st_size
    size_mb = video_size / 1_048_576
    log.info(
        "FB %s upload start: %s (%.1f MB) → page %s",
        container, video_path.name, size_mb, page_id,
    )

    init_url = f"{GRAPH_BASE}/{page_id}/{container}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Step 1: start
        r1 = await client.post(
            init_url,
            data={"access_token": access_token, "upload_phase": "start"},
        )
        if r1.status_code >= 400:
            raise PosterError(f"FB {container} init {r1.status_code}: {r1.text[:300]}")
        init_body = r1.json()
        video_id = init_body.get("video_id")
        if not video_id:
            raise PosterError(
                f"FB {container} init missing video_id: {init_body}",
            )
        log.info("FB %s init OK: video_id=%s", container, video_id)

        # Step 2: binary upload via rupload
        upload_url = f"https://rupload.facebook.com/video-upload/v25.0/{video_id}"
        with video_path.open("rb") as f:
            video_bytes = f.read()
        r2 = await client.post(
            upload_url,
            content=video_bytes,
            headers={
                "Authorization": f"OAuth {access_token}",
                "offset": "0",
                "file_size": str(video_size),
            },
        )
        if r2.status_code >= 400:
            raise PosterError(
                f"FB {container} rupload {r2.status_code}: {r2.text[:300]}",
            )
        log.info("FB %s rupload OK", container)

        # Step 3: finish + publish
        finish_data = {
            "access_token": access_token,
            "video_id": video_id,
            "upload_phase": "finish",
            **finish_extras,
        }
        r3 = await client.post(init_url, data=finish_data)
        if r3.status_code >= 400:
            raise PosterError(
                f"FB {container} finish {r3.status_code}: {r3.text[:300]}",
            )
        finish_body = r3.json()
        log.info("FB %s finish OK: %s", container, finish_body)

    return {"video_id": video_id, "container": container, **finish_body}


async def post_reel_to_facebook(
    video_path: Path,
    caption: str,
    page_id: str,
    access_token: str,
    *,
    cover_timestamp_ms: int = 1500,
    timeout: float = 300.0,
) -> dict:
    """Publish a video to a Page's Reels tab.

    Specs: 9:16, 1080×1920 ideal, 3-90s, max 1GB, H.264 + AAC. Our renders
    fit comfortably. Rate limit: 30 reels per 24h per Page.

    cover_timestamp_ms tells FB which video frame to use as the Reel's cover
    image. Default 1500ms (1.5s into video) — by then every layout has its
    native title fully animated in. Pass 0 to disable and let FB pick.

    Returns dict with `video_id` (FB Reel id). Build the public URL via
    `fb_reel_url(video_id)`. Raises PosterError on failure.
    """
    finish_extras = {
        "video_state": "PUBLISHED",
        "description": (caption or "")[:2200],
    }
    if cover_timestamp_ms > 0:
        finish_extras["video_cover_timestamp_ms"] = str(cover_timestamp_ms)
    return await _fb_resumable_upload(
        page_id=page_id,
        access_token=access_token,
        video_path=video_path,
        container="video_reels",
        finish_extras=finish_extras,
        timeout=timeout,
    )


async def post_story_to_facebook(
    video_path: Path,
    page_id: str,
    access_token: str,
    *,
    timeout: float = 300.0,
) -> dict:
    """Publish a video to a Page as a 24-hour Story.

    Spec: 9:16, max 60s ideal, 1GB max. We do NOT client-trim — if the video
    is too long, FB's API responds with an error and the caller logs it; the
    Reel upload (which allows 90s) still succeeds in that case.

    Stories don't carry a description/caption (the UI text is limited to a
    sticker overlay set in the FB app), so caption argument is intentionally
    absent.
    """
    return await _fb_resumable_upload(
        page_id=page_id,
        access_token=access_token,
        video_path=video_path,
        container="video_stories",
        finish_extras={},
        timeout=timeout,
    )


def fb_reel_url(video_id: str) -> str:
    """Public URL of a published Page Reel."""
    return f"https://www.facebook.com/reel/{video_id}"


async def post_photo_to_facebook(
    image_path: Path,
    caption: str,
    page_id: str,
    access_token: str,
) -> dict:
    """Upload single PNG/JPG to FB Page via /photos endpoint.

    Returns FB API response with 'id' and 'post_id' fields.
    """
    import httpx
    url = f"https://graph.facebook.com/v22.0/{page_id}/photos"
    size_mb = image_path.stat().st_size / (1024 * 1024)
    log.info("Uploading %s (%.2f MB) to FB Page %s as PHOTO",
             image_path.name, size_mb, page_id)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        with image_path.open("rb") as f:
            files = {"source": (image_path.name, f, "image/png")}
            data = {
                "caption": caption,
                "access_token": access_token,
                "published": "true",
            }
            resp = await client.post(url, data=data, files=files)

    if resp.status_code >= 400:
        raise PosterError(f"FB photo upload failed {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    if "id" not in body:
        raise PosterError(f"FB photo response missing 'id': {body}")
    log.info("FB photo upload OK: id=%s post_id=%s",
             body.get("id"), body.get("post_id", ""))
    return body


def build_phrases_comment_from_manifest(manifest: dict, max_chars: int = 7500) -> str:
    """Build FB comment listing ALL phrases from manifest dict.

    Works for any phrases-layout video — main.py reads manifest after upload.
    """
    intro = manifest.get("intro", {})
    display = intro.get("display", "").strip()
    phrases = manifest.get("phrases", [])
    lines: list[str] = []
    if display:
        lines += [f"📚 {display}", ""]
    for i, p in enumerate(phrases, start=1):
        target = (p.get("target") or "").strip()
        pron = (p.get("pronunciation") or "").strip()
        native = (p.get("native") or "").strip()
        if not target:
            continue
        lines.append(f"{i}. {target}")
        if pron:
            lines.append(f"   [{pron}]")
        if native:
            lines.append(f"   → {native}")
        lines.append("")
    lines.append("💾 Save & follow để học mỗi ngày!")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def build_phrases_comment(content, max_chars: int = 7500) -> str:
    """Build FB comment from a GeneratedContent object (used in auto_post.py)."""
    lines = [f"📚 {content.intro_display}", ""]
    for i, p in enumerate(content.phrases, start=1):
        lines.append(f"{i}. {p.target}")
        pron = getattr(p, "pronunciation", "")
        if pron:
            lines.append(f"   [{pron}]")
        if p.native:
            lines.append(f"   → {p.native}")
        lines.append("")
    lines.append("💾 Save & follow để học mỗi ngày!")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


async def telegram_notify(
    bot_token: str,
    chat_id: int | str,
    text: str,
    *,
    parse_mode: str = "HTML",
    disable_web_preview: bool = False,
    reply_markup: dict | None = None,
    timeout: float = 15.0,
) -> dict:
    """Send a notification message via Telegram Bot API.

    Used by auto_post.py to report each successful/failed post to the channel owner.
    chat_id = first user_id from TELEGRAM_ALLOWED_USER_IDS.

    reply_markup: optional inline-keyboard dict, e.g.
      {"inline_keyboard": [[{"text":"📘 FB", "callback_data":"postfb:JOB"}]]}
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload: dict = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_web_preview,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
    if r.status_code >= 400:
        log.warning("Telegram notify %s: %s", r.status_code, r.text[:200])
        return {}
    return r.json()


async def telegram_send_video(
    bot_token: str,
    chat_id: int | str,
    video_path: Path,
    *,
    caption: str = "",
    parse_mode: str = "HTML",
    reply_markup: dict | None = None,
    thumbnail_path: Path | None = None,
    timeout: float = 120.0,
) -> dict:
    """Send a video file to Telegram with optional caption + inline buttons.

    Used by auto_post.py "manual fallback" mode — when neither FB nor TikTok
    auto-post is enabled, the freshly rendered MP4 is delivered to the CEO's
    Telegram group with [📘 FB] [🎵 TikTok] buttons so they can publish on
    demand without dropping into the bot DM.

    reply_markup format matches Telegram Bot API:
      {"inline_keyboard": [[{"text":"...","callback_data":"postfb:JOB"}], ...]}
    """
    if not video_path.exists():
        raise PosterError(f"video not found for Telegram send: {video_path}")
    url = f"https://api.telegram.org/bot{bot_token}/sendVideo"

    import json as _json
    data: dict = {
        "chat_id": str(chat_id),
        "caption": (caption or "")[:1024],
        "parse_mode": parse_mode,
        "supports_streaming": "true",
    }
    if reply_markup:
        data["reply_markup"] = _json.dumps(reply_markup)

    vf = video_path.open("rb")
    tf = thumbnail_path.open("rb") if thumbnail_path and thumbnail_path.exists() else None
    try:
        files: dict = {"video": (video_path.name, vf, "video/mp4")}
        if tf is not None:
            files["thumbnail"] = (thumbnail_path.name, tf, "image/jpeg")
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, data=data, files=files)
    finally:
        vf.close()
        if tf is not None:
            tf.close()

    if r.status_code >= 400:
        log.warning("Telegram sendVideo %s: %s", r.status_code, r.text[:300])
        return {}
    return r.json()


async def fb_comment_on_video(
    video_id: str,
    message: str,
    access_token: str,
    *,
    timeout: float = 30.0,
) -> dict:
    """Post a comment on an uploaded Facebook video.

    Used to pin the quiz answer as the first comment — boosts engagement
    by making viewers tap "View comments" to see the answer.

    Returns the comment object; includes "id" on success.
    """
    url = f"{GRAPH_BASE}/{video_id}/comments"
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, data={
            "access_token": access_token,
            "message": message,
        })
    if r.status_code >= 400:
        try:
            err = r.json().get("error", {})
            raise PosterError(f"FB comment {r.status_code}: {err.get('message', r.text[:300])}")
        except ValueError:
            raise PosterError(f"FB comment HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    log.info("FB pinned comment OK on video %s: id=%s", video_id, body.get("id"))
    return body
