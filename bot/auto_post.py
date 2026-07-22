"""Auto-post entry point.

Run by systemd timer (ktb-auto-post@<channel>.timer) every ~2 min.
Self-throttles based on AUTO_POST_INTERVAL env var.

Pipeline:
  1. Throttle check
  2. Pick topic + layout (xen kẽ phrases ↔ quiz cho language niche)
  3. Gemini sinh content
  4. Edge TTS audio
  5. HyperFrames render
  6. Auto-post FB Page
  7. (quiz) Auto-pin đáp án trong comment đầu
  8. Save state
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Resolve channel + load .env
_cd = os.environ.get("CHANNEL_DIR")
if not _cd:
    print("CHANNEL_DIR env var required", file=sys.stderr); sys.exit(2)
CHANNEL_DIR = Path(_cd).resolve()
if not CHANNEL_DIR.is_dir():
    print(f"CHANNEL_DIR not a directory: {CHANNEL_DIR}", file=sys.stderr); sys.exit(2)

# override=True ensures channel-specific values (DEFAULT_TARGET_LANG, FB_PAGE_ID, THEME_*, etc.)
# win over any inherited env from the parent process. Critical for cross-channel subprocess
# spawned by /auto <channel> in bot/main.py — parent bot has its own channel env loaded.
load_dotenv(CHANNEL_DIR / ".env", override=True)

# bot/ on path for imports
sys.path.insert(0, str(Path(__file__).parent))
import composer       # noqa: E402
import generator      # noqa: E402
import image_gen      # noqa: E402
import poster         # noqa: E402
import renderer       # noqa: E402
import topic_picker   # noqa: E402
import tts            # noqa: E402


# ─── Dialogue avatar/voice gender consistency ──────────────────────────
# CEO bug 2026-07-02 (zh channel): dialogue nhân vật NAM nhưng giọng NỮ.
# Giọng chọn theo tên/voice_gender, còn avatar sinh từ image_prompt — 2 nguồn
# tách rời nên lệch được khi Gemini viết image_prompt không khớp gender. Sau khi
# CHỐT gender mỗi nhân vật, ép luôn gender word đầu image_prompt để avatar == giọng.
_GENDER_NOUN = {"male": "man", "female": "woman"}


def _sync_image_prompt_gender(image_prompt: str, gender: str) -> str:
    """Force the portrait's leading gender word to match the chosen voice gender."""
    noun = _GENDER_NOUN.get(gender, "woman")
    opp = "man" if noun == "woman" else "woman"
    p = image_prompt or ""
    # Flip the FIRST opposite-gender word (\bman\b won't match inside "woman").
    new, n = re.subn(rf"\b{opp}\b", noun, p, count=1)
    if n == 0 and not re.search(rf"\b{noun}\b", new):
        # No explicit gender word → prepend one so FLUX can't pick at random.
        new = f"a young {noun}, {p}" if p else f"a young {noun}"
    return new

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(f"auto-post[{CHANNEL_DIR.name}]")

JOBS_DIR = CHANNEL_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = CHANNEL_DIR / "auto_post_state.json"

# CEO 2026-07-22: layouts that use Pexels stock video as bg (composited post-HF).
BG_VIDEO_LAYOUTS = {"guess_word", "phrases", "quiz", "quiz_reverse", "fill_blank", "conjugation"}


async def _maybe_fetch_bg_video(scene_prompt: str, job_dir: Path, label: str) -> Path | None:
    """Fetch a Pexels bg video from a Gemini scene_image_prompt. Never raises."""
    import stock_video  # type: ignore  # noqa: E402
    query = stock_video.scene_prompt_to_pexels_query(scene_prompt or "")
    if not query or query == "landscape":
        log.info("%s: skip Pexels bg (empty scene prompt)", label)
        return None
    bg_dir = job_dir / "stock_bg"
    bg_dir.mkdir(parents=True, exist_ok=True)
    log.info("%s: Pexels bg query=%r", label, query)
    try:
        return await stock_video.fetch_bg_video(query, bg_dir / "bg.mp4")
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: Pexels fetch failed: %s", label, exc)
        return None

# Rate-limit the "CF quota exhausted → skipping turn" heads-up so we don't spam
# the Telegram group every timer tick while quota is globally out.
QUOTA_SKIP_NOTIFY_FILE = CHANNEL_DIR / ".quota_skip_notify_ts"
QUOTA_SKIP_NOTIFY_COOLDOWN_S = 6 * 3600


def _notify_chat_id() -> int | None:
    """Pick recipient for auto-post reports.

    Priority:
      1. TELEGRAM_NOTIFY_CHAT_ID — explicit override (group ID, negative for groups)
      2. First user ID from TELEGRAM_ALLOWED_USER_IDS (fallback to DM)
    """
    override = (os.environ.get("TELEGRAM_NOTIFY_CHAT_ID") or "").strip()
    if override:
        try:
            return int(override)  # supports negative (group) and positive (user) IDs
        except ValueError:
            pass
    raw = (os.environ.get("TELEGRAM_ALLOWED_USER_IDS") or "").strip()
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.lstrip("-").isdigit():
            return int(tok)
    return None


async def notify(text: str, *, reply_markup: dict | None = None) -> None:
    """Best-effort Telegram text report. Silent on failure."""
    bot_token = os.environ.get("TELEGRAM_NOTIFY_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = _notify_chat_id()
    if not bot_token or chat_id is None:
        log.info("Notify skipped (no TELEGRAM_BOT_TOKEN or chat_id)")
        return
    try:
        await poster.telegram_notify(
            bot_token, chat_id, text, reply_markup=reply_markup,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Notify failed: %s", exc)


async def notify_with_video(
    text: str,
    video_path: Path,
    *,
    reply_markup: dict | None = None,
    thumbnail_path: Path | None = None,
) -> None:
    """Send the rendered MP4 to the Telegram group with optional buttons.

    Used in manual mode (when both AUTO_POST_FB_ENABLED and
    AUTO_POST_TIKTOK_ENABLED are false) so the CEO can publish on demand
    with one tap. Falls back silently if the bot token is missing.
    """
    bot_token = os.environ.get("TELEGRAM_NOTIFY_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = _notify_chat_id()
    if not bot_token or chat_id is None:
        log.info("notify_with_video skipped (no TELEGRAM_BOT_TOKEN or chat_id)")
        return
    try:
        await poster.telegram_send_video(
            bot_token, chat_id, video_path,
            caption=text,
            reply_markup=reply_markup,
            thumbnail_path=thumbnail_path,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("notify_with_video failed: %s", exc)


def _manual_buttons(job_id: str, *, fb_enabled: bool, tiktok_enabled: bool) -> dict:
    """Inline-keyboard payload for the manual-mode auto-post Telegram report.

    Includes the button for any platform whose AUTO_POST_*_ENABLED is false
    (so the CEO can click to publish). Skips buttons for already-auto-posted
    platforms to avoid double-posts.
    """
    row: list[dict] = []
    if not fb_enabled:
        row.append({"text": "📘 Đăng FB", "callback_data": f"postfb:{job_id}"})
    if not tiktok_enabled:
        row.append({"text": "🎵 Đăng TikTok", "callback_data": f"posttiktok:{job_id}"})
    if not row:
        return {}
    return {"inline_keyboard": [row]}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("true", "1", "yes", "on")


# ───────────────────── helpers ─────────────────────


def _norm_rate(r: str) -> str:
    r = (r or "").strip()
    if r and not r.startswith(("+", "-")):
        r = "+" + r
    return r or "+0%"


def parse_interval(s: str) -> int:
    """'5min' → 300, '1h' → 3600, '30s' → 30, '30' → 30 (seconds)."""
    s = (s or "").strip().lower()
    if s.endswith("h"):
        return int(float(s[:-1]) * 3600)
    if s.endswith("min"):
        return int(float(s[:-3]) * 60)
    if s.endswith("m"):
        return int(float(s[:-1]) * 60)
    if s.endswith("s"):
        return int(float(s[:-1]))
    return int(float(s))


def parse_schedule(s: str) -> dict[int, int]:
    """Parse 'H:min,H:min,...' → {hour: interval_min}. Hours absent = pause.

    Example: "7:30,8:30,12:30,22:60" means:
      - At hour 7-7:59 VN: interval 30 min (max 2 posts in that hour)
      - At hour 12: interval 30 min
      - At hour 22: interval 60 min (max 1 post in that hour)
      - All other hours: no post.
    """
    out: dict[int, int] = {}
    for chunk in (s or "").split(","):
        chunk = chunk.strip()
        if ":" not in chunk:
            continue
        try:
            h_str, v_str = chunk.split(":", 1)
            out[int(h_str)] = int(v_str)
        except ValueError:
            continue
    return out


def current_local_hour() -> int:
    """Hour-of-day (0-23) in AUTO_POST_TZ (default Asia/Ho_Chi_Minh)."""
    tz_name = os.environ.get("AUTO_POST_TZ", "Asia/Ho_Chi_Minh")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    return datetime.now(tz).hour


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Corrupt state file — starting fresh")
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def _maybe_notify_quota_skip(topic: str = "") -> None:
    """Rate-limited heads-up that this channel is skipping auto-post turns
    because all CF accounts are quota-exhausted. Fires at most once per
    QUOTA_SKIP_NOTIFY_COOLDOWN_S so the group isn't spammed every timer tick.
    """
    now = int(time.time())
    try:
        last = int(QUOTA_SKIP_NOTIFY_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        last = 0
    if now - last < QUOTA_SKIP_NOTIFY_COOLDOWN_S:
        return
    try:
        QUOTA_SKIP_NOTIFY_FILE.write_text(str(now), encoding="utf-8")
    except OSError:
        pass
    await notify(
        f"⏸️ <b>Tạm bỏ lượt đăng — Cloudflare AI hết quota</b> [{CHANNEL_DIR.name}]\n"
        f"Tất cả tài khoản CF Workers AI đã cạn neuron hôm nay nên bot bỏ qua các "
        f"lượt cần tạo ảnh. Sẽ tự đăng lại khi quota reset hoặc thêm account mới.\n"
        f"<i>(thông báo tối đa 6h/lần)</i>"
    )


def _ffmpeg_bin() -> str:
    base = os.environ.get("FFMPEG_BIN", "")
    if base:
        cand = Path(base) / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
        if cand.exists():
            return str(cand)
    return shutil.which("ffmpeg") or "ffmpeg"


def extract_thumbnail(video: Path, *, at_seconds: float = 2.0) -> Path | None:
    """Grab a full-resolution frame as JPG for FB thumbnail."""
    thumb = video.with_suffix(".thumb-full.jpg")
    try:
        subprocess.run(
            [_ffmpeg_bin(), "-y", "-loglevel", "error",
             "-ss", f"{at_seconds:.2f}", "-i", str(video),
             "-vframes", "1", "-q:v", "4", str(thumb)],
            check=True, capture_output=True, text=True, timeout=20,
        )
        return thumb if thumb.exists() else None
    except Exception as exc:  # noqa: BLE001
        log.warning("thumbnail extract failed: %s", exc)
        return None


def prepend_thumb_freeze(
    video_path: Path,
    *,
    capture_at_ms: int = 2000,
    hold_duration_s: float = 1.0,
) -> bool:
    """Prepend a static frame extracted from `capture_at_ms` to the video.

    Workflow (CEO's idea — lingora 2026-06-12):
      1. Pick a known-good frame from the rendered video (e.g. 2.0s in, when
         the native title is fully animated and the per-round content is set).
      2. Build a brief static intro video from that frame.
      3. Concatenate [static intro] + [original] and replace output.mp4.

    Why: FB Reels / Stories / TikTok auto-pick a thumbnail from the first
    second of the video. Without this, the thumbnail can land on a fade-in
    frame (dark) or a mid-animation frame (blurry). With this, t=0 of the
    final video is guaranteed to be a clean frame WITH the title rendered.

    Re-encodes via libx264 (veryfast, crf 23) so the concat is reliable
    across the original-vs-intro codec boundary. Adds ~3-8 seconds per video
    on the VPS.

    Returns True on success; False if any step fails (caller keeps the
    original video and continues — the original is still posted as-is).
    """
    if not video_path.exists():
        log.warning("prepend_thumb_freeze: video not found: %s", video_path)
        return False

    work = video_path.parent / "_thumb_work"
    work.mkdir(exist_ok=True)
    still = work / "still.png"
    intro = work / "intro.mp4"
    final = work / "final.mp4"
    concat_list = work / "list.txt"

    ffmpeg = _ffmpeg_bin()
    ffprobe = _ffprobe_bin()

    try:
        # 1. Extract still frame at capture_at_ms.
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-ss", f"{capture_at_ms / 1000:.2f}",
             "-i", str(video_path),
             "-vframes", "1",
             str(still)],
            check=True, capture_output=True, text=True, timeout=20,
        )

        # 2. Probe original video frame rate so the intro matches.
        probe = subprocess.run(
            [ffprobe, "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(video_path)],
            capture_output=True, text=True, check=True, timeout=10,
        )
        fr = (probe.stdout.strip().splitlines() or ["30/1"])[0]

        # 3. Generate a static MP4 from the still + silent mono audio at
        #    48 kHz to MATCH HyperFrames' MP4 audio output (verified via
        #    ffprobe on raw HF render: sample_rate=48000 mono).
        #
        # CEO drift bug 2026-06-18 (FINAL ROOT CAUSE): previous code used
        # 44.1 kHz stereo intro audio, then concat demuxer merged with HF's
        # 48 kHz mono audio. The concat demuxer does NOT resample — it
        # treats 48 kHz samples as 44.1 kHz, playing them 8.98 % slower
        # (48000/44100 = 1.0898). For a 42 s lesson that's ~3.8 s of
        # cumulative drift by the end — exactly matching the "phrase 8
        # scene shown but phrase 7 voice still playing" observation.
        # telegram-video-bot doesn't have this thumb_freeze step at all,
        # which is why CEO never saw drift there.
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-loop", "1",
             "-framerate", fr,
             "-t", f"{hold_duration_s:.2f}",
             "-i", str(still),
             "-f", "lavfi",
             "-t", f"{hold_duration_s:.2f}",
             "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "1",
             "-shortest",
             str(intro)],
            check=True, capture_output=True, text=True, timeout=45,
        )

        # 4. Concat using the ffmpeg concat FILTER (not the demuxer). The
        # filter decodes both streams + resamples them through the same
        # pipeline before encoding, so audio timing is preserved even when
        # the two inputs have different sample rates / channel layouts.
        # The concat demuxer would just copy samples verbatim — that's how
        # the 9 % drift sneaks in.
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-i", str(intro),
             "-i", str(video_path),
             "-filter_complex",
             "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[outv][outa]",
             "-map", "[outv]",
             "-map", "[outa]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
             "-movflags", "+faststart",
             str(final)],
            check=True, capture_output=True, text=True, timeout=180,
        )

        # 5. Replace original.
        shutil.move(str(final), str(video_path))
        log.info(
            "prepend_thumb_freeze OK: held %.2fs at start (capture=%dms)",
            hold_duration_s, capture_at_ms,
        )
        return True

    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-300:] if hasattr(exc, "stderr") else ""
        log.warning("prepend_thumb_freeze ffmpeg failed: %s | %s", exc, tail)
        return False
    except Exception as exc:  # noqa: BLE001
        log.warning("prepend_thumb_freeze failed: %s", exc)
        return False
    finally:
        # Best-effort cleanup of intermediates
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass


def _fill_blank_narration_lines(
    content,
    native_lang: str,
    target_lang_name: str,
) -> list[tuple[str, str]]:
    """Build per-segment scripts for fill_blank narration.

    Returns a list of (segment_name, text) tuples — intro + one segment per
    option (A/B/C) read in the TARGET language + outro. Segment names match
    the timing keys used downstream so the highlight overlay can pin a yellow
    ring to each option box while its voice plays.

    Each option text is "A. <option>" / "B. <option>" / "C. <option>" so the
    viewer hears the label and the word together.
    """
    options = list(getattr(content, "options", []) or [])[:3]
    labels = ["A", "B", "C", "D"]

    if native_lang == "en":
        intro = (
            "Quick quiz — fill in the blank. Read the sentence, then pick "
            "the answer that fits."
        )
        outro = "Which one is right? Drop your answer below!"
    elif native_lang == "ko":
        intro = "퀴즈 시간! 빈칸에 들어갈 단어를 골라보세요."
        outro = "정답이 뭐예요? 댓글로 알려주세요!"
    elif native_lang == "ja":
        intro = "今日のクイズ!空欄に入る単語を選んでね。"
        outro = "答えはどれ?コメントで教えて!"
    else:  # vi
        intro = (
            "Đố vui ngắn — điền vào chỗ trống. Đọc câu rồi chọn từ phù hợp nhé."
        )
        outro = "Đáp án của bạn là gì? Comment ngay bên dưới nhé!"

    out: list[tuple[str, str]] = [("static_intro", intro)]
    for i, opt in enumerate(options):
        out.append((f"static_opt_{labels[i]}", f"{labels[i]}. {opt}"))
    out.append(("static_outro", outro))
    return out


def overlay_option_highlights(
    video_path: Path,
    *,
    option_windows: list[tuple[float, float]],
    time_offset_s: float = 0.0,
    box_y: int = 940,
    box_h: int = 290,
    thickness: int = 12,
    color: str = "yellow@0.95",
) -> bool:
    """Draw a yellow rectangle around each option for its narration window.

    fill_blank renders 3 option chips horizontally centered. The visual y-axis
    landing (~y=950-1220 in the 1920 frame) is BELOW the template-declared
    top:670 — the HyperFrames render and the scene image push the chips
    down. Coords below match the empirically observed render at zoom≈1.0.

    option_windows = [(start_A, end_A), ...] in the AUDIO timeline.
    time_offset_s = seconds added to each window (use when overlay runs AFTER
                    a video-prepend step). 0 when overlay runs first.
    """
    if not video_path.exists() or not option_windows:
        return False

    ffmpeg = _ffmpeg_bin()
    work = video_path.parent / "_opt_overlay"
    work.mkdir(exist_ok=True)
    final = work / "final.mp4"

    # Empirical per-cell layout — 3 boxes ~220 wide with ~110 gap, total
    # ~880 centered. Slight overshoot on width so the ring hugs the chip
    # even when option text varies in length.
    BOX_W = 240
    BOX_GAP = 90
    canvas_w = 1080
    total_w = len(option_windows) * BOX_W + (len(option_windows) - 1) * BOX_GAP
    x_left_first = (canvas_w - total_w) // 2

    boxes = []
    for i, (t_start, t_end) in enumerate(option_windows):
        x = x_left_first + i * (BOX_W + BOX_GAP)
        t0 = t_start + time_offset_s
        t1 = t_end + time_offset_s
        boxes.append(
            f"drawbox=x={x}:y={box_y}:w={BOX_W}:h={box_h}:"
            f"color={color}:t={thickness}:"
            f"enable='between(t,{t0:.2f},{t1:.2f})'"
        )
    filter_chain = ",".join(boxes)

    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-i", str(video_path),
             "-vf", filter_chain,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
             "-pix_fmt", "yuv420p",
             "-c:a", "copy",
             "-movflags", "+faststart",
             str(final)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        shutil.move(str(final), str(video_path))
        log.info(
            "overlay_option_highlights OK (n=%d windows)", len(option_windows),
        )
        return True
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-300:] if hasattr(exc, "stderr") else ""
        log.warning("overlay_option_highlights failed: %s | %s", exc, tail)
        return False
    finally:
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass


def extract_full_frame_png_raw(video: Path, *, at_seconds: float = 0.8) -> Path | None:
    """Extract a single 9:16 frame as PNG — NO crop to 4:5.

    Sibling of `extract_full_frame_png` (which crops to 1080×1350 for the FB
    /photos feed). This raw variant is used when posting a static layout as
    a Reel: we want to keep the original 9:16 aspect so the Reels thumbnail
    isn't distorted.
    """
    out = video.with_suffix(".raw9x16.png")
    try:
        subprocess.run(
            [_ffmpeg_bin(), "-y", "-loglevel", "error",
             "-ss", f"{at_seconds:.2f}", "-i", str(video),
             "-vframes", "1", str(out)],
            check=True, capture_output=True, text=True, timeout=20,
        )
        return out if out.exists() else None
    except Exception as exc:  # noqa: BLE001
        log.warning("raw 9:16 PNG extract failed: %s", exc)
        return None


def _static_narration_script(
    layout_type: str,
    content,
    native_lang: str,
    target_lang_name: str,
) -> tuple[str, str]:
    """Return (intro_text, outro_text) for a static layout's TTS narration.

    Used by vocab_table and compare (intro + outro only).
    For fill_blank, see `_fill_blank_narration_clips()` which returns a per-
    option script that lets the highlight ring follow the voice.
    """
    short = (getattr(content, "short_title", "") or "").strip()
    if native_lang == "en":
        if layout_type == "vocab_table":
            intro = (
                f"Master these {target_lang_name} words for {short.lower() or 'today'}. "
                f"Tap save so you can come back."
            )
            outro = f"Hit save and follow for more daily {target_lang_name}!"
        elif layout_type == "fill_blank":
            intro = (
                "Quick quiz — can you spot the right word? "
                "Read the sentence and pick A, B or C."
            )
            outro = "What's your answer? Drop it in the comments!"
        else:  # compare
            intro = (
                f"Sound smoother in {target_lang_name}. "
                f"Eight swaps that make you sound like a native."
            )
            outro = f"Save this for later and follow for more {target_lang_name} tips!"
    elif native_lang == "ko":
        if layout_type == "vocab_table":
            intro = f"{target_lang_name} 어휘 {short or '오늘의 단어'}, 저장해두고 매일 봐요!"
            outro = "좋아요와 팔로우로 매일 한 단어!"
        elif layout_type == "fill_blank":
            intro = "이 빈칸에 들어갈 단어는? 정답을 골라보세요."
            outro = "정답이 뭐예요? 댓글로 알려주세요!"
        else:
            intro = f"더 자연스러운 {target_lang_name} 표현 — 이렇게 바꿔보세요."
            outro = "저장하고 팔로우 부탁드려요!"
    elif native_lang == "ja":
        if layout_type == "vocab_table":
            intro = f"{target_lang_name}の{short or '今日の単語'}を一気に覚えよう!"
            outro = "いいねとフォローで毎日学習!"
        elif layout_type == "fill_blank":
            intro = "空欄に入る言葉は?コメントで答えてね。"
            outro = "あなたの答えは?コメントで教えて!"
        else:
            intro = f"もっと自然な{target_lang_name} — 8つの言い換え。"
            outro = "保存していいね&フォローお願いします!"
    else:  # vi default
        # NOTE: target_lang_name là TÊN ngôn ngữ trần ("Đức", "Trung", "Nhật",
        # "Hàn", "Pháp"...) — trong tiếng Việt phải kèm tiền tố "tiếng" để câu
        # nghe tự nhiên. Voice TTS đọc "tiếng Đức" → "tiếng-đức" (đúng), còn
        # nếu để nguyên target_lang_name là "Đức" thì câu thành "học Đức mỗi
        # ngày" — sai ngữ pháp (CEO 2026-06-29).
        if layout_type == "vocab_table":
            intro = (
                f"Học ngay tiếng {target_lang_name} với chủ đề {short.lower() or 'hôm nay'}. "
                f"Lưu lại để xem mỗi ngày!"
            )
            outro = f"Hãy like và follow để học tiếng {target_lang_name} mỗi ngày nhé!"
        elif layout_type == "fill_blank":
            intro = "Điền vào chỗ trống — bạn chọn đáp án nào? Đọc kỹ câu và chọn A, B hoặc C nhé."
            outro = "Đáp án của bạn là gì? Comment ngay bên dưới!"
        else:  # compare
            intro = (
                f"Nói tiếng {target_lang_name} sang chảnh hơn — "
                f"8 cách diễn đạt khiến bạn nghe như người bản xứ."
            )
            outro = f"Lưu lại và follow để học tiếng {target_lang_name} mỗi ngày!"
    return intro, outro


def mux_audio_into_video(
    video_path: Path,
    audio_segments: list[Path],
    *,
    silence_between_s: float = 0.6,
    silence_lead_s: float = 0.3,
    silence_tail_s: float = 0.4,
) -> bool:
    """Concatenate audio_segments with silence padding and mux onto video.

    Workflow:
      1. ffmpeg builds combined audio: silence(lead) + seg1 + silence(gap) +
         seg2 + ... + silence(tail). Total duration target = video duration.
      2. ffmpeg muxes the combined audio onto the video (replaces existing
         audio track which is silent for Ken-Burns videos).

    Returns True on success; False on any failure (caller keeps silent video).
    """
    if not video_path.exists() or not audio_segments:
        return False

    ffmpeg = _ffmpeg_bin()
    work = video_path.parent / "_audio_mux"
    work.mkdir(exist_ok=True)
    # WAV PCM matches tts._strip_silence_to_wav output (mono, 44100 Hz, s16le).
    # Concat demuxer requires identical codec/sample-rate/channels across inputs,
    # so silence pads MUST match TTS format exactly.
    combined = work / "combined.wav"
    final = work / "final.mp4"

    def _make_silence(out_path: Path, duration_s: float) -> None:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "lavfi", "-t", f"{duration_s:.2f}",
             "-i", "anullsrc=channel_layout=mono:sample_rate=44100",
             "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1", str(out_path)],
            check=True, capture_output=True, text=True, timeout=20,
        )

    try:
        parts: list[Path] = []
        if silence_lead_s > 0:
            lead = work / "_lead.wav"
            _make_silence(lead, silence_lead_s)
            parts.append(lead)
        for i, seg in enumerate(audio_segments):
            parts.append(seg)
            if i < len(audio_segments) - 1 and silence_between_s > 0:
                gap = work / f"_gap_{i}.wav"
                _make_silence(gap, silence_between_s)
                parts.append(gap)
        if silence_tail_s > 0:
            tail = work / "_tail.wav"
            _make_silence(tail, silence_tail_s)
            parts.append(tail)

        list_file = work / "list.txt"
        list_file.write_text(
            "\n".join(f"file '{p.resolve().as_posix()}'" for p in parts) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0",
             "-i", str(list_file),
             "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
             str(combined)],
            check=True, capture_output=True, text=True, timeout=60,
        )

        # Mux audio onto video (keep video, replace audio)
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-i", str(video_path),
             "-i", str(combined),
             "-map", "0:v:0",
             "-map", "1:a:0",
             "-c:v", "copy",
             "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
             "-shortest",
             "-movflags", "+faststart",
             str(final)],
            check=True, capture_output=True, text=True, timeout=60,
        )

        shutil.move(str(final), str(video_path))
        log.info("mux_audio_into_video OK (segments=%d)", len(audio_segments))
        return True
    except subprocess.CalledProcessError as exc:
        tail_err = (exc.stderr or "")[-300:] if hasattr(exc, "stderr") else ""
        log.warning("mux_audio_into_video failed: %s | %s", exc, tail_err)
        return False
    except Exception as exc:  # noqa: BLE001
        log.warning("mux_audio_into_video error: %s", exc)
        return False
    finally:
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass


def static_to_motion_video(
    image_path: Path,
    output_path: Path,
    *,
    duration: float = 5.0,
    fps: int = 30,
    zoom_amount: float = 0.10,
) -> Path | None:
    """Convert a static PNG into a short MP4 with a Ken Burns zoom for Reels.

    Why: vocab_table / fill_blank / compare layouts currently render as a
    static frame. Posted to FB /photos they reach few people (Photo distribution
    < Reel distribution on the 2026 algorithm). Re-encoded with a subtle
    zoom-in via ffmpeg's `zoompan` filter, the same content becomes a Reel
    eligible for Reels distribution — much higher organic reach for a new
    page building its audience. Once the page has traction, the static
    posters can return (just set STATIC_LAYOUTS_AS_VIDEO=false in the .env).

    zoom_amount = relative scale increase over the clip (0.10 = 110% zoom).
    duration must be ≥ 3.0s (FB Reels minimum).
    """
    if not image_path.exists():
        log.warning("static_to_motion_video: image not found: %s", image_path)
        return None
    duration = max(3.5, float(duration))
    fps = max(24, int(fps))

    # CEO feedback 2026-06-12: drop the Ken-Burns zoom for static→video. The
    # poster content is dense (tables, comparisons, sentence-with-blank);
    # zooming in over the duration cuts off the bottom rows. Keep the frame
    # rock-still and add only a subtle 0.5s fade-in from black at t=0 so
    # there's still some entrance motion — viewers don't pause on a hard cut.
    static_filter = (
        "scale=1080:1920:flags=lanczos,"
        "fade=in:st=0:d=0.5:color=#0a0e27,"
        "format=yuv420p"
    )
    _ = zoom_amount  # retained for API compatibility, no longer used

    ffmpeg = _ffmpeg_bin()
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-loop", "1", "-t", f"{duration:.2f}", "-i", str(image_path),
             "-f", "lavfi", "-t", f"{duration:.2f}",
             "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
             "-vf", static_filter,
             "-r", str(fps),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
             "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
             "-movflags", "+faststart",
             "-shortest",
             str(output_path)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        return output_path if output_path.exists() else None
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-300:] if hasattr(exc, "stderr") else ""
        log.warning("static_to_motion_video ffmpeg failed: %s | %s", exc, tail)
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("static_to_motion_video failed: %s", exc)
        return None


def _cleanup_job_dir(job_dir: Path) -> None:
    """Delete a finished job's working directory to reclaim VPS disk.

    Called ONLY after successful FB upload. A typical phrases job is ~40-50 MB
    (TTS folder + AI scene image + HF artifacts + output.mp4). Across 3 channels
    at 1h/post that's ~3 GB/week — without cleanup the VPS would fill up over
    months.

    The FB post URL is already persisted in `state.json` (last_video_id) and
    Telegram notify, so the local working dir is throwaway. To opt OUT (e.g.
    for debugging), set env `KEEP_JOB_DIR_AFTER_UPLOAD=true`.

    Best-effort: never raises — if the cleanup fails the post still succeeded.
    """
    keep = (os.environ.get("KEEP_JOB_DIR_AFTER_UPLOAD", "") or "").strip().lower()
    if keep in ("true", "1", "yes", "on"):
        log.info("Job dir cleanup SKIPPED (KEEP_JOB_DIR_AFTER_UPLOAD=true)")
        return
    try:
        if not job_dir.exists():
            return
        # Compute size before delete for log visibility
        try:
            total = sum(f.stat().st_size for f in job_dir.rglob("*") if f.is_file())
            size_mb = total / (1024 * 1024)
        except Exception:  # noqa: BLE001
            size_mb = -1.0
        shutil.rmtree(job_dir, ignore_errors=True)
        log.info("🧹 Cleaned up job_dir: %s (freed %.1f MB)", job_dir.name, size_mb)
    except Exception as exc:  # noqa: BLE001
        log.warning("Job dir cleanup failed (non-fatal): %s", exc)


def _ffprobe_bin() -> str:
    """Locate ffprobe binary (same logic as _ffmpeg_bin)."""
    base = os.environ.get("FFMPEG_BIN", "")
    if base:
        cand = Path(base) / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if cand.exists():
            return str(cand)
    return shutil.which("ffprobe") or "ffprobe"


def mp4_has_audio_stream(video: Path) -> bool:
    """Return True iff the MP4 contains at least one audio stream.

    Used to catch the HyperFrames-renders-video-only bug: HF sometimes returns
    exit 0 but skips audio embedding (Chromium audio context issues, race
    conditions on heavy concurrent loads, etc). When that happens the MP4 is
    visually correct but silent — we want to detect this BEFORE uploading to FB.
    """
    try:
        result = subprocess.run(
            [_ffprobe_bin(), "-v", "error",
             "-select_streams", "a",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(video)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        return "audio" in (result.stdout or "")
    except Exception as exc:  # noqa: BLE001
        log.warning("ffprobe audio check failed (%s) — assuming audio OK", exc)
        return True  # don't block upload on probe tool failure


def extract_full_frame_png(video: Path, *, at_seconds: float = 0.8) -> Path | None:
    """Extract a single FRAME as PNG, then crop/normalize to FB-feed-optimal 1080x1350.

    Used by photo-post layouts (vocab_table + fill_blank). Templates already render
    at native 1080x1350 (4:5 — max real estate on mobile feed without center-crop).
    The crop pass below is a DEFENSIVE normalization step that:
      1) re-encodes to ensure exact 1080x1350 (defensive vs any HF dim drift)
      2) center-crops if HF accidentally rendered taller (e.g. 1080x1920)
      3) flattens to sRGB and strips video metadata → clean photo PNG for FB
    """
    raw_out = video.with_suffix(".raw-frame.png")
    final_out = video.with_suffix(".frame.png")
    try:
        # Step 1: extract one frame as PNG (whatever the source dim is)
        subprocess.run(
            [_ffmpeg_bin(), "-y", "-loglevel", "error",
             "-ss", f"{at_seconds:.2f}", "-i", str(video),
             "-vframes", "1", str(raw_out)],
            check=True, capture_output=True, text=True, timeout=20,
        )
        if not raw_out.exists():
            return None
        # Step 2: normalize → exact 1080x1350 (FB feed mobile no-crop optimum).
        # Scale to width=1080 keeping AR, then center-crop 1080x1350.
        # If source is already 1080x1350 → identity. If 1080x1920 → crops top+bottom
        # symmetrically (285 px each side), keeping middle 1350 px.
        subprocess.run(
            [_ffmpeg_bin(), "-y", "-loglevel", "error",
             "-i", str(raw_out),
             "-vf", "scale=1080:-1,crop=1080:1350:0:(ih-1350)/2",
             "-pix_fmt", "rgb24",
             str(final_out)],
            check=True, capture_output=True, text=True, timeout=20,
        )
        # Best-effort cleanup of intermediate raw frame
        try:
            raw_out.unlink()
        except Exception:
            pass
        return final_out if final_out.exists() else None
    except Exception as exc:  # noqa: BLE001
        log.warning("PNG frame extract failed: %s", exc)
        return None


def _strip_hashtags(caption: str) -> str:
    """Remove all #hashtag tokens from a caption + tidy leftover blank lines.

    Applied ONLY on the production auto-post path when CAPTION_HASHTAGS=false
    (per-channel .env). Shortcraft uses a different code path (render_chain, not
    auto_post) so it is unaffected; the open-source build defaults to keeping
    hashtags (env unset → True).
    """
    if not caption:
        return caption
    import re
    out = re.sub(r"#[^\s#]+", "", caption)   # drop hashtag tokens
    out = re.sub(r"[ \t]+\n", "\n", out)      # trailing spaces on lines
    out = re.sub(r"\n{3,}", "\n\n", out)      # collapse blank-line runs
    return out.strip()


# ───────────────────── render lock ─────────────────────

_RENDER_LOCK_PATH = "/tmp/ktb-lingora-render.lock"
_render_lock_fh = None


def _acquire_render_lock(*, blocking: bool) -> bool:
    """Global cross-channel render lock — only ONE channel renders at a time.

    Avoids VPS overload when many channels share a schedule slot. Held until the
    process exits (each timer fire is its own short-lived process, so the lock
    releases naturally when the cycle ends).

    - Scheduled runs (blocking=False): return False if another render holds the
      lock → caller skips this tick and retries on the next 2-min timer fire,
      so channels drain one-by-one (sequential queue).
    - Force/manual runs (blocking=True): wait for the turn.
    - No fcntl (e.g. Windows open-source single-shot): no-op → returns True.
    """
    global _render_lock_fh
    try:
        import fcntl
    except ImportError:
        return True
    # Open the lock file. If we cannot even OPEN it (e.g. a stale file owned by
    # another user / permission error), DO NOT block posting — proceed without
    # the lock. Only a genuine flock contention (below) should cause a skip.
    try:
        _render_lock_fh = open(_RENDER_LOCK_PATH, "a+")
        try:
            os.chmod(_RENDER_LOCK_PATH, 0o666)  # best-effort: any user can lock
        except OSError:
            pass
    except OSError as exc:  # noqa: BLE001
        log.warning("Render lock file unavailable (%s) — proceeding WITHOUT lock", exc)
        _render_lock_fh = None
        return True
    # Acquire. A failure here = another render genuinely holds it → skip.
    try:
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(_render_lock_fh, flags)
        return True
    except OSError:
        _render_lock_fh.close()
        _render_lock_fh = None
        return False


# ───────────────────── core ─────────────────────


async def run_once(force: bool = False) -> int:
    """Run one auto-post cycle.

    force=True bypasses BOTH the schedule (outside-hours pause) and the
    throttle check. Used by Telegram `/auto` command to trigger a post
    on-demand without disturbing the regular schedule afterwards.
    """
    enabled = (os.environ.get("AUTO_POST_ENABLED", "false") or "").strip().lower()
    if enabled not in ("true", "1", "yes", "on") and not force:
        log.info("AUTO_POST_ENABLED is false — skipping")
        return 0

    state = load_state()

    if force:
        log.info("FORCE mode — bypassing schedule + throttle")
    else:
        # Schedule-aware interval: AUTO_POST_SCHEDULE wins over AUTO_POST_INTERVAL.
        schedule = parse_schedule(os.environ.get("AUTO_POST_SCHEDULE", ""))
        if schedule:
            hour = current_local_hour()
            interval_min = schedule.get(hour, 0)
            if interval_min == 0:
                log.info("Hour %02d local — outside schedule, pause", hour)
                return 0
            interval_s = interval_min * 60
            log.info("Hour %02d local → interval %d min", hour, interval_min)
        else:
            interval_s = parse_interval(os.environ.get("AUTO_POST_INTERVAL", "30min"))

        last_ts = int(state.get("last_post_ts", 0))
        now_check = int(time.time())
        elapsed = now_check - last_ts
        if elapsed < interval_s:
            log.info(
                "Throttle: %ds left until next post (interval=%ds, last=%ds ago)",
                interval_s - elapsed, interval_s, elapsed,
            )
            return 0
    # Force path skips throttle check above
    now = int(time.time())

    # Global render lock — serialize renders across ALL channels (1 at a time)
    # to avoid VPS overload at shared schedule slots. Scheduled runs skip+retry
    # next tick if busy (sequential drain); force/manual runs wait their turn.
    if not _acquire_render_lock(blocking=force):
        log.info("Another channel is rendering — skip this tick, retry next fire")
        return 0

    # 1. Pick topic + layout
    niche = os.environ.get("NICHE", "language")
    target_lang = os.environ.get("DEFAULT_TARGET_LANG", "de")
    native_lang = os.environ.get("DEFAULT_NATIVE_LANG", "vi").lower()

    # Map target_lang → human-readable name in the channel's NATIVE language.
    # ktb-lingora: when native=en the names are English (German, Russian, ...)
    # so Gemini parses native_lang correctly per channel.
    LANG_NAME_BY_NATIVE = {
        "vi": {
            "de": "Đức", "ru": "Nga", "en": "Anh", "ko": "Hàn",
            "ja": "Nhật", "fr": "Pháp", "es": "Tây Ban Nha",
            "zh": "Trung", "th": "Thái",
        },
        "en": {
            "de": "German", "ru": "Russian", "ko": "Korean",
            "ja": "Japanese", "fr": "French", "es": "Spanish",
            "zh": "Mandarin Chinese", "th": "Thai", "vi": "Vietnamese",
            "it": "Italian", "pt": "Portuguese", "pl": "Polish",
            "tr": "Turkish",
        },
    }
    name_map = LANG_NAME_BY_NATIVE.get(native_lang, LANG_NAME_BY_NATIVE["vi"])
    target_lang_name = name_map.get(target_lang, target_lang.upper())

    # DEMO_MODE: allow CLI override of layout + request, skip topic_picker
    demo_layout = os.environ.get("_DEMO_LAYOUT")
    demo_request = os.environ.get("_DEMO_REQUEST")
    if demo_layout and demo_request:
        layout_type = demo_layout
        request_text = demo_request
        log.info("DEMO override: layout=%s request=%r", layout_type, request_text)
    else:
        request_text, layout_type = topic_picker.pick_next_request(
            state, niche=niche,
            target_lang=target_lang, target_lang_name=target_lang_name,
            native_lang=native_lang,
        )
        log.info("Picked layout=%s request=%r [native=%s, target=%s]",
                 layout_type, request_text, native_lang, target_lang)

    job_id = f"auto-{now}-{layout_type}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # 2. Gemini content
    if layout_type == "quiz":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_quiz, request_text,
        )
    elif layout_type == "quiz_reverse":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_quiz_reverse, request_text,
        )
    elif layout_type == "whats_this":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_whats_this, request_text,
        )
    elif layout_type == "whats_board":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_whats_board, request_text,
        )
    elif layout_type == "dialogue":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_dialogue, request_text,
        )
    elif layout_type == "fill_blank":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_fill_blank, request_text,
        )
    elif layout_type == "vocab_table":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_vocab_table, request_text,
        )
    elif layout_type == "compare":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_compare, request_text,
        )
    elif layout_type == "guess_word":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_guess_word, request_text,
        )
    elif layout_type == "vocab_card":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_vocab_card, request_text,
        )
    elif layout_type == "conjugation":
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate_conjugation, request_text,
        )
    else:
        intent, content = await asyncio.to_thread(
            generator.parse_and_generate, request_text, niche=niche,
        )
    log.info("Content generated (intent.topic=%r, count=%d, layout=%s)",
             intent.topic, intent.count, layout_type)

    # Production auto channels can opt out of caption hashtags via
    # CAPTION_HASHTAGS=false in their .env. Default ON (Shortcraft uses a
    # different path; open-source keeps hashtags).
    if not _env_bool("CAPTION_HASHTAGS", default=True):
        content.caption = _strip_hashtags(content.caption)

    # 3. Voice picking
    native_voice = await tts.pick_voice_async(intent.native_lang, "female")
    if niche == "language":
        try:
            voice = await tts.pick_voice_async(intent.target_lang, intent.voice_gender)
        except ValueError as exc:
            log.error("Voice picking failed: %s", exc)
            return 1
    else:
        voice = native_voice

    # 4. Audio segments
    audio_dir = job_dir / "tts"
    audio_dir.mkdir(parents=True, exist_ok=True)

    base_rate = _norm_rate(os.environ.get("VOICE_RATE", "-10%"))
    desc_env_rate_raw = os.environ.get("DESC_VOICE_RATE", "")
    desc_env_rate = _norm_rate(desc_env_rate_raw) if desc_env_rate_raw else ""
    if layout_type in ("quiz", "quiz_reverse") and os.environ.get("QUIZ_VOICE_RATE"):
        base_rate = _norm_rate(os.environ["QUIZ_VOICE_RATE"])

    # Unified rate for ALL VN narration across ALL layouts (feedback v6).
    # Vietnamese viewers understand instantly — no need to read slowly.
    # Applied to: intro_vi, outro_vi, per-phrase VN translations, VN option labels.
    # Target-language voice stays slower (base_rate / target_rate) for learning clarity.
    # Native voice rate — controls speed of the intro/outro/per-card-translation
    # voice (i.e. the channel's NATIVE language voice — vi for Tiếng Nhật 5 Phút,
    # en for Russian Path). Defaults to +30% because native speakers parse their
    # own language quickly so we can speed it up without hurting comprehension;
    # only the TARGET-language voice stays at the slower base_rate for learning.
    #
    # Env vars (in priority order):
    #   NATIVE_VOICE_RATE  — preferred name (lingora)
    #   VN_VOICE_RATE      — legacy alias from production telegram-video-bot,
    #                        when every channel was vi-native. Still honoured
    #                        so production-derived .env files don't break.
    native_voice_rate = _norm_rate(
        os.environ.get("NATIVE_VOICE_RATE")
        or os.environ.get("VN_VOICE_RATE")
        or "+30%",
    )
    # Old code referenced `vn_voice_rate` and `intro_outro_rate` — keep aliases
    # so the rest of this function reads identically to the production source.
    vn_voice_rate = native_voice_rate
    intro_outro_rate = native_voice_rate  # backward-compat alias

    if layout_type == "quiz":
        # Native voices (intro/outro/letter labels) all +25%. Target option voice = base_rate.
        # Letter A/B/C/D spoken differently per native language so the narrator
        # doesn't break character — VN says "A Bê Xê Đê" while EN says "A B C D".
        LETTER_BY_NATIVE = {
            "vi": {"A": "A", "B": "Bê", "C": "Xê", "D": "Đê"},
            "en": {"A": "A", "B": "B", "C": "C", "D": "D"},
            "ko": {"A": "에이", "B": "비", "C": "씨", "D": "디"},
            "ja": {"A": "エー", "B": "ビー", "C": "シー", "D": "ディー"},
            "es": {"A": "A", "B": "Be", "C": "Ce", "D": "De"},
        }
        letter_map = LETTER_BY_NATIVE.get(intent.native_lang, LETTER_BY_NATIVE["vi"])
        segments = [tts.AudioSegment("intro_vi", content.intro_native, native_voice, rate=vn_voice_rate)]
        for opt in content.options:
            letter = letter_map.get(opt.label, opt.label)
            segments.append(tts.AudioSegment(f"opt_{opt.label}_label", letter, native_voice, rate=vn_voice_rate))
            segments.append(tts.AudioSegment(f"opt_{opt.label}", opt.text, voice, rate=base_rate))
        segments.append(tts.AudioSegment("outro_vi", content.outro_native, native_voice, rate=vn_voice_rate))
    elif layout_type == "whats_this":
        # Visual vocab v3+: NO intro voice. Start with item 1 immediately for retention.
        # 21 segments: 10 (q_i target voice + r_i target voice) + 1 outro_vi.
        target_rate = _norm_rate(os.environ.get("QUIZ_REVERSE_TARGET_RATE", "-15%"))
        segments = []
        for i, item in enumerate(content.items, start=1):
            segments.append(tts.AudioSegment(f"q_{i}", item.voice_question, voice, rate=target_rate))
            segments.append(tts.AudioSegment(f"r_{i}", item.voice_reveal, voice, rate=target_rate))
        segments.append(tts.AudioSegment("outro_vi", content.outro_native, native_voice, rate=vn_voice_rate))
    elif layout_type == "whats_board":
        # 9-grid cheat sheet. No intro voice. Target voice reads each word twice (built into voice_repeat).
        # 10 segments: v_1..v_9 (target voice) + outro_vi.
        target_rate = _norm_rate(os.environ.get("QUIZ_REVERSE_TARGET_RATE", "-15%"))
        segments = []
        for i, item in enumerate(content.items, start=1):
            segments.append(tts.AudioSegment(f"v_{i}", item.voice_repeat, voice, rate=target_rate))
        segments.append(tts.AudioSegment("outro_vi", content.outro_native, native_voice, rate=vn_voice_rate))
    elif layout_type == "fill_blank":
        # Video v3 (2026-06-29): intro_vi + 3 target-voice options + outro_vi.
        # Intro/outro narration text is lang-keyed in composer.fill_blank_voice_texts
        # so the same script is used by lingora bot AND Shortcraft webapp.
        intro_text, outro_text = composer.fill_blank_voice_texts(intent.native_lang)
        target_rate = _norm_rate(os.environ.get("QUIZ_REVERSE_TARGET_RATE", "-10%"))
        segments = [tts.AudioSegment("intro_vi", intro_text, native_voice, rate=vn_voice_rate)]
        for label, opt_text in zip(["A", "B", "C"], content.options[:3]):
            segments.append(tts.AudioSegment(f"opt_{label}", opt_text, voice, rate=target_rate))
        segments.append(tts.AudioSegment("outro_vi", outro_text, native_voice, rate=vn_voice_rate))
    elif layout_type == "vocab_table":
        # Static PNG poster — NO audio.
        segments = []
    elif layout_type == "compare":
        # Static PNG poster — 2-column comparison, NO audio.
        segments = []
    elif layout_type == "dialogue":
        # 2-character dialogue. Use the per-character voice_gender from Gemini,
        # but override via VN-prefix heuristic so name + voice are always
        # consistent (the old code force-flipped char_b which created name/voice
        # mismatches like "Bác Hans" voiced by a female TTS).
        target_rate = _norm_rate(os.environ.get("DIALOGUE_TARGET_RATE", "-10%"))

        def _gender_from_name(name: str, fallback: str) -> str:
            """VN-prefix heuristic. Falls back to Gemini's value for foreign-only names."""
            s = (name or "").strip()
            low = s.lower()
            male_prefix = ("anh ", "bác ", "ông ", "chú ", "cậu ")
            # "em " bỏ ra: mơ hồ giới tính (em trai/em gái) — để rơi về fallback Gemini.
            female_prefix = ("chị ", "cô ", "bà ", "dì ")
            if any(low.startswith(p) for p in male_prefix):
                return "male"
            if any(low.startswith(p) for p in female_prefix):
                return "female"
            return fallback if fallback in ("male", "female") else "female"

        raw_a = (content.char_a.voice_gender or "female").lower()
        raw_b = (content.char_b.voice_gender or "male").lower()
        gender_a = _gender_from_name(content.char_a.name, raw_a)
        gender_b = _gender_from_name(content.char_b.name, raw_b)
        content.char_a.voice_gender = gender_a
        content.char_b.voice_gender = gender_b
        # Keep the rendered avatar's gender in lockstep with the chosen voice.
        content.char_a.image_prompt = _sync_image_prompt_gender(content.char_a.image_prompt, gender_a)
        content.char_b.image_prompt = _sync_image_prompt_gender(content.char_b.image_prompt, gender_b)

        try:
            voice_a = await tts.pick_voice_async(intent.target_lang, gender_a)
            voice_b = await tts.pick_voice_async(intent.target_lang, gender_b)
        except ValueError as exc:
            log.error("Dialogue voice picking failed: %s", exc)
            return 1

        # If voice picker returned the SAME voice (rare but happens when locale
        # has only 1 voice per gender), or both genders are equal, vary char_b's
        # rate slightly so the two speakers don't blur together.
        rate_a = target_rate
        rate_b = target_rate
        same_voice = (voice_a == voice_b) or (gender_a == gender_b)
        if same_voice:
            try:
                pct = int(target_rate.rstrip("%"))
            except ValueError:
                pct = 0
            rate_b = f"{pct + 8:+d}%"  # bump 8% faster to differentiate
            log.info("Dialogue same-gender voices — bumping char_b rate: %s → %s",
                     target_rate, rate_b)

        log.info("Dialogue voices: A=%s (%s, %s) | B=%s (%s, %s)",
                 voice_a, gender_a, rate_a, voice_b, gender_b, rate_b)
        segments = []
        for i, turn in enumerate(content.turns, start=1):
            is_a = turn.speaker.upper() == "A"
            v = voice_a if is_a else voice_b
            r = rate_a if is_a else rate_b
            segments.append(tts.AudioSegment(f"t_{i}", turn.target, v, rate=r))
        segments.append(tts.AudioSegment("outro_vi", content.outro_native, native_voice, rate=vn_voice_rate))
    elif layout_type == "quiz_reverse":
        # Reverse: target voice reads question_target. All 4 option texts in NATIVE
        # language → native voice at +25%. Letter A/B/C/D spoken per native_lang.
        target_rate = _norm_rate(os.environ.get("QUIZ_REVERSE_TARGET_RATE", "-15%"))
        LETTER_BY_NATIVE = {
            "vi": {"A": "A", "B": "Bê", "C": "Xê", "D": "Đê"},
            "en": {"A": "A", "B": "B", "C": "C", "D": "D"},
            "ko": {"A": "에이", "B": "비", "C": "씨", "D": "디"},
            "ja": {"A": "エー", "B": "ビー", "C": "シー", "D": "ディー"},
            "es": {"A": "A", "B": "Be", "C": "Ce", "D": "De"},
        }
        letter_map = LETTER_BY_NATIVE.get(intent.native_lang, LETTER_BY_NATIVE["vi"])
        segments = [
            tts.AudioSegment("intro_vi", content.intro_native, native_voice, rate=vn_voice_rate),
            tts.AudioSegment("intro_target", content.question_target, voice, rate=target_rate),
        ]
        for opt in content.options:
            letter = letter_map.get(opt.label, opt.label)
            segments.append(tts.AudioSegment(f"opt_{opt.label}_label", letter, native_voice, rate=vn_voice_rate))
            # Option text is in NATIVE language → native voice at +25%
            segments.append(tts.AudioSegment(f"opt_{opt.label}", opt.text, native_voice, rate=vn_voice_rate))
        segments.append(tts.AudioSegment("outro_vi", content.outro_native, native_voice, rate=vn_voice_rate))
    elif layout_type == "guess_word":
        # 10× target-word reveal (target voice, slow for clarity) + outro_native.
        # composer.build_guess_word_project expects EXACTLY 11 clips named
        # reveal_1..reveal_10 + outro_native (NOT outro_vi).
        target_rate = _norm_rate(os.environ.get("QUIZ_REVERSE_TARGET_RATE", "-15%"))
        segments = []
        for i, w in enumerate(content.words, start=1):
            segments.append(tts.AudioSegment(f"reveal_{i}", w.target_word, voice, rate=target_rate))
        segments.append(tts.AudioSegment("outro_native", content.outro_native, native_voice, rate=vn_voice_rate))
    elif layout_type == "vocab_card":
        # CEO 2026-06-30: target-lang ONLY. 2 clips — `word` reads the focal
        # vocab, `example` reads the example sentence. Slow (-15%) so a
        # learner can hear each syllable.
        target_rate = _norm_rate(os.environ.get("QUIZ_REVERSE_TARGET_RATE", "-15%"))
        segments = [
            tts.AudioSegment("word", content.target_word, voice, rate=target_rate),
            tts.AudioSegment("example", content.example_sentence, voice, rate=target_rate),
        ]
    elif layout_type == "conjugation":
        # CEO 2026-07-22: read pronoun + conjugated form (e.g. "Я читаю"),
        # not the form alone. For "он/она" row, voice says the primary
        # pronoun only (first token before the slash) so TTS doesn't stumble.
        # 7 clips total: verb infinitive + 6 pronoun+form clips.
        target_rate = _norm_rate(os.environ.get("QUIZ_REVERSE_TARGET_RATE", "-15%"))
        segments = [
            tts.AudioSegment("verb", content.verb_target, voice, rate=target_rate),
        ]
        for i, f in enumerate(content.forms, start=1):
            primary_pn = (f.pronoun or "").split("/")[0].strip()
            speech = (primary_pn + " " + f.conjugated).strip() if primary_pn else f.conjugated
            segments.append(tts.AudioSegment(f"form_{i}", speech, voice, rate=target_rate))
    else:
        # phrases layout — VN voice (intro/outro/per-phrase translation) all +30%.
        # Target language phrase stays at base_rate for learning clarity.
        segments = [tts.AudioSegment("intro_vi", content.intro_native, native_voice, rate=vn_voice_rate)]
        for i, p in enumerate(content.phrases, start=1):
            segments.append(tts.AudioSegment(f"p{i}_target", p.target, voice, rate=base_rate))
            # Per-phrase VN translation = native voice at +30% (used to be desc_rate which was slow)
            segments.append(tts.AudioSegment(f"p{i}_vi", p.native, native_voice, rate=vn_voice_rate))
        segments.append(tts.AudioSegment("outro_vi", content.outro_native, native_voice, rate=vn_voice_rate))

    # Skip TTS for audio-less layouts (vocab_table is a static PNG poster)
    clips = await tts.synth_segments_async(segments, audio_dir) if segments else []

    # 5. Compose + render
    hf_ver = os.environ.get("HYPERFRAMES_VERSION", "0.6.52")
    if layout_type in ("quiz", "quiz_reverse"):
        # Quiz layouts: fetch Pexels bg (composited post-HF); scene image kept
        # as fallback for the pre-refresh path if bg fetch fails.
        scene_dir = job_dir / "ai_images"
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_prompt = getattr(content, "scene_image_prompt", "") or ""
        bg_path = await _maybe_fetch_bg_video(scene_prompt, job_dir, layout_type)
        if bg_path is None and scene_prompt.strip():
            try:
                await image_gen.gen_image(scene_prompt, scene_dir / "scene.png")
                log.info("Quiz scene image gen OK (fallback path)")
            except Exception as exc:  # noqa: BLE001
                log.warning("Quiz scene image gen failed: %s", exc)
        composer.build_quiz_project(
            content=content, audio_clips=clips, audio_src_dir=audio_dir,
            out_dir=job_dir, target_lang_name=intent.target_lang_name,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
            direction="reverse" if layout_type == "quiz_reverse" else "forward",
            image_src_dir=scene_dir,
            native_lang=intent.native_lang,
            bg_video_path=bg_path,
        )
    elif layout_type == "whats_this":
        # 5a. Generate 10 AI images via Cloudflare Workers AI (FLUX)
        image_dir = job_dir / "ai_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_items = [
            (item.image_prompt, image_dir / f"whats_{i}.png")
            for i, item in enumerate(content.items, start=1)
        ]
        log.info("Gen %d AI images via Cloudflare FLUX...", len(image_items))
        try:
            await image_gen.gen_images_batch(image_items, max_concurrent=4)
            log.info("All %d images generated", len(image_items))
        except image_gen.CFQuotaExhaustedError:
            log.warning("⏭ CF quota cạn — bỏ lượt đăng [%s], thử lại lượt sau", CHANNEL_DIR.name)
            await _maybe_notify_quota_skip(intent.topic)
            return 0
        except Exception as exc:  # noqa: BLE001
            log.exception("Image gen failed: %s", exc)
            await notify(
                f"❌ <b>whats_this image gen lỗi</b> [{CHANNEL_DIR.name}]\n"
                f"📝 <i>{intent.topic}</i>\n"
                f"💥 <code>{str(exc)[:300]}</code>"
            )
            return 1

        # 5b. Compose project with images + audio
        composer.build_whats_this_project(
            content=content, audio_clips=clips, audio_src_dir=audio_dir,
            image_src_dir=image_dir,
            out_dir=job_dir, target_lang_name=intent.target_lang_name,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
            native_lang=intent.native_lang,
        )
    elif layout_type == "whats_board":
        # 5a. Generate 9 AI images for the grid via CF FLUX
        image_dir = job_dir / "ai_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_items = [
            (item.image_prompt, image_dir / f"board_{i}.png")
            for i, item in enumerate(content.items, start=1)
        ]
        log.info("Gen %d AI images via Cloudflare FLUX (board)...", len(image_items))
        try:
            await image_gen.gen_images_batch(image_items, max_concurrent=3)
            log.info("All %d images generated", len(image_items))
        except image_gen.CFQuotaExhaustedError:
            log.warning("⏭ CF quota cạn — bỏ lượt đăng [%s], thử lại lượt sau", CHANNEL_DIR.name)
            await _maybe_notify_quota_skip(intent.topic)
            return 0
        except Exception as exc:  # noqa: BLE001
            log.exception("Image gen failed: %s", exc)
            await notify(
                f"❌ <b>whats_board image gen lỗi</b> [{CHANNEL_DIR.name}]\n"
                f"📝 <i>{intent.topic}</i>\n"
                f"💥 <code>{str(exc)[:300]}</code>"
            )
            return 1

        # 5b. Compose board project
        composer.build_whats_board_project(
            content=content, audio_clips=clips, audio_src_dir=audio_dir,
            image_src_dir=image_dir,
            out_dir=job_dir, target_lang_name=intent.target_lang_name,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
        )
    elif layout_type == "dialogue":
        # 5a. Generate 3 AI images: char_a, char_b, scene
        image_dir = job_dir / "ai_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_items = [
            (content.char_a.image_prompt, image_dir / "char_a.png"),
            (content.char_b.image_prompt, image_dir / "char_b.png"),
            (content.scene_image_prompt,  image_dir / "scene.png"),
        ]
        log.info("Gen 3 AI images (dialogue: char_a + char_b + scene)...")
        try:
            await image_gen.gen_images_batch(image_items, max_concurrent=3)
            log.info("All 3 dialogue images generated")
        except image_gen.CFQuotaExhaustedError:
            log.warning("⏭ CF quota cạn — bỏ lượt đăng [%s], thử lại lượt sau", CHANNEL_DIR.name)
            await _maybe_notify_quota_skip(intent.topic)
            return 0
        except Exception as exc:  # noqa: BLE001
            log.exception("Image gen failed: %s", exc)
            await notify(
                f"❌ <b>dialogue image gen lỗi</b> [{CHANNEL_DIR.name}]\n"
                f"📝 <i>{intent.topic}</i>\n"
                f"💥 <code>{str(exc)[:300]}</code>"
            )
            return 1

        # 5b. Compose dialogue project
        composer.build_dialogue_project(
            content=content, audio_clips=clips, audio_src_dir=audio_dir,
            image_src_dir=image_dir,
            out_dir=job_dir, target_lang_name=intent.target_lang_name,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
        )
    elif layout_type == "fill_blank":
        # 5a. Gen 1 photorealistic scene image
        image_dir = job_dir / "ai_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        log.info("Gen photorealistic scene for fill_blank...")
        try:
            await image_gen.gen_image(content.scene_image_prompt, image_dir / "scene.png")
        except image_gen.CFQuotaExhaustedError:
            log.warning("⏭ CF quota cạn — bỏ lượt đăng [%s], thử lại lượt sau", CHANNEL_DIR.name)
            await _maybe_notify_quota_skip(intent.topic)
            return 0
        except Exception as exc:  # noqa: BLE001
            log.exception("fill_blank scene image gen failed: %s", exc)
            await notify(
                f"❌ <b>fill_blank image gen lỗi</b> [{CHANNEL_DIR.name}]\n"
                f"📝 <i>{intent.topic}</i>\n"
                f"💥 <code>{str(exc)[:300]}</code>"
            )
            return 1
        bg_path = await _maybe_fetch_bg_video(content.scene_image_prompt, job_dir, "fill_blank")
        composer.build_fill_blank_project(
            content=content,
            image_src_dir=image_dir,
            out_dir=job_dir, target_lang_name=intent.target_lang_name,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
            native_lang=intent.native_lang,
            audio_clips=clips,
            audio_src_dir=audio_dir,
            bg_video_path=bg_path,
        )
    elif layout_type == "vocab_table":
        # v4: Gen full photorealistic scene (like fill_blank) instead of small icon.
        image_dir = job_dir / "ai_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        log.info("Gen photorealistic scene for vocab_table...")
        try:
            await image_gen.gen_image(content.scene_image_prompt, image_dir / "scene.png")
        except image_gen.CFQuotaExhaustedError:
            log.warning("⏭ CF quota cạn — bỏ lượt đăng [%s], thử lại lượt sau", CHANNEL_DIR.name)
            await _maybe_notify_quota_skip(intent.topic)
            return 0
        except Exception as exc:  # noqa: BLE001
            log.exception("vocab_table scene image gen failed: %s", exc)
            await notify(
                f"❌ <b>vocab_table image gen lỗi</b> [{CHANNEL_DIR.name}]\n"
                f"📝 <i>{intent.topic}</i>\n"
                f"💥 <code>{str(exc)[:300]}</code>"
            )
            return 1
        composer.build_vocab_table_project(
            content=content,
            image_src_dir=image_dir,
            out_dir=job_dir, target_lang_name=intent.target_lang_name,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
            native_lang=intent.native_lang,
        )
    elif layout_type == "compare":
        # 2-column compare poster — NO image gen, NO TTS, pure HTML render.
        composer.build_compare_project(
            content=content,
            out_dir=job_dir, target_lang_name=intent.target_lang_name,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
            native_lang=intent.native_lang,
        )
    elif layout_type == "guess_word":
        # CEO 2026-07-21: Pexels stock bg video + glass card. Fetch first,
        # composer copies to static/bg.mp4; template plays it muted/looped.
        # If Pexels fails (no key, 4xx, no candidate), composer falls back
        # to the built-in gradient background so the video still renders.
        import stock_video  # type: ignore  # noqa: E402
        bg_dir = job_dir / "stock_bg"
        bg_dir.mkdir(parents=True, exist_ok=True)
        bg_path: Path | None = None
        query = (getattr(content, "stock_video_query", "") or "").strip()
        if query:
            log.info("Gen guess_word bg from Pexels: %r", query)
            try:
                bg_path = await stock_video.fetch_bg_video(query, bg_dir / "bg.mp4")
            except Exception as exc:  # noqa: BLE001
                log.warning("guess_word bg fetch failed: %s", exc)
                bg_path = None
        if bg_path is None:
            log.info("guess_word: falling back to gradient bg (no Pexels clip)")
        composer.build_guess_word_project(
            content=content, audio_clips=clips, audio_src_dir=audio_dir,
            out_dir=job_dir, target_lang_name=intent.target_lang_name,
            target_lang=intent.target_lang,
            native_lang=intent.native_lang,
            lesson_number=int(state.get("post_count", 0)) + 1,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
            bg_video_path=bg_path,
        )
    elif layout_type == "vocab_card":
        # 1 photorealistic illustration + 2 target-voice clips already synth'd
        # (`word`, `example`). composer wires the master.wav premix.
        image_dir = job_dir / "ai_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        log.info("Gen photorealistic scene for vocab_card...")
        try:
            await image_gen.gen_image(content.image_prompt, image_dir / "scene.png", style="cinematic")
        except image_gen.CFQuotaExhaustedError:
            log.warning("CF quota cạn — bỏ lượt vocab_card [%s]", CHANNEL_DIR.name)
            await _maybe_notify_quota_skip(intent.topic)
            return 0
        except Exception as exc:  # noqa: BLE001
            log.exception("vocab_card image gen failed: %s", exc)
            await notify(
                f"❌ <b>vocab_card image gen lỗi</b> [{CHANNEL_DIR.name}]\n"
                f"📝 <i>{intent.topic}</i>\n"
                f"💥 <code>{str(exc)[:300]}</code>"
            )
            return 1
        composer.build_vocab_card_project(
            content=content, audio_clips=clips, audio_src_dir=audio_dir,
            image_src_dir=image_dir,
            out_dir=job_dir, target_lang_name=intent.target_lang_name,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
            native_lang=intent.native_lang,
        )
    elif layout_type == "conjugation":
        # CEO 2026-07-22: 1 verb + 6 personal-pronoun forms. Bg = Pexels stock
        # video (composited post-HF via chromakey). No AI scene image needed.
        bg_path = await _maybe_fetch_bg_video(content.scene_image_prompt, job_dir, "conjugation")
        composer.build_conjugation_project(
            content=content, audio_clips=clips, audio_src_dir=audio_dir,
            out_dir=job_dir, target_lang_name=intent.target_lang_name,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
            native_lang=intent.native_lang,
            bg_video_path=bg_path,
        )
    else:
        # phrases layout: fetch Pexels bg (composited post-HF); scene image kept
        # as fallback path if Pexels fails.
        scene_dir = job_dir / "ai_images"
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_prompt = getattr(content, "scene_image_prompt", "") or ""
        bg_path = await _maybe_fetch_bg_video(scene_prompt, job_dir, "phrases")
        if bg_path is None and scene_prompt.strip():
            try:
                await image_gen.gen_image(scene_prompt, scene_dir / "scene.png")
                log.info("Phrases scene image gen OK (fallback path)")
            except Exception as exc:  # noqa: BLE001
                log.warning("Phrases scene image gen failed: %s", exc)
        composer.build_project(
            content=content, audio_clips=clips, audio_src_dir=audio_dir,
            out_dir=job_dir,
            native_lang=intent.native_lang,
            target_lang_name=intent.target_lang_name,
            hyperframes_version=hf_ver, channel_dir=CHANNEL_DIR,
            image_src_dir=scene_dir,
            post_number=int(state.get("post_count", 0)) + 1,
            bg_video_path=bg_path,
        )

    output_mp4 = job_dir / "output.mp4"
    await asyncio.to_thread(
        renderer.render, job_dir, output_mp4, hyperframes_version=hf_ver,
    )
    log.info("Rendered MP4: %s (%d bytes)", output_mp4, output_mp4.stat().st_size)

    # CEO 2026-07-21: bg-video composite for layouts in BG_VIDEO_LAYOUTS. HF's
    # stepped virtual clock freezes <video> at frame 1, so the stock clip is
    # composited AFTER HF via ffmpeg chromakey + dim + overlay. Templates
    # rendered content on solid green #00ff00; we chromakey it out, overlay
    # onto the looping bg + dim layer.
    if layout_type in BG_VIDEO_LAYOUTS:
        bg_static = job_dir / "static" / "bg.mp4"
        if bg_static.exists():
            import stock_video  # type: ignore  # noqa: E402
            composite_out = job_dir / "output_composited.mp4"
            try:
                await stock_video.composite_bg(output_mp4, bg_static, composite_out)
                output_mp4.unlink(missing_ok=True)
                composite_out.rename(output_mp4)
                log.info("%s bg composite ok: %d bytes", layout_type, output_mp4.stat().st_size)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s bg composite failed, keeping HF-only output: %s", layout_type, exc)

    # ───── Defensive: validate audio stream presence ─────
    # HyperFrames sometimes returns exit 0 but renders video-only MP4 (Chromium
    # audio context bug under high concurrency). If we expected audio but the
    # MP4 is silent → re-render once. Skip the check for photo-post layouts
    # (vocab_table, compare) which legitimately have no audio. fill_blank
    # moved to real-video mode 2026-06-29 — it's NO LONGER a photo layout.
    is_photo_layout = layout_type in ("vocab_table", "compare")
    expected_audio = bool(segments) and not is_photo_layout
    if expected_audio:
        has_audio = await asyncio.to_thread(mp4_has_audio_stream, output_mp4)
        if not has_audio:
            log.error(
                "🔴 SILENT RENDER detected: expected audio but MP4 has NO audio stream. "
                "Re-rendering once..."
            )
            try:
                await notify(
                    f"⚠️ <b>Silent render detected</b> [{CHANNEL_DIR.name}]\n"
                    f"🎬 Layout: {layout_type}\n"
                    f"📝 Topic: <i>{intent.topic}</i>\n"
                    f"🔁 Re-rendering once before upload..."
                )
            except Exception:  # noqa: BLE001
                pass
            # Retry render once (same composition, same TTS, just re-invoke HF)
            await asyncio.to_thread(
                renderer.render, job_dir, output_mp4, hyperframes_version=hf_ver,
            )
            has_audio = await asyncio.to_thread(mp4_has_audio_stream, output_mp4)
            if not has_audio:
                log.error("🔴 Re-render STILL silent — aborting FB upload")
                err_msg = "Silent MP4 after 2 render attempts (HF audio embed bug)"
                state["last_error"] = err_msg
                if not force:
                    state["last_post_ts"] = now
                save_state(state)
                prefix_err = "⚡ Force-post lỗi" if force else "❌ Auto-post lỗi"
                await notify(
                    f"<b>{prefix_err}</b> [{CHANNEL_DIR.name}]\n"
                    f"📝 <i>{intent.topic}</i>\n"
                    f"🎬 Layout: {layout_type}\n"
                    f"💥 <code>{err_msg}</code>\n"
                    f"📁 Job: <code>{job_dir.name}</code> (giữ để debug)"
                )
                return 1
            log.info("✅ Re-render OK: audio stream now present")

    # ───── Static layout → Ken-Burns Reel (lingora 2026-06-12) ───────────
    # vocab_table / fill_blank / compare render as a static frame (no
    # animation, no audio). When STATIC_LAYOUTS_AS_VIDEO=true (default for
    # new lingora pages) we re-encode the frame as a 5s Ken-Burns zoom-in
    # video. The result is eligible for Reels distribution — far wider reach
    # than the legacy /photos path on the 2026 FB algorithm. Flip the env to
    # false once the page has audience to switch back to high-CTR statics.
    static_as_video_enabled = (
        os.environ.get("STATIC_LAYOUTS_AS_VIDEO", "true").strip().lower()
        in ("true", "1", "yes", "on")
    )
    is_static_kb_video = (
        is_photo_layout and static_as_video_enabled
    )
    if is_static_kb_video:
        log.info(
            "Static layout %s → Ken-Burns Reel with native voice narration",
            layout_type,
        )
        # Use the RAW 9:16 frame extractor — NOT the legacy 4:5 crop. The
        # underlying HyperFrames render is already 1080×1920, perfect for
        # Reels. Cropping to 4:5 first and then padding back to 9:16 would
        # distort the thumbnail; skipping the crop preserves the original.
        png_path = await asyncio.to_thread(
            extract_full_frame_png_raw, output_mp4, at_seconds=0.8,
        )
        if png_path is None:
            log.warning("PNG extract failed — falling back to /photos")
            is_static_kb_video = False

        # 1. Generate narration via Edge TTS. For fill_blank we read each
        #    option separately so a yellow highlight ring can follow the
        #    voice; vocab_table / compare just get intro + outro.
        narration_segments: list[Path] = []
        narration_total_s: float = 0.0
        per_segment_durations: list[float] = []
        if is_static_kb_video:
            try:
                native_rate = _norm_rate(
                    os.environ.get("NATIVE_VOICE_RATE")
                    or os.environ.get("VN_VOICE_RATE")
                    or "+25%",
                )
                target_rate = _norm_rate(
                    os.environ.get("VOICE_RATE", "-10%"),
                )
                narration_dir = job_dir / "tts_static"
                narration_dir.mkdir(parents=True, exist_ok=True)

                if layout_type == "fill_blank":
                    lines = _fill_blank_narration_lines(
                        content, intent.native_lang, intent.target_lang_name,
                    )
                    segments_spec = []
                    for name, text in lines:
                        # Use target voice for the option lines so the
                        # foreign word is pronounced correctly; native voice
                        # for intro/outro.
                        is_opt = name.startswith("static_opt_")
                        segments_spec.append(tts.AudioSegment(
                            name, text,
                            voice if is_opt else native_voice,
                            rate=target_rate if is_opt else native_rate,
                        ))
                else:
                    intro_text, outro_text = _static_narration_script(
                        layout_type, content,
                        intent.native_lang, intent.target_lang_name,
                    )
                    segments_spec = [
                        tts.AudioSegment(
                            "static_intro", intro_text, native_voice,
                            rate=native_rate,
                        ),
                        tts.AudioSegment(
                            "static_outro", outro_text, native_voice,
                            rate=native_rate,
                        ),
                    ]
                static_clips = await tts.synth_segments_async(
                    segments_spec, narration_dir,
                )
                narration_segments = [
                    narration_dir / c.file for c in static_clips
                ]
                per_segment_durations = [c.duration for c in static_clips]
                narration_total_s = sum(per_segment_durations)
                log.info(
                    "Static narration: %d segments, total %.1fs",
                    len(static_clips), narration_total_s,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Static narration TTS failed (%s) — proceeding silent",
                    exc,
                )
                narration_segments = []
                narration_total_s = 0.0
                per_segment_durations = []

        # 2. Compute Ken-Burns duration to COVER the FULL narration so the
        #    voice is never cut. Combined audio = lead + Σsegments + gap·(N−1)
        #    + tail (N gaps between N segments = N−1). The mux uses -shortest,
        #    so the video MUST be ≥ audio or the trailing voice gets clipped —
        #    add a 0.5s safety margin. (Old formula assumed only 1 gap → for
        #    fill_blank's 5 segments it undershot by ~1.8s → cut voice.)
        lead_s = 0.3
        gap_s = 0.6
        tail_s = 0.4
        if narration_total_s > 0:
            n_seg = len(per_segment_durations)
            audio_total_s = (
                lead_s + narration_total_s + gap_s * max(0, n_seg - 1) + tail_s
            )
            kb_duration = max(5.0, audio_total_s + 1.0)
        else:
            kb_duration = 5.0  # fallback silent Ken-Burns

        # 3. Render Ken-Burns motion video.
        if is_static_kb_video:
            motion_path = job_dir / "output_motion.mp4"
            ok = await asyncio.to_thread(
                static_to_motion_video, png_path, motion_path,
                duration=kb_duration, fps=30, zoom_amount=0.10,
            )
            if ok is None:
                log.warning(
                    "Ken-Burns conversion failed — falling back to /photos",
                )
                is_static_kb_video = False
            else:
                try:
                    output_mp4.unlink()
                except OSError:
                    pass
                shutil.move(str(motion_path), str(output_mp4))
                log.info(
                    "Ken-Burns Reel ready: %s (%.1fs)",
                    output_mp4, kb_duration,
                )

        # 4. Mux narration audio onto the motion video. Uses the SAME
        #    lead_s/gap_s/tail_s computed above so video length ≥ audio length.
        if is_static_kb_video and narration_segments:
            mux_ok = await asyncio.to_thread(
                mux_audio_into_video, output_mp4, narration_segments,
                silence_between_s=gap_s, silence_lead_s=lead_s,
                silence_tail_s=tail_s,
            )
            if not mux_ok:
                log.warning(
                    "Audio mux failed — posting silent Ken-Burns Reel",
                )

        # 5. fill_blank only: overlay a yellow highlight ring around the
        #    option being read. Off by default for now — the empirical x/y
        #    coordinates of the chips vary per scene image and audio-codec
        #    rounding makes the timing windows misalign by ~100ms. Flip the
        #    env to true once we add a proper timed-clip path to the
        #    composer + template (planned v2).
        overlay_enabled = (
            os.environ.get("FILL_BLANK_RING_OVERLAY", "false").strip().lower()
            in ("true", "1", "yes", "on")
        )
        if (
            overlay_enabled
            and is_static_kb_video
            and layout_type == "fill_blank"
            and len(per_segment_durations) >= 4
        ):
            cursor = lead_s + per_segment_durations[0] + gap_s
            fill_blank_option_windows: list[tuple[float, float]] = []
            for i in range(1, min(len(per_segment_durations) - 1, 4)):
                start = cursor
                end = cursor + per_segment_durations[i]
                fill_blank_option_windows.append((start, end))
                cursor = end + gap_s
            ok_overlay = await asyncio.to_thread(
                overlay_option_highlights, output_mp4,
                option_windows=fill_blank_option_windows,
                time_offset_s=0.0,
            )
            if not ok_overlay:
                log.warning(
                    "Option-highlight overlay failed — posting without rings",
                )

    # ───── Prepend static thumbnail freeze frame (lingora 2026-06-12) ─────
    # FB Reels / Stories / TikTok auto-pick a thumbnail from the first ~1s of
    # the video — too early for our title to fully animate in. Prepend a 1s
    # static frame from t=2s so the chosen thumb is guaranteed to show the
    # native title. Skipped for photo-only path (when STATIC_LAYOUTS_AS_VIDEO
    # is off and the video will be discarded for /photos upload).
    thumb_freeze_enabled = (
        os.environ.get("PREPEND_THUMB_FREEZE", "true").strip().lower()
        in ("true", "1", "yes", "on")
    )
    should_freeze = thumb_freeze_enabled and (
        not is_photo_layout or is_static_kb_video
    )
    if should_freeze:
        capture_ms = int(os.environ.get("THUMB_CAPTURE_MS", "2000"))
        hold_s = float(os.environ.get("THUMB_HOLD_S", "1.0"))
        ok = await asyncio.to_thread(
            prepend_thumb_freeze, output_mp4,
            capture_at_ms=capture_ms, hold_duration_s=hold_s,
        )
        if not ok:
            log.warning(
                "Thumbnail freeze failed — posting original video as-is",
            )

    # DEMO_MODE: stop after render, no FB post, keep job_dir for export
    if os.environ.get("DEMO_MODE", "0").strip() in ("1", "true", "yes", "on"):
        log.info("DEMO_MODE — skipping FB upload, output kept at %s", output_mp4)
        # Save MP4 path for the CLI wrapper to find
        marker = job_dir / "DEMO_OUTPUT.txt"
        marker.write_text(str(output_mp4), encoding="utf-8")
        return 0

    # 6. Per-platform auto-post toggles (lingora feature)
    #
    # AUTO_POST_FB_ENABLED      → upload to FB Page after render. Default true
    #                             (production-compat). Requires FB_PAGE_ID +
    #                             FB_PAGE_ACCESS_TOKEN.
    # AUTO_POST_TIKTOK_ENABLED  → upload to TikTok after render. Default false.
    #                             Requires TIKTOK_ACCESS_TOKEN.
    #
    # If BOTH are false → "manual mode": skip every auto-upload, ship the
    # rendered MP4 + caption + [📘 FB] [🎵 TikTok] buttons to the Telegram
    # group instead. CEO taps a button to publish on demand.
    fb_enabled = _env_bool("AUTO_POST_FB_ENABLED", default=True)
    tiktok_enabled = _env_bool("AUTO_POST_TIKTOK_ENABLED", default=False)
    page_id = os.environ.get("FB_PAGE_ID")
    token = os.environ.get("FB_PAGE_ACCESS_TOKEN")
    tiktok_token = os.environ.get("TIKTOK_ACCESS_TOKEN")
    tiktok_mode = (os.environ.get("TIKTOK_POST_MODE") or "draft").lower()
    tiktok_privacy = os.environ.get("TIKTOK_PRIVACY_LEVEL", "PUBLIC_TO_EVERYONE")

    # Soft-disable if credentials are missing — manual mode handles delivery.
    if fb_enabled and (not page_id or not token):
        log.warning(
            "AUTO_POST_FB_ENABLED=true but FB_PAGE_ID/FB_PAGE_ACCESS_TOKEN "
            "missing — falling back to manual FB button.",
        )
        fb_enabled = False
    if tiktok_enabled and not tiktok_token:
        log.warning(
            "AUTO_POST_TIKTOK_ENABLED=true but TIKTOK_ACCESS_TOKEN missing — "
            "falling back to manual TikTok button.",
        )
        tiktok_enabled = False

    # Manual mode — neither platform auto-posts. Ship MP4 + buttons to TG.
    if not fb_enabled and not tiktok_enabled:
        log.info(
            "Manual mode (FB_ENABLED=false, TIKTOK_ENABLED=false) — "
            "sending MP4 + buttons to Telegram.",
        )
        state["last_layout"] = layout_type
        state["last_topic"] = intent.topic
        state["post_count"] = int(state.get("post_count", 0)) + 1
        state.pop("last_error", None)
        if not force:
            state["last_post_ts"] = now
        save_state(state)

        prefix = "⚡ <b>Force-render</b>" if force else "🤖 <b>Auto-render</b>"
        layout_emoji = {"quiz": "🎯", "quiz_reverse": "🔁", "phrases": "📚",
                        "whats_this": "🖼️", "whats_board": "🧩",
                        "dialogue": "🎭", "fill_blank": "✍️",
                        "vocab_table": "📋", "compare": "⚖️",
                    "guess_word": "🔤"}.get(layout_type, "📚")
        text = (
            f"{prefix} #{state['post_count']} [{CHANNEL_DIR.name}]\n"
            f"{layout_emoji} Layout: <b>{layout_type}</b>\n"
            f"📝 Topic: <i>{intent.topic}</i>\n"
            f"💡 Manual mode — chọn nền tảng đăng bên dưới:"
        )
        thumb = await asyncio.to_thread(extract_thumbnail, output_mp4)
        await notify_with_video(
            text,
            output_mp4,
            reply_markup=_manual_buttons(job_id, fb_enabled=False, tiktok_enabled=False),
            thumbnail_path=thumb,
        )
        return 0

    # Static layout photo-post branch: only enter if STATIC_LAYOUTS_AS_VIDEO
    # is disabled OR the Ken-Burns conversion above failed (is_static_kb_video
    # flipped back to False). Otherwise output_mp4 is now a Reel-ready video
    # and the regular Reels+Story upload path below handles it.
    if layout_type in ("vocab_table", "compare") and not is_static_kb_video:
        png_path = await asyncio.to_thread(extract_full_frame_png, output_mp4, at_seconds=0.8)
        if png_path is None:
            log.error("Failed to extract PNG frame for %s", layout_type)
            return 1
        try:
            result = await poster.post_photo_to_facebook(
                image_path=png_path, caption=content.caption,
                page_id=page_id, access_token=token,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("FB photo post failed")
            state["last_error"] = str(exc)[:300]
            if not force:
                state["last_post_ts"] = now
            save_state(state)
            prefix_err = "⚡ Force-post lỗi" if force else "❌ Auto-post lỗi"
            await notify(
                f"<b>{prefix_err}</b> [{CHANNEL_DIR.name}]\n"
                f"📝 <i>{intent.topic}</i>\n"
                f"🎬 Layout: {layout_type}\n"
                f"💥 <code>{str(exc)[:300]}</code>"
            )
            return 1
        # Update state + notify, skip the regular video-post path below
        video_id = result.get("post_id", "") or result.get("id", "")
        log.info("FB photo post OK: id=%s", video_id)
        state["last_layout"] = layout_type
        state["last_video_id"] = video_id
        state["last_topic"] = intent.topic
        state["post_count"] = int(state.get("post_count", 0)) + 1
        state.pop("last_error", None)
        if not force:
            state["last_post_ts"] = now
        save_state(state)
        fb_url = f"https://www.facebook.com/{page_id}/posts/{video_id.split('_')[-1]}" if video_id else ""
        prefix = "⚡ <b>Force-post</b>" if force else "🤖 <b>Auto-post</b>"
        emoji = {"vocab_table": "📋", "fill_blank": "✍️", "compare": "⚖️"}.get(layout_type, "🖼️")

        # Admin-only extra: layout-specific enrichment in Telegram notify
        extra = ""
        if layout_type == "compare":
            try:
                lines = [f"\n📊 <b>{content.left_header} → {content.right_header}</b>"]
                for i, p in enumerate(content.pairs[:8], start=1):
                    lt = p.left_target.replace("<", "&lt;").replace(">", "&gt;")
                    rt = p.right_target.replace("<", "&lt;").replace(">", "&gt;")
                    ln = p.left_native.replace("<", "&lt;").replace(">", "&gt;")
                    rn = p.right_native.replace("<", "&lt;").replace(">", "&gt;")
                    lines.append(f"{i}. {p.emoji} <i>{ln}</i> → <i>{rn}</i>")
                    lines.append(f"   <code>{lt}</code> → <code>{rt}</code>")
                extra = "\n" + "\n".join(lines)
            except Exception:
                pass
        elif layout_type == "fill_blank":
            try:
                correct = content.options[content.correct_index]
                extra = (
                    f"\n🟢 <b>Đáp án: {correct}</b> (admin only)"
                    f"\n📝 <i>{content.sentence_template.replace('___', '<b>' + correct + '</b>')}</i>"
                    f"\n🇻🇳 <i>{content.native_translation}</i>"
                )
                if getattr(content, "explanation", ""):
                    expl = content.explanation.strip().replace("<", "&lt;").replace(">", "&gt;")
                    extra += f"\n💡 <i>{expl[:300]}</i>"
            except Exception:
                pass

        await notify(
            f"{prefix} #{state['post_count']} [{CHANNEL_DIR.name}]\n"
            f"{emoji} Layout: <b>{layout_type}</b> (PHOTO POST)\n"
            f"📝 Topic: <i>{intent.topic}</i>"
            f"{extra}\n"
            f"🔗 {fb_url}"
        )
        # Reclaim disk: drop the working dir now that FB has the asset
        await asyncio.to_thread(_cleanup_job_dir, job_dir)
        return 0

    fb_thumb = await asyncio.to_thread(extract_thumbnail, output_mp4)
    fb_url = ""        # canonical URL for the primary FB post (Reel preferred)
    video_id = ""      # Reel video_id (or legacy Video id if reel disabled)
    reel_id = ""
    story_id = ""
    tiktok_publish_id = ""
    tiktok_err = ""
    fb_errs: list[str] = []  # collected per-destination errors (non-fatal)

    # ── 6a. FB upload (if AUTO_POST_FB_ENABLED) ───────────────────────
    # When the FB master is on, each destination toggle decides whether to
    # publish there. Defaults: Reel=ON, Story=ON, Feed Video=OFF (Reels has
    # taken over Page Feed video reach since 2024). Toggles are read from
    # env per channel — see channels/*/.env.
    if fb_enabled:
        reel_destination = _env_bool("FB_POST_REEL_ENABLED", default=True)
        story_destination = _env_bool("FB_POST_STORY_ENABLED", default=True)
        feed_destination = _env_bool("FB_POST_VIDEO_ENABLED", default=False)

        if not (reel_destination or story_destination or feed_destination):
            log.warning(
                "AUTO_POST_FB_ENABLED=true but no FB_POST_* destination enabled — "
                "skipping FB upload entirely.",
            )

        # Build the upload tasks dict. Each task runs in parallel so a 12MB
        # video uploaded to all 3 endpoints completes in ~max-of-three, not sum.
        fb_tasks: dict[str, asyncio.Task] = {}
        if reel_destination:
            fb_tasks["reel"] = asyncio.create_task(
                poster.post_reel_to_facebook(
                    output_mp4, content.caption, page_id, token,
                ),
                name=f"fb-reel-{job_id}",
            )
        if story_destination:
            fb_tasks["story"] = asyncio.create_task(
                poster.post_story_to_facebook(output_mp4, page_id, token),
                name=f"fb-story-{job_id}",
            )
        if feed_destination:
            fb_tasks["feed"] = asyncio.create_task(
                poster.post_to_facebook(
                    video_path=output_mp4, caption=content.caption,
                    page_id=page_id, access_token=token,
                    thumbnail_path=fb_thumb,
                ),
                name=f"fb-feed-{job_id}",
            )

        if fb_tasks:
            fb_results = await asyncio.gather(
                *fb_tasks.values(), return_exceptions=True,
            )
            for dest, res in zip(fb_tasks.keys(), fb_results):
                if isinstance(res, Exception):
                    err_msg = str(res)[:200]
                    log.warning("FB %s upload failed: %s", dest, err_msg)
                    fb_errs.append(f"{dest}: {err_msg}")
                    continue
                if dest == "reel":
                    reel_id = res.get("video_id", "")
                    fb_url = poster.fb_reel_url(reel_id) if reel_id else ""
                    video_id = reel_id  # primary id for state.last_video_id
                elif dest == "story":
                    story_id = res.get("video_id", "")
                elif dest == "feed":
                    feed_video_id = res.get("id", "")
                    # Only overwrite fb_url with feed link if no Reel was posted
                    if not fb_url:
                        fb_url = poster.fb_post_url(feed_video_id) if feed_video_id else ""
                        video_id = feed_video_id

        # If EVERY enabled destination failed, escalate to manual fallback so
        # the CEO can retry from the Telegram group with a button.
        if fb_tasks and not (reel_id or story_id or video_id):
            err_summary = " | ".join(fb_errs)[:300] or "all destinations failed"
            state["last_error"] = err_summary
            if not force:
                state["last_post_ts"] = now
            save_state(state)
            prefix_err = "⚡ Force-post lỗi" if force else "❌ Auto-post lỗi"
            await notify_with_video(
                (
                    f"<b>{prefix_err}</b> [{CHANNEL_DIR.name}]\n"
                    f"📝 <i>{intent.topic}</i>\n"
                    f"🎬 Layout: {layout_type}\n"
                    f"💥 <code>{err_summary}</code>\n"
                    f"💡 Bấm nút để retry manual:"
                ),
                output_mp4,
                reply_markup=_manual_buttons(
                    job_id, fb_enabled=False, tiktok_enabled=tiktok_enabled,
                ),
                thumbnail_path=fb_thumb,
            )
            return 1

    # ── 6b. TikTok upload (if AUTO_POST_TIKTOK_ENABLED) ───────────────
    if tiktok_enabled:
        try:
            import tiktok_poster  # lazy import
            tt_result = await tiktok_poster.post_to_tiktok(
                video_path=output_mp4,
                caption=content.caption,
                access_token=tiktok_token,
                mode=tiktok_mode,
                privacy=tiktok_privacy,
            )
            tiktok_publish_id = tt_result.get("publish_id", "")
            log.info(
                "TikTok upload OK: publish_id=%s mode=%s",
                tiktok_publish_id, tiktok_mode,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("TikTok upload failed (non-fatal): %s", exc)
            tiktok_err = str(exc)[:200]

    # 7. Auto-comment: DISABLED (feedback v8).

    # 8. Update state
    state["last_layout"] = layout_type
    state["last_video_id"] = video_id
    state["last_topic"] = intent.topic
    state["post_count"] = int(state.get("post_count", 0)) + 1
    state.pop("last_error", None)
    if not force:
        state["last_post_ts"] = now
    save_state(state)

    log.info(
        "✅ Auto-post complete (#%d) [force=%s, reel=%s, story=%s, tiktok=%s]",
        state["post_count"], force,
        "ok" if reel_id else "skip",
        "ok" if story_id else "skip",
        "ok" if tiktok_publish_id else ("err" if tiktok_err else "skip"),
    )

    # 9. Telegram report
    layout_emoji = {"quiz": "🎯", "quiz_reverse": "🔁", "phrases": "📚",
                    "whats_this": "🖼️", "whats_board": "🧩",
                    "dialogue": "🎭", "fill_blank": "✍️",
                    "vocab_table": "📋", "compare": "⚖️",
                    "guess_word": "🔤"}.get(layout_type, "📚")
    prefix = "⚡ <b>Force-post</b>" if force else "🤖 <b>Auto-post</b>"

    extra = ""
    if layout_type in ("quiz", "quiz_reverse") and isinstance(content, generator.QuizContent):
        extra = f"\n✅ <b>Đáp án: {content.correct_answer}</b> (admin only)"
        if getattr(content, "explanation", ""):
            expl = content.explanation.strip().replace("<", "&lt;").replace(">", "&gt;")
            extra += f"\n💡 <i>{expl[:300]}</i>"

    status_lines = []
    if fb_enabled:
        if reel_id:
            status_lines.append(f"📘 Reel: {poster.fb_reel_url(reel_id)}")
        if story_id:
            status_lines.append(f"📱 Story: <code>{story_id}</code> (24h)")
        if fb_url and not reel_id:
            # Only the Feed Video fallback path
            status_lines.append(f"📘 FB Video: {fb_url}")
        if fb_errs:
            status_lines.append(
                f"⚠️ FB partial: <code>{' | '.join(fb_errs)[:200]}</code>"
            )
        if not (reel_id or story_id or fb_url):
            status_lines.append("📘 FB: skip (no destination enabled)")
    if tiktok_enabled and tiktok_publish_id:
        status_lines.append(
            f"🎵 TikTok ({tiktok_mode}): <code>{tiktok_publish_id}</code>"
        )
    elif tiktok_enabled and tiktok_err:
        status_lines.append(f"🎵 TikTok lỗi: <code>{tiktok_err}</code>")

    text = (
        f"{prefix} #{state['post_count']} [{CHANNEL_DIR.name}]\n"
        f"{layout_emoji} Layout: <b>{layout_type}</b>\n"
        f"📝 Topic: <i>{intent.topic}</i>"
        f"{extra}\n"
        + "\n".join(status_lines)
    )
    await notify(text)

    # Reclaim disk: drop the working dir now that uploads are done
    await asyncio.to_thread(_cleanup_job_dir, job_dir)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="One-shot auto-post run. Without --force, respects schedule + throttle."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass AUTO_POST_SCHEDULE check AND throttle interval. Used by /auto in bot.",
    )
    args = parser.parse_args()
    try:
        rc = asyncio.run(run_once(force=args.force))
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception("Unhandled error")
        rc = 1
    sys.exit(rc)
