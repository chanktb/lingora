"""Lingora — local CLI to render one language-learning video.

Usage:
    python run.py                          # interactive wizard (recommended)
    python run.py --layout phrases --lang de --topic "ordering coffee"

Output:
    channels/<channel>/jobs/<auto-…>/output.mp4

The wizard auto-skips any question whose value is already set in the
channel's .env file (e.g. if DEFAULT_TARGET_LANG=de is set, it won't
ask which language).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── Catalog ──────────────────────────────────────────────────────────
LAYOUTS = [
    ("phrases",       "Phrases — 8-10 short sentences with voiceover"),
    ("quiz",          "Quiz — one phrase, three options, reveal answer"),
    ("dialogue",      "Dialogue — 6-8 turn mini-skit between 2 characters"),
    ("whats_this",    "What's this — 10 vocab items with AI illustration"),
    ("whats_board",   "What's on the board — 9-grid cheat sheet"),
    ("vocab_table",   "Vocab table — 8 items + character mascot (static poster)"),
    ("compare",       "Compare — 8 pairs side-by-side (static poster)"),
    ("fill_blank",    "Fill-blank — one sentence with a missing word + 3 options"),
    ("guess_word",    "Guess-word — reveal target word letter by letter"),
    ("quiz_reverse",  "Reverse quiz — guess the meaning of a phrase"),
]

LANGS = [
    ("de", "German",     "Đức"),
    ("ru", "Russian",    "Nga"),
    ("zh", "Chinese",    "Trung"),
    ("ja", "Japanese",   "Nhật"),
    ("ko", "Korean",     "Hàn"),
    ("en", "English",    "Anh"),
    ("fr", "French",     "Pháp"),
    ("es", "Spanish",    "Tây Ban Nha"),
    ("vi", "Vietnamese", "Việt"),
]
LANG_VI = {code: vi for code, _, vi in LANGS}

VOICES = [
    ("any",    "Any (let the engine pick)"),
    ("female", "Female"),
    ("male",   "Male"),
]


# ── Helpers ──────────────────────────────────────────────────────────
def _read_env(env_path: Path) -> dict[str, str]:
    """Minimal .env parser — no quoting, no expansion. Enough to check keys."""
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def _menu(prompt: str, items: list[tuple[str, str]], *, default_idx: int = 0) -> str:
    """Show numbered menu. items = [(key, label), ...]. Returns chosen key."""
    print(f"\n{prompt}")
    for i, (_, label) in enumerate(items, 1):
        marker = "  ← default" if i - 1 == default_idx else ""
        print(f"  {i:2}. {label}{marker}")
    while True:
        raw = input(f"Choice [1-{len(items)}, Enter = {default_idx + 1}]: ").strip()
        if not raw:
            return items[default_idx][0]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(items):
                return items[idx][0]
        except ValueError:
            pass
        print(f"  → please enter a number between 1 and {len(items)}")


def _ask_text(prompt: str, *, default: str = "", allow_empty: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        val = input(f"{prompt}{suffix}: ").strip()
        if val:
            return val
        if default:
            return default
        if allow_empty:
            return ""
        print("  → please type something")


def _confirm(prompt: str, *, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default_yes
    return raw.startswith("y")


# ── Channel discovery & setup ────────────────────────────────────────
def _pick_channel(channels_dir: Path) -> Path:
    """Pick or create a channel folder. Exits if .env needs to be filled in."""
    existing = sorted(
        [p for p in channels_dir.iterdir() if p.is_dir() and (p / ".env").exists()]
    ) if channels_dir.exists() else []

    if len(existing) == 0:
        # First-time setup: scaffold channels/myvideo/.env from template
        ch = channels_dir / "myvideo"
        ch.mkdir(parents=True, exist_ok=True)
        template = HERE / ".env.example"
        env_file = ch / ".env"
        if template.exists():
            shutil.copy2(template, env_file)
            print(f"\n  Created {env_file}")
            print(f"  → Open it, paste your GEMINI_API_KEYS, then re-run.")
            print(f"  → Free key: https://aistudio.google.com/apikey")
        else:
            print(f"\n  ERROR: .env.example missing at repo root.")
        sys.exit(1)

    if len(existing) == 1:
        return existing[0]

    items = [(p.name, p.name) for p in existing]
    chosen = _menu("Pick a channel:", items, default_idx=0)
    return channels_dir / chosen


def _wizard() -> argparse.Namespace:
    print("\n=== Lingora — interactive wizard ===")

    channel_dir = _pick_channel(HERE / "channels")
    env = _read_env(channel_dir / ".env")

    # Layout: always ask (changes per video)
    layout = _menu("Pick a video layout:", LAYOUTS, default_idx=0)

    # Language: skip if .env has DEFAULT_TARGET_LANG
    lang = env.get("DEFAULT_TARGET_LANG", "").strip()
    if lang:
        print(f"\nLanguage: {lang} (from .env DEFAULT_TARGET_LANG — skipping prompt)")
    else:
        lang = _menu(
            "Pick the language to teach:",
            [(code, f"{name} ({code})") for code, name, _ in LANGS],
            default_idx=0,
        )

    # Voice: skip if .env has DEFAULT_VOICE_GENDER
    voice = env.get("DEFAULT_VOICE_GENDER", "").strip().lower()
    if voice in {"any", "female", "male"}:
        print(f"Voice: {voice} (from .env DEFAULT_VOICE_GENDER — skipping prompt)")
    else:
        voice = _menu("Pick a voice:", VOICES, default_idx=0)

    # Topic: always ask, free-text
    print()
    topic = _ask_text("Topic (e.g. 'ordering coffee', 'kitchen items')")

    # Summary + confirm
    print("\n─── Summary ───")
    print(f"  channel : {channel_dir.name}")
    print(f"  layout  : {layout}")
    print(f"  lang    : {lang}")
    print(f"  voice   : {voice}")
    print(f"  topic   : {topic}")
    if not _confirm("\nRender now?"):
        sys.exit("Cancelled.")

    return argparse.Namespace(
        channel=channel_dir.name,
        layout=layout, lang=lang, topic=topic, voice=voice, count=10,
    )


# ── CLI (non-interactive) ────────────────────────────────────────────
def _parse_cli() -> argparse.Namespace:
    layout_keys = [k for k, _ in LAYOUTS]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", default="myvideo", help="Channel folder under channels/")
    p.add_argument("--layout", choices=layout_keys, help="Video layout")
    p.add_argument("--lang", help="Target language code (de, ru, ja, ko, zh, en, fr, es, vi, ...)")
    p.add_argument("--topic", help='Topic, e.g. "ordering coffee"')
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--voice", choices=["female", "male", "any"], default="any")
    args = p.parse_args()

    if not (args.layout and args.lang and args.topic):
        return _wizard()
    return args


# ── Request builder ──────────────────────────────────────────────────
def _build_request(args: argparse.Namespace) -> str:
    """Build the natural-language request the generator parses.

    Requests are written in Vietnamese because that's what topic_picker
    emits in production and the generator's examples are tuned for it.
    The target language is whatever --lang says.
    """
    lang_vi = LANG_VI.get(args.lang, args.lang.upper())
    voice_clause = f", giọng {args.voice}" if args.voice != "any" else ""

    if args.layout == "phrases":
        return f"{args.count} câu tiếng {lang_vi} về {args.topic}{voice_clause}"
    if args.layout == "quiz":
        return f"Quiz: '{args.topic}' trong tiếng {lang_vi} nói thế nào?{voice_clause}"
    if args.layout == "quiz_reverse":
        return (f"Reverse quiz về chủ đề '{args.topic}': chọn 1 cụm từ tiếng {lang_vi} "
                f"phổ biến rồi yêu cầu user đoán nghĩa tiếng Việt.{voice_clause}")
    if args.layout == "whats_this":
        return (f"Whats-this visual vocab — theme '{args.topic}' bằng tiếng {lang_vi}. "
                f"Sinh 10 items concrete (noun đơn hoặc verb đơn), MỖI item có image_prompt "
                f"mô tả cụ thể để gen AI illustration.{voice_clause}")
    if args.layout == "whats_board":
        return (f"Whats-board 9-grid cheat sheet — theme '{args.topic}' bằng tiếng {lang_vi}. "
                f"Sinh ĐÚNG 9 items concrete NOUN imageable. MỖI item có image_prompt "
                f"mô tả tượng hình.{voice_clause}")
    if args.layout == "dialogue":
        return (f"Dialogue mini-skit — scenario '{args.topic}' bằng tiếng {lang_vi}. "
                f"Sinh 2 nhân vật (A và B) + cảnh nền + 6-8 lượt thoại. "
                f"Mỗi lượt có target text + phiên âm + dịch tiếng Việt.{voice_clause}")
    if args.layout == "fill_blank":
        return (f"Fill-blank short quiz — chủ đề '{args.topic}' bằng tiếng {lang_vi}. "
                f"Sinh 1 câu duy nhất có 1 từ blank ___ + 3 options + "
                f"image_prompt photo realistic người đang làm hành động liên quan câu.")
    if args.layout == "vocab_table":
        return (f"Vocab table static poster — chủ đề '{args.topic}' bằng tiếng {lang_vi}. "
                f"Sinh 8 items concrete + character mascot fitting the theme.")
    if args.layout == "compare":
        return (f"Compare 2-column static poster — chủ đề '{args.topic}' bằng tiếng {lang_vi}. "
                f"Sinh 8 cặp so sánh (left=basic/casual/wrong, right=fluent/formal/correct).")
    if args.layout == "guess_word":
        return (f"Guess-word: chọn 1 từ tiếng {lang_vi} chủ đề '{args.topic}' rồi reveal "
                f"từng chữ cái một, cuối cùng đọc tiếng Việt.{voice_clause}")
    sys.exit(f"Unsupported layout: {args.layout}")


# ── Main ─────────────────────────────────────────────────────────────
def main() -> int:
    args = _parse_cli()

    channel_dir = HERE / "channels" / args.channel
    env_file = channel_dir / ".env"
    if not env_file.exists():
        sys.exit(
            f"ERROR: {env_file} not found.\n"
            f"  Create it: cp .env.example channels/{args.channel}/.env\n"
            f"  Then open it and paste your GEMINI_API_KEYS."
        )

    request = _build_request(args)

    # Hand off to the engine (auto_post reads CHANNEL_DIR + .env)
    os.environ["CHANNEL_DIR"] = str(channel_dir)
    os.environ["AUTO_POST_ENABLED"] = "true"
    os.environ["DEMO_MODE"] = "1"               # skip FB upload + Telegram notify
    os.environ["_DEMO_LAYOUT"] = args.layout
    os.environ["_DEMO_REQUEST"] = request
    os.environ["DEFAULT_TARGET_LANG"] = args.lang
    os.environ["DEFAULT_VOICE_GENDER"] = args.voice
    os.environ.pop("TELEGRAM_BOT_TOKEN", None)
    os.environ.pop("TELEGRAM_NOTIFY_BOT_TOKEN", None)

    bot_dir = HERE / "bot"
    sys.path.insert(0, str(bot_dir))
    import auto_post  # noqa: E402

    print(f"\n[lingora] channel={args.channel} layout={args.layout} lang={args.lang}")
    print(f"[lingora] topic={args.topic!r}")
    print(f"[lingora] working in {channel_dir}\n")

    rc = asyncio.run(auto_post.run_once(force=True))
    if rc != 0:
        return rc

    jobs = channel_dir / "jobs"
    markers = sorted(jobs.glob("*/DEMO_OUTPUT.txt"), key=lambda p: p.stat().st_mtime, reverse=True) if jobs.exists() else []
    if markers:
        mp4 = Path(markers[0].read_text(encoding="utf-8").strip())
        if mp4.exists():
            size_mb = mp4.stat().st_size / 1024 / 1024
            print(f"\n[lingora] ✓ rendered: {mp4} ({size_mb:.1f} MB)")
            return 0

    mp4s = sorted(jobs.glob("*/output.mp4"), key=lambda p: p.stat().st_mtime, reverse=True) if jobs.exists() else []
    if mp4s:
        print(f"\n[lingora] ✓ rendered: {mp4s[0]}")
        return 0

    print("\n[lingora] WARNING: render returned 0 but no output.mp4 found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
