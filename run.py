"""Lingora — local CLI to render one language-learning video.

Usage:
    python run.py                          # interactive wizard
    python run.py --layout phrases --lang de --topic "ordering coffee"
    python run.py --layout dialogue --lang ja --topic "asking directions"

Output:
    channels/<channel-name>/jobs/<auto-...>/output.mp4

The channel folder is just a config holder (one .env per "video config").
The repo ships one starter channel template; create more by copying
.env.example to channels/<your-name>/.env.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

LAYOUTS = [
    "phrases", "quiz", "quiz_reverse",
    "whats_this", "whats_board", "dialogue",
    "fill_blank", "vocab_table", "compare",
    "guess_word",
]

LANG_PRESETS = {
    "de": ("German",   "Đức"),
    "ru": ("Russian",  "Nga"),
    "zh": ("Chinese",  "Trung"),
    "ja": ("Japanese", "Nhật"),
    "ko": ("Korean",   "Hàn"),
    "en": ("English",  "Anh"),
    "fr": ("French",   "Pháp"),
    "es": ("Spanish",  "Tây Ban Nha"),
    "vi": ("Vietnamese", "Việt"),
}

HERE = Path(__file__).resolve().parent


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default


def _wizard() -> argparse.Namespace:
    print("\n=== Lingora — interactive wizard ===\n")

    # Channel
    channels_dir = HERE / "channels"
    existing = sorted([p.name for p in channels_dir.iterdir() if p.is_dir() and (p / ".env").exists()]) if channels_dir.exists() else []
    if existing:
        print(f"Existing channels: {', '.join(existing)}")
        channel = _ask("Channel name", existing[0])
    else:
        print("No channel found. We'll create one for you.")
        channel = _ask("Channel name", "myvideo")
    ch_dir = channels_dir / channel
    env_file = ch_dir / ".env"
    if not env_file.exists():
        ch_dir.mkdir(parents=True, exist_ok=True)
        example = HERE / ".env.example"
        if example.exists():
            shutil.copy2(example, env_file)
            print(f"\n  Created {env_file}")
            print(f"  → Open it and paste your GEMINI_API_KEYS, then re-run.\n")
        else:
            print(f"\n  ERROR: .env.example missing at repo root.")
        sys.exit(1)

    # Layout
    print(f"\nLayouts: {', '.join(LAYOUTS)}")
    layout = _ask("Layout", "phrases")
    if layout not in LAYOUTS:
        sys.exit(f"Unknown layout: {layout}")

    # Language
    print(f"\nLanguages: {', '.join(f'{k}={v[0]}' for k, v in LANG_PRESETS.items())}")
    lang = _ask("Target language code", "de")
    if lang not in LANG_PRESETS:
        print(f"  (unknown code {lang} — passing through, hope your model knows it)")

    # Topic
    topic = _ask("Topic (e.g. 'ordering coffee')", "ordering coffee")

    # Voice
    voice = _ask("Voice (any/female/male)", "any")
    if voice not in ("any", "female", "male"):
        voice = "any"

    return argparse.Namespace(
        channel=channel, layout=layout, lang=lang, topic=topic, voice=voice, count=10,
    )


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", default="myvideo", help="Channel folder under channels/ (holds .env)")
    p.add_argument("--layout", choices=LAYOUTS, help="Video layout")
    p.add_argument("--lang", help="Target language code (de, ru, ja, ko, zh, en, fr, es, vi, ...)")
    p.add_argument("--topic", help='Topic, e.g. "ordering coffee"')
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--voice", choices=["female", "male", "any"], default="any")
    args = p.parse_args()

    if not (args.layout and args.lang and args.topic):
        return _wizard()
    return args


def _build_request(args: argparse.Namespace) -> str:
    """Build the natural-language request that the generator parses.

    Requests are written in Vietnamese because that's what topic_picker
    emits in production, and the prompts/examples in the generator are
    tuned for vi-native requests. The target language stays whatever you
    asked for via --lang.
    """
    lang_vi = LANG_PRESETS.get(args.lang, (args.lang.upper(), args.lang.upper()))[1]
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

    # Env handoff to the engine (auto_post reads CHANNEL_DIR + .env)
    os.environ["CHANNEL_DIR"] = str(channel_dir)
    os.environ["AUTO_POST_ENABLED"] = "true"
    os.environ["DEMO_MODE"] = "1"              # skip FB upload + Telegram notify
    os.environ["_DEMO_LAYOUT"] = args.layout
    os.environ["_DEMO_REQUEST"] = request
    os.environ["DEFAULT_TARGET_LANG"] = args.lang
    os.environ["DEFAULT_VOICE_GENDER"] = args.voice
    os.environ.pop("TELEGRAM_BOT_TOKEN", None)
    os.environ.pop("TELEGRAM_NOTIFY_BOT_TOKEN", None)

    # Import engine — must happen after env is set
    bot_dir = HERE / "bot"
    sys.path.insert(0, str(bot_dir))
    import auto_post  # noqa: E402

    print(f"\n[lingora] channel={args.channel} layout={args.layout} lang={args.lang}")
    print(f"[lingora] topic={args.topic!r}")
    print(f"[lingora] working in {channel_dir}\n")

    rc = asyncio.run(auto_post.run_once(force=True))
    if rc != 0:
        return rc

    # Surface the rendered MP4 path (DEMO_MODE writes a marker)
    jobs = channel_dir / "jobs"
    markers = sorted(jobs.glob("*/DEMO_OUTPUT.txt"), key=lambda p: p.stat().st_mtime, reverse=True) if jobs.exists() else []
    if markers:
        mp4 = Path(markers[0].read_text(encoding="utf-8").strip())
        if mp4.exists():
            size_mb = mp4.stat().st_size / 1024 / 1024
            print(f"\n[lingora] ✓ rendered: {mp4} ({size_mb:.1f} MB)")
            return 0

    # Fallback: find the newest output.mp4 under jobs/
    mp4s = sorted(jobs.glob("*/output.mp4"), key=lambda p: p.stat().st_mtime, reverse=True) if jobs.exists() else []
    if mp4s:
        print(f"\n[lingora] ✓ rendered: {mp4s[0]}")
        return 0

    print("\n[lingora] WARNING: render returned 0 but no output.mp4 found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
