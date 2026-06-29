"""Lingora — local CLI to render one language-learning video.

Usage:
    python run.py                          # interactive wizard (recommended)
    python run.py --layout phrases --lang de --topic "ordering coffee"

Output:
    channels/<channel>/jobs/<auto-…>/output.mp4

Wizard behaviour:
    - First run with no channel → interactive setup creates one for you
    - Skips any question whose value is already set in the channel's .env
    - Topic step offers: type your own, or auto-generate (channel-aware,
      avoids the last 100 topics already used)
"""
from __future__ import annotations

import argparse
import asyncio
import json
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
    ("de", "German"),
    ("ru", "Russian"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("en", "English"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("vi", "Vietnamese"),
]
LANG_NAME = {code: name for code, name in LANGS}
LANG_VI = {
    "de": "Đức", "ru": "Nga", "zh": "Trung", "ja": "Nhật", "ko": "Hàn",
    "en": "Anh", "fr": "Pháp", "es": "Tây Ban Nha", "vi": "Việt",
}

VOICES = [
    ("any",    "Any (let the engine pick)"),
    ("female", "Female"),
    ("male",   "Male"),
]


# ── Generic prompts ──────────────────────────────────────────────────
def _menu(prompt: str, items: list[tuple[str, str]], *, default_idx: int = 0) -> str:
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


# ── .env helpers ─────────────────────────────────────────────────────
def _read_env(env_path: Path) -> dict[str, str]:
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


def _apply_env_to_os(env: dict[str, str]) -> None:
    """Push channel .env values into os.environ so the engine can read them.

    Use setdefault so explicit env (CLI/PowerShell) still wins.
    """
    for k, v in env.items():
        if v and k not in os.environ:
            os.environ[k] = v


def _write_env_from_template(target: Path, overrides: dict[str, str]) -> None:
    """Copy .env.example to target, replacing KEY= lines with overrides."""
    template = HERE / ".env.example"
    if not template.exists():
        sys.exit("ERROR: .env.example missing at repo root.")
    target.parent.mkdir(parents=True, exist_ok=True)
    lines_out: list[str] = []
    seen: set[str] = set()
    for raw in template.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if (not stripped) or stripped.startswith("#") or "=" not in stripped:
            lines_out.append(raw); continue
        key = stripped.split("=", 1)[0].strip()
        if key in overrides:
            lines_out.append(f"{key}={overrides[key]}")
            seen.add(key)
        else:
            lines_out.append(raw)
    # Append any overrides not present in template (rare)
    extras = [k for k in overrides if k not in seen]
    if extras:
        lines_out.append("")
        lines_out.append("# ── added by wizard ──")
        for k in extras:
            lines_out.append(f"{k}={overrides[k]}")
    target.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


# ── Channel: pick existing or create new ─────────────────────────────
def _create_channel_wizard(channels_dir: Path) -> Path:
    print("\n=== Lingora — create your first channel ===")
    print(
        "A 'channel' is just a config folder — one .env per video config.\n"
        "You can add more later by re-running this wizard or copying the folder.\n"
    )

    name = _ask_text("Channel folder name", default="myvideo")
    display = _ask_text("Display name (used on-screen)", default="My Language Channel")
    niche = _ask_text(
        "Niche / scope (what kind of videos? — used by auto-topic)\n"
        "  e.g. 'everyday German for travelers', 'JLPT N5 vocab', 'K-pop & Korean culture'\n"
        "> ",
        default="general language learning",
    )

    target = _menu(
        "Language to teach (target):",
        [(c, f"{n} ({c})") for c, n in LANGS],
        default_idx=0,
    )
    native = _menu(
        "On-screen translation + native voice (native):",
        [(c, f"{n} ({c})") for c, n in LANGS],
        default_idx=8,  # default Vietnamese
    )
    voice = _menu("Voice preference:", VOICES, default_idx=0)

    print(
        "\nGemini API key (required — generates the content)\n"
        "  Free key: https://aistudio.google.com/apikey"
    )
    key = _ask_text("Paste GEMINI_API_KEY here")

    print("\nOptional: Cloudflare Workers AI for scene/character images.")
    print("  Format: account_id:api_token  (multiple accounts: id1:tok1,id2:tok2)")
    cf = _ask_text("Paste CLOUDFLARE_ACCOUNTS (Enter to skip)", allow_empty=True)

    overrides = {
        "CHANNEL_NAME": display,
        "NICHE": niche,
        "DEFAULT_TARGET_LANG": target,
        "DEFAULT_NATIVE_LANG": native,
        "DEFAULT_VOICE_GENDER": voice,
        "GEMINI_API_KEY": key,
        "GEMINI_API_KEYS": key,  # single-key user — auto_post falls back fine
    }
    if cf:
        overrides["CLOUDFLARE_ACCOUNTS"] = cf

    ch_dir = channels_dir / name
    env_file = ch_dir / ".env"
    if env_file.exists() and not _confirm(
        f"\n{env_file} already exists. Overwrite?", default_yes=False,
    ):
        sys.exit("Cancelled.")
    _write_env_from_template(env_file, overrides)
    print(f"\n  ✓ Created {env_file}")
    return ch_dir


def _pick_or_create_channel(channels_dir: Path) -> Path:
    existing = sorted(
        [p for p in channels_dir.iterdir() if p.is_dir() and (p / ".env").exists()]
    ) if channels_dir.exists() else []

    if not existing:
        return _create_channel_wizard(channels_dir)

    if len(existing) == 1:
        return existing[0]

    items = [(p.name, p.name) for p in existing] + [("__new__", "+ Create a new channel")]
    chosen = _menu("Pick a channel:", items, default_idx=0)
    if chosen == "__new__":
        return _create_channel_wizard(channels_dir)
    return channels_dir / chosen


# ── Auto-topic generation ────────────────────────────────────────────
def _recent_topics(channel_dir: Path, limit: int = 100) -> list[str]:
    state_file = channel_dir / "auto_post_state.json"
    if not state_file.exists():
        return []
    try:
        data = json.loads(state_file.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    used = data.get("used_topics") or []
    last_topic = data.get("last_topic")
    if last_topic and last_topic not in used:
        used = [*used, last_topic]
    return used[-limit:]


def _auto_topic(channel_dir: Path, env: dict[str, str], target_code: str) -> str:
    """Ask Gemini for one fresh topic that avoids the recent history."""
    niche = env.get("NICHE") or env.get("CHANNEL_NAME") or "general language learning"
    target_name = LANG_NAME.get(target_code, target_code.upper())
    used = _recent_topics(channel_dir, limit=100)
    avoid_block = "\n".join(f"- {t}" for t in used) if used else "(none yet)"

    prompt = (
        f"Pick ONE fresh topic for a short {target_name} language-learning video.\n"
        f"Channel niche / scope: {niche}\n\n"
        f"AVOID these topics already used (do not repeat or rephrase):\n{avoid_block}\n\n"
        f"Output ONE topic line, 3-8 English words, concrete and teachable. "
        f"No numbering, no quotes, no period. Just the topic line."
    )

    sys.path.insert(0, str(HERE / "bot"))
    import generator
    from google.genai import types

    generator._reset_gemini_keys()  # we set env vars after import-time
    model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
    try:
        resp = generator._call_gemini(
            client=None, model=model, contents=prompt,
            config=types.GenerateContentConfig(temperature=1.1, max_output_tokens=80),
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"  ! auto-topic failed: {exc}")
        return ""

    text = (resp.text or "").strip().splitlines()[0]
    return text.strip().strip('"').strip("'").strip("•- ").rstrip(".")


def _topic_phase(channel_dir: Path, env: dict[str, str], target_code: str) -> str:
    """Either type a topic or have one auto-generated (with regen + override)."""
    niche = env.get("NICHE") or "(no niche set)"
    items = [
        ("manual", "Type your own"),
        ("auto",   f"Auto-generate from niche: {niche!r}"),
    ]
    mode = _menu("Topic:", items, default_idx=0)
    if mode == "manual":
        return _ask_text("\nTopic (e.g. 'ordering coffee', 'kitchen items')")

    while True:
        print("\n[generating…]")
        suggestion = _auto_topic(channel_dir, env, target_code)
        if not suggestion:
            print("  Falling back to manual input.")
            return _ask_text("Topic")
        print(f"\nSuggested topic: \"{suggestion}\"")
        choice = _menu(
            "What now?",
            [("use", "Use this topic"),
             ("regen", "Regenerate another"),
             ("manual", "Type my own instead")],
            default_idx=0,
        )
        if choice == "use":
            return suggestion
        if choice == "manual":
            return _ask_text("Topic", default=suggestion)
        # regen → loop


# ── Main wizard ──────────────────────────────────────────────────────
def _wizard() -> argparse.Namespace:
    print("\n=== Lingora — interactive wizard ===")

    channel_dir = _pick_or_create_channel(HERE / "channels")
    env = _read_env(channel_dir / ".env")
    _apply_env_to_os(env)  # so Gemini auto-topic can read keys

    layout = _menu("Pick a video layout:", LAYOUTS, default_idx=0)

    target = env.get("DEFAULT_TARGET_LANG", "").strip()
    if target:
        print(f"\nTarget language: {target} ({LANG_NAME.get(target, '?')})  ← from .env")
    else:
        target = _menu(
            "Pick the language to teach (target):",
            [(c, f"{n} ({c})") for c, n in LANGS],
            default_idx=0,
        )

    native = env.get("DEFAULT_NATIVE_LANG", "").strip()
    if native:
        print(f"Native language: {native} ({LANG_NAME.get(native, '?')})  ← from .env")
    else:
        native = _menu(
            "Pick the on-screen translation language (native):",
            [(c, f"{n} ({c})") for c, n in LANGS],
            default_idx=8,
        )

    voice = env.get("DEFAULT_VOICE_GENDER", "").strip().lower()
    if voice in {"any", "female", "male"}:
        print(f"Voice: {voice}  ← from .env")
    else:
        voice = _menu("Pick a voice:", VOICES, default_idx=0)

    topic = _topic_phase(channel_dir, env, target)

    print("\n─── Summary ───")
    print(f"  channel : {channel_dir.name}")
    print(f"  layout  : {layout}")
    print(f"  target  : {target} ({LANG_NAME.get(target, '?')})")
    print(f"  native  : {native} ({LANG_NAME.get(native, '?')})")
    print(f"  voice   : {voice}")
    print(f"  topic   : {topic}")
    if not _confirm("\nRender now?"):
        sys.exit("Cancelled.")

    return argparse.Namespace(
        channel=channel_dir.name, layout=layout, lang=target, native=native,
        topic=topic, voice=voice, count=10,
    )


# ── CLI (non-interactive) ────────────────────────────────────────────
def _parse_cli() -> argparse.Namespace:
    layout_keys = [k for k, _ in LAYOUTS]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", default="myvideo", help="Channel folder under channels/")
    p.add_argument("--layout", choices=layout_keys, help="Video layout")
    p.add_argument("--lang", help="Target language code")
    p.add_argument("--native", help="Native (translation) language code")
    p.add_argument("--topic", help='Topic text (omit to use auto-gen in wizard)')
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--voice", choices=["female", "male", "any"], default="any")
    args = p.parse_args()
    if not (args.layout and args.lang and args.topic):
        return _wizard()
    args.native = args.native or "vi"
    return args


# ── Request builder ──────────────────────────────────────────────────
def _build_request(args: argparse.Namespace) -> str:
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
            f"  Run RUN.bat / ./RUN.sh with no args to create one interactively."
        )

    request = _build_request(args)

    os.environ["CHANNEL_DIR"] = str(channel_dir)
    os.environ["AUTO_POST_ENABLED"] = "true"
    os.environ["DEMO_MODE"] = "1"
    os.environ["_DEMO_LAYOUT"] = args.layout
    os.environ["_DEMO_REQUEST"] = request
    os.environ["DEFAULT_TARGET_LANG"] = args.lang
    os.environ["DEFAULT_NATIVE_LANG"] = args.native
    os.environ["DEFAULT_VOICE_GENDER"] = args.voice
    os.environ.pop("TELEGRAM_BOT_TOKEN", None)
    os.environ.pop("TELEGRAM_NOTIFY_BOT_TOKEN", None)

    bot_dir = HERE / "bot"
    sys.path.insert(0, str(bot_dir))
    import auto_post  # noqa: E402

    print(f"\n[lingora] channel={args.channel} layout={args.layout} "
          f"target={args.lang} native={args.native}")
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
