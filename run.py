"""Lingora — local CLI to render one language-learning video.

Usage:
    python run.py                          # interactive wizard (recommended)
    python run.py --layout phrases --lang de --topic "ordering coffee"

Output:
    channels/<channel>/jobs/<auto-…>/output.mp4

Wizard behaviour:
    - First run: uses channels/example/ (shipped with the repo) and asks
      for your free Gemini API key the first time only — it's written
      back into the channel .env so next runs go straight to the layout
      picker.
    - Skips any question whose value is already set in the channel's .env
      (target language, native language, voice).
    - Topic step defaults to auto-generate (channel-niche aware, avoids
      the last 100 topics already used). Just type a topic to override.
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
    ("vocab_card",    "Vocab card, 1 word + illustration + multi-language grid"),
    ("conjugation",   "Conjugation, 1 verb + 6 personal-pronoun forms (RU only)"),
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
LANG_NAME_EN = {code: name for code, name in LANGS}
LANG_NAME_VI = {
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
    """Push channel .env values into os.environ so the engine can read them."""
    for k, v in env.items():
        if v and k not in os.environ:
            os.environ[k] = v


def _patch_env_file(env_path: Path, updates: dict[str, str]) -> None:
    """Rewrite specific KEY= lines in an existing .env, preserving comments + order."""
    lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines() if env_path.exists() else []
    remaining = dict(updates)
    out: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if (not stripped) or stripped.startswith("#") or "=" not in stripped:
            out.append(raw); continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(raw)
    if remaining:
        out.append("")
        out.append("# ── added by wizard ──")
        for k, v in remaining.items():
            out.append(f"{k}={v}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _write_env_from_template(target: Path, overrides: dict[str, str]) -> None:
    """Copy .env.example to target then patch overrides."""
    template = HERE / ".env.example"
    if not template.exists():
        sys.exit("ERROR: .env.example missing at repo root.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, target)
    _patch_env_file(target, overrides)


# ── Channel: pick existing or create new ─────────────────────────────
def _create_channel_wizard(channels_dir: Path) -> Path:
    print("\n=== Create a new channel ===")
    print(
        "A 'channel' is just a config folder — one .env per video config.\n"
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
    print("  Format: account_id:api_token  (multiple: id1:tok1,id2:tok2)")
    cf = _ask_text("Paste CLOUDFLARE_ACCOUNTS (Enter to skip)", allow_empty=True)

    overrides = {
        "CHANNEL_NAME": display,
        "NICHE": niche,
        "DEFAULT_TARGET_LANG": target,
        "DEFAULT_NATIVE_LANG": native,
        "DEFAULT_VOICE_GENDER": voice,
        "GEMINI_API_KEY": key,
        "GEMINI_API_KEYS": key,
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
        # No channel at all (user deleted the shipped example)
        return _create_channel_wizard(channels_dir)

    if len(existing) == 1:
        return existing[0]

    items = [(p.name, p.name) for p in existing] + [("__new__", "+ Create a new channel")]
    chosen = _menu("Pick a channel:", items, default_idx=0)
    if chosen == "__new__":
        return _create_channel_wizard(channels_dir)
    return channels_dir / chosen


def _ensure_api_key(channel_dir: Path, env: dict[str, str]) -> dict[str, str]:
    """If the channel .env has no Gemini key, prompt for one and write it back."""
    if env.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEYS"):
        return env
    print(
        f"\nChannel '{channel_dir.name}' has no GEMINI_API_KEY yet.\n"
        f"  Get a free one (takes ~30 sec): https://aistudio.google.com/apikey"
    )
    key = _ask_text("Paste GEMINI_API_KEY here")
    _patch_env_file(channel_dir / ".env", {"GEMINI_API_KEY": key, "GEMINI_API_KEYS": key})
    print(f"  ✓ Saved to {channel_dir / '.env'}")
    return _read_env(channel_dir / ".env")


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
    target_name = LANG_NAME_EN.get(target_code, target_code.upper())
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

    generator._reset_gemini_keys()
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

    text = (resp.text or "").strip().splitlines()[0] if resp.text else ""
    return text.strip().strip('"').strip("'").strip("•- ").rstrip(".")


def _topic_phase(channel_dir: Path, env: dict[str, str], target_code: str) -> str:
    """Default = auto-generate. Typing any text overrides it as manual input."""
    niche = env.get("NICHE") or "your channel niche"
    print(
        f"\nTopic (default = auto-generate based on niche: '{niche}')\n"
        f"  Press Enter for auto, or type your own topic:"
    )
    raw = input("> ").strip()
    if raw:
        return raw

    while True:
        print("\n[generating…]")
        suggestion = _auto_topic(channel_dir, env, target_code)
        if not suggestion:
            print("  Falling back to manual input.")
            return _ask_text("Topic")
        print(f"\nSuggested topic: \"{suggestion}\"")
        choice = _menu(
            "What now?",
            [("use",    "Use this topic"),
             ("regen",  "Regenerate another"),
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
    env = _ensure_api_key(channel_dir, env)
    _apply_env_to_os(env)  # so Gemini auto-topic can read keys

    layout = _menu("Pick a video layout:", LAYOUTS, default_idx=0)

    target = env.get("DEFAULT_TARGET_LANG", "").strip()
    if target:
        print(f"\nTarget language: {target} ({LANG_NAME_EN.get(target, '?')})  ← from .env")
    else:
        target = _menu(
            "Pick the language to teach (target):",
            [(c, f"{n} ({c})") for c, n in LANGS],
            default_idx=0,
        )

    native = env.get("DEFAULT_NATIVE_LANG", "").strip()
    if native:
        print(f"Native language: {native} ({LANG_NAME_EN.get(native, '?')})  ← from .env")
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
    print(f"  target  : {target} ({LANG_NAME_EN.get(target, '?')})")
    print(f"  native  : {native} ({LANG_NAME_EN.get(native, '?')})")
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
    p.add_argument("--channel", default="example", help="Channel folder under channels/")
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


# ── Request builder (uses topic_picker — same as production engine) ──
def _build_request(args: argparse.Namespace) -> str:
    """Build the natural-language request via topic_picker.

    topic_picker._format_<layout>_request handles native-language switching:
        native=vi  → Vietnamese phrasing (default pool style)
        native=en  → prepends _EN_META so Gemini emits idiomatic English
        native=xx  → falls through to vi pool (engine limitation; non-vi
                     non-en channels should set DEFAULT_NATIVE_LANG=en)
    Using topic_picker keeps the public CLI 1:1 with what the prod engine
    sends, so bug fixes there flow through without parallel maintenance.
    """
    sys.path.insert(0, str(HERE / "bot"))
    import topic_picker as tp

    # Map target code → name as written in the *native* language.
    if args.native == "vi":
        target_name = LANG_NAME_VI.get(args.lang, args.lang.upper())
    else:
        target_name = LANG_NAME_EN.get(args.lang, args.lang.upper())

    # Tell topic_picker which native we're in so its format helpers branch.
    tp._NATIVE_LANG = args.native

    topic = args.topic
    layout = args.layout

    if layout == "phrases":
        return tp._format_phrases_request(topic, target_lang_name=target_name)
    if layout == "quiz":
        return tp._format_quiz_request(topic, target_lang_name=target_name)
    if layout == "quiz_reverse":
        return tp._format_quiz_reverse_request(topic, target_lang_name=target_name)
    if layout == "whats_this":
        return tp._format_whats_this_request(topic, target_lang_name=target_name)
    if layout == "whats_board":
        return tp._format_whats_board_request(topic, target_lang_name=target_name)
    if layout == "guess_word":
        return tp._format_guess_word_request(topic, target_lang_name=target_name)
    if layout == "vocab_card":
        return tp._format_vocab_card_request(topic, target_lang_name=target_name)
    if layout == "conjugation":
        # topic is an ASCII slug (e.g. "nhan") or Cyrillic infinitive; look up
        # the matching verb entry so we send the full triple to Gemini. Empty
        # topic just picks the first verb in the pool.
        pool = tp._CONJUGATION_VERBS.get(args.lang, [])
        if not pool:
            sys.exit(f"conjugation: no verb pool for target_lang={args.lang}")
        pick = next((v for v in pool if v[2] == topic or v[0] == topic), pool[0])
        verb_target, verb_native, verb_slug = pick
        return tp._format_conjugation_request(verb_target, verb_slug, verb_native, target_name)
    if layout == "dialogue":
        return tp._format_dialogue_request(topic, target_lang_name=target_name)
    if layout == "fill_blank":
        return tp._format_fill_blank_request(topic, target_lang_name=target_name)
    if layout == "vocab_table":
        return tp._format_vocab_table_request(topic, target_lang_name=target_name)
    if layout == "compare":
        return tp._format_compare_request(topic, target_lang_name=target_name)
    sys.exit(f"Unsupported layout: {layout}")


# ── Main ─────────────────────────────────────────────────────────────
def main() -> int:
    args = _parse_cli()

    channel_dir = HERE / "channels" / args.channel
    env_file = channel_dir / ".env"
    if not env_file.exists():
        sys.exit(
            f"ERROR: {env_file} not found.\n"
            f"  Run RUN.bat / ./RUN.sh with no args — the wizard will set it up."
        )

    # Make sure the engine sees the channel env (CLI path skips the wizard's
    # _apply_env_to_os; if Gemini key sits only in the .env, the engine still
    # needs it in os.environ).
    _apply_env_to_os(_read_env(env_file))

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
