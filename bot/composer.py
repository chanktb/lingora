"""Assemble a HyperFrames project from generated content + TTS manifest."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from generator import (
    GeneratedContent, QuizContent,
    WhatsThisContent, WhatsThisItem,
    WhatsBoardContent, WhatsBoardItem,
    DialogueContent, DialogueTurn, DialogueCharacter,
    FillBlankContent,
    VocabTableContent, VocabTableItem,
    CompareContent, ComparePair,
    GuessWordContent, GuessWord,
)
from tts import AudioClip

TEMPLATE_DIR = Path(__file__).parent / "template"


def _demo_mode() -> bool:
    """When DEMO_MODE=1 env is set, templates skip channel branding
    (brand-row + tagline + outro card). Used for rendering clean demo
    videos for showcase / landing pages."""
    return os.environ.get("DEMO_MODE", "0").strip() in ("1", "true", "yes", "on")
EMOJI_BY_TOPIC = {
    "tình yêu": "💕",
    "công việc": "💼",
    "du lịch": "✈️",
    "ăn uống": "🍽️",
    "gia đình": "👨‍👩‍👧",
    "sức khỏe": "💪",
    "học tập": "📚",
    "thời tiết": "🌤️",
    "tiền bạc": "💰",
    "tạo động lực": "🔥",
    "cà phê": "☕",
    "âm nhạc": "🎵",
    "thể thao": "⚽",
    "mua sắm": "🛍️",
    "giao tiếp": "💬",
}

DEFAULT_CHANNEL_NAME = "@hocngoaingumoingay"
DEFAULT_CHANNEL_TAGLINE = "1 phút mỗi ngày · giỏi mỗi tháng"
DEFAULT_AVATAR_EMOJI = "🌍"

# Default theme = pink/dark (current Tiếng Đức look).
# Override via THEME_* env vars per channel.
DEFAULT_THEME = {
    "primary":       "#ff4d8d",
    "primary_light": "#ffcce0",
    "primary_soft":  "#ff8aa8",
    "accent":        "#ffa45c",
    "accent_gold":   "#ffd700",
    "ipa_color":     "rgba(140, 220, 255, 0.85)",
    "bg_grad":       "linear-gradient(135deg, #1a0f2e 0%, #0a0e27 50%, #0c0820 100%)",
    "halo_1":        "rgba(255, 100, 180, 0.45)",
    "halo_2":        "rgba(100, 150, 255, 0.40)",
    "halo_3":        "rgba(180, 100, 255, 0.30)",
    "primary_shadow_rgb": "255, 77, 141",   # for rgba(...,0.45)
}


def _theme_from_env() -> dict:
    """Build theme dict from THEME_* env vars, falling back to DEFAULT_THEME."""
    return {
        key: os.environ.get(f"THEME_{key.upper()}", default)
        for key, default in DEFAULT_THEME.items()
    }

# Short sticky-header copy by native language. {topic} = topic stripped of prefix
# (e.g. "về tình yêu" → "tình yêu"). {lang} = target language name in native lang.
SHORT_TITLE_FORMAT = {
    "vi": "Câu nói về {topic}",
    "en": "Phrases about {topic}",
}
TRANSCRIPT_LABEL_FORMAT = {
    "vi": "bản ghi bằng tiếng {lang}",
    "en": "transcript in {lang}",
}
_TOPIC_PREFIXES = ("về ", "chủ đề ", "about ", "topic ")


# Flag emoji per native language. Used by visual layouts (e.g. whats_this)
# to mark the native-language answer next to the target word. Falls back to
# 🏳️ for unknown codes.
NATIVE_LANG_FLAG = {
    "vi": "🇻🇳",
    "en": "🇬🇧",  # neutral English flag; could also be 🇺🇸
    "ko": "🇰🇷",
    "ja": "🇯🇵",
    "zh": "🇨🇳",
    "de": "🇩🇪",
    "fr": "🇫🇷",
    "es": "🇪🇸",
    "it": "🇮🇹",
    "pt": "🇵🇹",
    "ru": "🇷🇺",
    "th": "🇹🇭",
    "tr": "🇹🇷",
    "pl": "🇵🇱",
    "nl": "🇳🇱",
}


def _native_flag(native_lang: str) -> str:
    return NATIVE_LANG_FLAG.get((native_lang or "").lower(), "🏳️")


def _strip_topic_prefix(topic_label: str) -> str:
    t = (topic_label or "").strip()
    for p in _TOPIC_PREFIXES:
        if t.lower().startswith(p):
            return t[len(p):].strip()
    return t


@dataclass
class CardTiming:
    idx: int
    target: str
    pronunciation: str
    ipa: str
    native: str
    start: float
    slot: float
    # v6 timing: each phrase = chime + target_audio + vi_audio
    chime_start: float           # when chime plays (just before target audio)
    target_audio_start: float
    target_audio_duration: float
    vi_audio_start: float
    vi_audio_duration: float


def _split_intro_display(text: str) -> tuple[str, str]:
    """Best-effort line break for the title.

    Tries to balance the two lines around the middle.
    """
    words = text.split()
    if len(words) <= 3:
        return text, ""
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def _pick_emoji(topic_label: str) -> str:
    label = topic_label.lower().lstrip()
    for key, emoji in EMOJI_BY_TOPIC.items():
        if key in label:
            return emoji
    return "✨"


def _slot(audio_duration: float, min_slot: float, buffer: float = 0.7) -> float:
    return round(max(min_slot, audio_duration + buffer), 2)


# Approximate chime duration on disk (matches static/chime.mp3 from ffmpeg)
CHIME_DURATION = 0.45
# CEO bug 2026-06-13 (phrases video russianpath drift mid-video):
# GAP=0.10 too tight — gave only 100ms between target voice end → vi voice start
# AND between vi voice end → next chime start. With Chromium audio context
# 50-100ms warmup per <audio> element, late phrases overlapped.
# Bumped to 0.30 = 300ms buffer per junction = 6 audio elements per phrase
# × 10 phrases gives comfortable headroom even with cumulative drift.
GAP = 0.30                # was 0.10 — voice lag scene at mid-video


def _ffmpeg_bin() -> str:
    base = os.environ.get("FFMPEG_BIN", "")
    if base:
        candidate = Path(base) / "ffmpeg.exe"
        if candidate.exists():
            return str(candidate)
    return "ffmpeg"


def _premix_audio_track(
    *,
    out_path: Path,
    total_duration: float,
    clips: list[tuple[Path, float, float]],
) -> bool:
    """Pre-mix N audio clips into ONE master WAV at exact data-start positions.

    CEO bug report 2026-06-16 v5 (FINAL ROOT CAUSE):
    After all silence-strip + WAV PCM + GAP-bump iterations, CEO reported
    phrases voice still drifts ~1 full phrase behind scene by item 8
    (~670 ms per phrase × 7 phrases = ~4.7 s cumulative). Magnitude rules
    out encoder padding (~50 ms). The only remaining culprit is
    HyperFrames' per-`<audio>`-element scheduling overhead — every clip
    in the timeline adds ~200 ms of mixer latency that accumulates over
    the lesson.

    Permanent fix: bypass HF's per-clip scheduler entirely. Mix all
    audio (chime + intro voice + per-phrase chime/target/vi + outro)
    into ONE long master WAV using ffmpeg `adelay` + `amix`, and have
    the template schedule a single `<audio data-start="0">`. HF sees
    exactly 1 audio clip → zero per-clip overhead → zero drift.

    Args:
        out_path: where to write the master WAV (e.g. .../master.wav).
        total_duration: timeline length in seconds (for `apad` tail).
        clips: list of (file_path, start_seconds, volume) tuples.
               volume = 0.0..1.0; chime usually 0.95, voice 1.0.

    Returns True on success.
    """
    if not clips:
        return False
    inputs: list[str] = []
    filter_parts: list[str] = []
    for i, (path, start_s, volume) in enumerate(clips):
        inputs.extend(["-i", str(path)])
        delay_ms = max(0, int(round(start_s * 1000)))
        vol = max(0.0, min(1.0, float(volume)))
        # Resample to 44.1k mono FIRST so amix receives uniform streams
        # (Edge-TTS WAVs are 44.1k mono already; chime.mp3 is stereo 48k).
        # Then apply per-clip volume + adelay to position on the timeline.
        filter_parts.append(
            f"[{i}:a]aresample=44100,aformat=channel_layouts=mono,"
            f"volume={vol:.2f},adelay={delay_ms}|{delay_ms}[a{i}]"
        )
    mix_inputs = "".join(f"[a{i}]" for i in range(len(clips)))
    pad_ms = int(round(total_duration * 1000))
    filter_complex = (
        ";".join(filter_parts)
        + f";{mix_inputs}amix=inputs={len(clips)}:duration=longest:normalize=0"
        + f",apad=whole_dur={pad_ms}ms[mix]"
    )
    cmd = [
        _ffmpeg_bin(), "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[mix]",
        "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return True
        sys.stderr.write(
            f"[composer] premix failed (rc={result.returncode}): "
            f"{(result.stderr or b'')[-300:].decode('utf-8', 'replace')}\n"
        )
        return False
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[composer] premix error: {exc}\n")
        return False


def build_project(
    *,
    content: GeneratedContent,
    audio_clips: list[AudioClip],
    audio_src_dir: Path,
    out_dir: Path,
    native_lang: str = "vi",
    target_lang_name: str = "",
    hyperframes_version: str = "0.6.52",
    channel_dir: Path | None = None,
    image_src_dir: Path | None = None,
    post_number: int = 0,
) -> Path:
    """Materialize a HyperFrames project at out_dir, ready for `npm run render`.

    channel_dir: optional path to channels/<name>/. If given, its static/ overlays
    template/static/ (so per-channel logo.png replaces the default).

    Returns the path to out_dir.
    """
    # v6 expects: intro_native + (target_i + vi_i)*N + outro_native
    expected = 2 + 2 * len(content.phrases)
    if len(audio_clips) != expected:
        raise ValueError(f"Expected {expected} audio clips for v6, got {len(audio_clips)}")

    clip_by_name = {c.name: c for c in audio_clips}
    intro_clip = clip_by_name["intro_vi"]
    outro_clip = clip_by_name["outro_vi"]

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_audio = out_dir / "assets" / "audio"
    assets_audio.mkdir(parents=True, exist_ok=True)
    for clip in audio_clips:
        shutil.copy2(audio_src_dir / clip.file, assets_audio / clip.file)

    # Copy static (logo + chime.mp3).
    # Layered: template/static (default) overlaid by channels/<name>/static (per-channel).
    static_dst = out_dir / "static"
    static_dst.mkdir(parents=True, exist_ok=True)
    has_logo = False
    has_chime = False
    has_scene = False
    ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".mp3", ".wav"}
    layers: list[Path] = [TEMPLATE_DIR / "static"]
    if channel_dir is not None:
        layers.append(channel_dir / "static")
    for layer in layers:
        if not layer.exists():
            continue
        for f in layer.iterdir():
            if f.is_file() and f.suffix.lower() in ASSET_EXTS:
                shutil.copy2(f, static_dst / f.name)
                if f.name.lower() == "logo.png":
                    has_logo = True
                if f.name.lower() == "chime.mp3":
                    has_chime = True

    # AI-generated background scene (best-effort — fallback gradient if missing)
    if image_src_dir is not None:
        scene_src = image_src_dir / "scene.png"
        if scene_src.exists():
            shutil.copy2(scene_src, static_dst / "scene.png")
            has_scene = True

    # ────────────────── TIMING ──────────────────
    # Intro section: chime → intro_vi voice
    intro_chime_start = 0.05
    intro_voice_start = intro_chime_start + CHIME_DURATION + 0.05
    intro_voice_dur = intro_clip.duration
    intro_slot = round(intro_voice_start + intro_voice_dur + 0.5, 2)
    intro_top, intro_bottom = _split_intro_display(content.intro_display)
    intro = {
        "slot": intro_slot,
        "display_top": intro_top,
        "display_bottom": intro_bottom,
        "chime_start": round(intro_chime_start, 2),
        "voice_start": round(intro_voice_start, 2),
        "voice_duration": round(intro_voice_dur + 0.1, 2),
    }

    # Phrase sections: each = chime → target_audio → vi_audio
    cards: list[CardTiming] = []
    cursor = intro_slot
    for i, phrase in enumerate(content.phrases):
        idx = i + 1
        target_clip = clip_by_name[f"p{idx}_target"]
        vi_clip = clip_by_name[f"p{idx}_vi"]

        chime_start = round(cursor + 0.1, 2)
        target_start = round(chime_start + CHIME_DURATION + 0.05, 2)
        target_dur = target_clip.duration
        vi_start = round(target_start + target_dur + GAP, 2)
        vi_dur = vi_clip.duration
        phrase_end = vi_start + vi_dur + GAP
        slot = round(phrase_end - cursor, 2)

        cards.append(
            CardTiming(
                idx=idx,
                target=phrase.target,
                pronunciation=phrase.pronunciation,
                ipa=getattr(phrase, "ipa", "") or "",
                native=phrase.native,
                start=round(cursor, 2),
                slot=slot,
                chime_start=chime_start,
                target_audio_start=target_start,
                target_audio_duration=round(target_dur + 0.1, 2),
                vi_audio_start=vi_start,
                vi_audio_duration=round(vi_dur + 0.1, 2),
            )
        )
        cursor += slot

    phrases_total_slot = round(cursor - intro_slot, 2)

    # Outro section: outro_vi voice (no chime — feels like a natural fade)
    outro_voice_start = round(cursor + 0.3, 2)
    outro_voice_dur = outro_clip.duration
    outro_slot = round(outro_voice_dur + 1.2, 2)
    outro = {
        "start": round(cursor, 2),
        "slot": outro_slot,
        "voice_start": outro_voice_start,
        "voice_duration": round(outro_voice_dur + 0.1, 2),
    }
    cursor += outro_slot

    total_duration = round(cursor, 2)

    # ─── Pre-mix all audio into ONE master.wav (v5 cumulative drift fix) ───
    # See _premix_audio_track for rationale. When premix succeeds we set
    # use_master_audio=True; template emits a single <audio> tag instead of
    # per-clip elements, eliminating HF per-clip scheduler overhead.
    use_master_audio = False
    if has_chime:
        chime_path = static_dst / "chime.mp3"
        premix_clips: list[tuple[Path, float, float]] = []
        premix_clips.append((chime_path, intro["chime_start"], 0.95))
        premix_clips.append((assets_audio / intro_clip.file, intro["voice_start"], 1.0))
        for card in cards:
            premix_clips.append((chime_path, card.chime_start, 0.95))
            premix_clips.append((assets_audio / f"p{card.idx}_target.wav", card.target_audio_start, 1.0))
            premix_clips.append((assets_audio / f"p{card.idx}_vi.wav", card.vi_audio_start, 1.0))
        premix_clips.append((assets_audio / outro_clip.file, outro["voice_start"], 1.0))
        master_path = assets_audio / "master.wav"
        if _premix_audio_track(
            out_path=master_path,
            total_duration=total_duration,
            clips=premix_clips,
        ):
            use_master_audio = True

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("index.html.j2")
    html = template.render(
        demo_mode=_demo_mode(),
        total_duration=total_duration,
        phrases_total_slot=phrases_total_slot,
        intro=intro,
        outro=outro,
        use_master_audio=use_master_audio,
        outro_text=content.outro_native,
        intro_translation=content.intro_translation,
        short_title=content.short_title,
        short_title_target=content.short_title_target,
        phrases=[card.__dict__ for card in cards],
        topic_label=content.topic_label,
        emoji=_pick_emoji(content.topic_label),
        target_lang_name=target_lang_name or "Đức",
        post_number=post_number,
        native_lang=native_lang,
        channel_name=os.environ.get("CHANNEL_NAME", DEFAULT_CHANNEL_NAME),
        channel_tagline=os.environ.get("CHANNEL_TAGLINE", DEFAULT_CHANNEL_TAGLINE),
        avatar_emoji=os.environ.get("AVATAR_EMOJI", DEFAULT_AVATAR_EMOJI),
        theme=_theme_from_env(),
        has_logo=has_logo,
        has_chime=has_chime,
        has_scene=has_scene,
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    # Project metadata
    (out_dir / "meta.json").write_text(
        json.dumps({"id": out_dir.name, "name": out_dir.name}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "hyperframes.json").write_text(
        json.dumps(
            {
                "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
                "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
                "paths": {
                    "blocks": "compositions",
                    "components": "compositions/components",
                    "assets": "assets",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "package.json").write_text(
        json.dumps(
            {
                "name": out_dir.name,
                "private": True,
                "type": "module",
                "scripts": {
                    "render": f"npx --yes hyperframes@{hyperframes_version} render",
                    "lint": f"npx --yes hyperframes@{hyperframes_version} lint",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Persist a manifest for debugging / re-render
    manifest = {
        "total_duration": total_duration,
        "intro": {
            "display": content.intro_display,
            "tts": content.intro_tts,
            **intro,
        },
        "phrases": [c.__dict__ for c in cards],
        "caption": content.caption,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir


# ─────────────────────────────────────────────────────────────────────────
# QUIZ project builder — 4-option multi-choice video
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class OptionTiming:
    idx: int          # 1..4
    label: str        # "A" | "B" | "C" | "D"
    text: str
    pronunciation: str
    ipa: str
    start: float
    slot: float
    chime_start: float
    # VN-letter voice (e.g. "Đê" for D) — spoken first
    label_audio_start: float
    label_audio_duration: float
    # Target-lang voice (e.g. "Ich liebe dich") — spoken right after label
    target_audio_start: float
    target_audio_duration: float
    # Highlight window covers BOTH voices
    highlight_start: float
    highlight_end: float


def build_quiz_project(
    *,
    content: QuizContent,
    audio_clips: list[AudioClip],
    audio_src_dir: Path,
    out_dir: Path,
    target_lang_name: str = "",
    hyperframes_version: str = "0.6.52",
    channel_dir: Path | None = None,
    direction: str = "forward",
    image_src_dir: Path | None = None,
    native_lang: str = "vi",
) -> Path:
    """Materialize a quiz HyperFrames project.

    direction:
      "forward" (default) — VN question, 4 target-lang options. 10 audio clips:
          intro_vi + (opt_X_label + opt_X) × 4 + outro_vi
      "reverse"           — target phrase shown, 4 VN options. 11 audio clips:
          intro_vi + intro_target + (opt_X_label + opt_X) × 4 + outro_vi
          intro_target = target voice reading question_target ONCE between intro and options
    """
    is_reverse = (direction == "reverse")
    expected = 11 if is_reverse else 10
    if len(audio_clips) != expected:
        raise ValueError(
            f"Quiz (direction={direction}) expects {expected} audio clips, got {len(audio_clips)}"
        )

    clip_by_name = {c.name: c for c in audio_clips}
    intro_clip = clip_by_name["intro_vi"]
    intro_target_clip = clip_by_name.get("intro_target") if is_reverse else None
    outro_clip = clip_by_name["outro_vi"]

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_audio = out_dir / "assets" / "audio"
    assets_audio.mkdir(parents=True, exist_ok=True)
    for clip in audio_clips:
        shutil.copy2(audio_src_dir / clip.file, assets_audio / clip.file)

    # Layered statics (template defaults + channel overrides)
    static_dst = out_dir / "static"
    static_dst.mkdir(parents=True, exist_ok=True)
    has_logo = False
    has_chime = False
    has_scene = False
    ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".mp3", ".wav"}
    layers: list[Path] = [TEMPLATE_DIR / "static"]
    if channel_dir is not None:
        layers.append(channel_dir / "static")
    for layer in layers:
        if not layer.exists():
            continue
        for f in layer.iterdir():
            if f.is_file() and f.suffix.lower() in ASSET_EXTS:
                shutil.copy2(f, static_dst / f.name)
                if f.name.lower() == "logo.png":
                    has_logo = True
                if f.name.lower() == "chime.mp3":
                    has_chime = True

    # AI-generated background scene (best-effort — fallback to gradient if missing)
    if image_src_dir is not None:
        scene_src = image_src_dir / "scene.png"
        if scene_src.exists():
            shutil.copy2(scene_src, static_dst / "scene.png")
            has_scene = True

    # ────── TIMING ──────
    # Intro: short hook (chime + intro_vi voice)
    intro_chime_start = 0.05
    intro_voice_start = intro_chime_start + CHIME_DURATION + 0.05
    intro_voice_dur = intro_clip.duration

    # Reverse: target voice reads the question_target phrase right after the VN narrator.
    # Sequence: intro_vi → small gap → intro_target (target voice) → larger pause → options.
    intro_target = None
    if is_reverse and intro_target_clip is not None:
        target_voice_start = round(intro_voice_start + intro_voice_dur + 0.35, 2)
        target_voice_dur = intro_target_clip.duration
        intro_slot = round(target_voice_start + target_voice_dur + 0.6, 2)
        intro_target = {
            "voice_start": target_voice_start,
            "voice_duration": round(target_voice_dur + 0.1, 2),
        }
    else:
        intro_slot = round(intro_voice_start + intro_voice_dur + 0.5, 2)

    intro = {
        "slot": intro_slot,
        "question_native": content.question_native,
        "question_target": content.question_target,
        "voice_start": round(intro_voice_start, 2),
        "voice_duration": round(intro_voice_dur + 0.1, 2),
    }

    # Options: each = chime + label_VN voice + small gap + target_lang voice
    options: list[OptionTiming] = []
    cursor = intro_slot
    GAP_AFTER_LABEL = 0.15  # tiny pause between "Đê" and "Hallo"
    PAD_AFTER_TARGET = 0.45  # let viewer read before moving on
    for i, opt in enumerate(content.options, start=1):
        label_clip = clip_by_name[f"opt_{opt.label}_label"]
        target_clip = clip_by_name[f"opt_{opt.label}"]
        chime_start = round(cursor + 0.1, 2)
        label_start = round(chime_start + CHIME_DURATION + 0.05, 2)
        label_dur = label_clip.duration
        target_start = round(label_start + label_dur + GAP_AFTER_LABEL, 2)
        target_dur = target_clip.duration
        opt_end = target_start + target_dur + PAD_AFTER_TARGET
        slot = round(opt_end - cursor, 2)

        options.append(OptionTiming(
            idx=i, label=opt.label, text=opt.text,
            pronunciation=opt.pronunciation, ipa=opt.ipa,
            start=round(cursor, 2),
            slot=slot,
            chime_start=chime_start,
            label_audio_start=round(label_start, 2),
            label_audio_duration=round(label_dur + 0.1, 2),
            target_audio_start=round(target_start, 2),
            target_audio_duration=round(target_dur + 0.1, 2),
            highlight_start=round(label_start - 0.1, 2),
            highlight_end=round(target_start + target_dur + 0.2, 2),
        ))
        cursor += slot

    options_total_slot = round(cursor - intro_slot, 2)

    # Outro
    outro_voice_start = round(cursor + 0.3, 2)
    outro_voice_dur = outro_clip.duration
    outro_slot = round(outro_voice_dur + 1.4, 2)
    outro = {
        "start": round(cursor, 2),
        "slot": outro_slot,
        "voice_start": outro_voice_start,
        "voice_duration": round(outro_voice_dur + 0.1, 2),
    }
    cursor += outro_slot
    total_duration = round(cursor, 2)

    # Render
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("quiz.html.j2")
    html = template.render(
        demo_mode=_demo_mode(),
        total_duration=total_duration,
        options_total_slot=options_total_slot,
        intro=intro,
        intro_target=intro_target,
        is_reverse=is_reverse,
        outro=outro,
        outro_text=content.outro_native,
        outro_heading=(
            "YOUR ANSWER?" if native_lang == "en" else "ĐÁP ÁN CỦA BẠN?"
        ),
        question_native=content.question_native,
        question_target=content.question_target,
        options=[o.__dict__ for o in options],
        short_title=content.short_title,
        short_title_target=content.short_title_target,
        topic_label=content.topic_label,
        target_lang_name=target_lang_name or "Đức",
        native_lang=native_lang,
        emoji=_pick_emoji(content.topic_label),
        channel_name=os.environ.get("CHANNEL_NAME", DEFAULT_CHANNEL_NAME),
        channel_tagline=os.environ.get("CHANNEL_TAGLINE", DEFAULT_CHANNEL_TAGLINE),
        avatar_emoji=os.environ.get("AVATAR_EMOJI", DEFAULT_AVATAR_EMOJI),
        theme=_theme_from_env(),
        has_logo=has_logo,
        has_chime=has_chime,
        has_scene=has_scene,
        chime_duration=round(CHIME_DURATION, 2),
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    # HyperFrames project metadata
    (out_dir / "meta.json").write_text(
        json.dumps({"id": out_dir.name, "name": out_dir.name}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "hyperframes.json").write_text(
        json.dumps({
            "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
            "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
            "paths": {"blocks": "compositions", "components": "compositions/components", "assets": "assets"},
        }, indent=2),
        encoding="utf-8",
    )
    (out_dir / "package.json").write_text(
        json.dumps({
            "name": out_dir.name, "private": True, "type": "module",
            "scripts": {
                "render": f"npx --yes hyperframes@{hyperframes_version} render",
                "lint":   f"npx --yes hyperframes@{hyperframes_version} lint",
            },
        }, indent=2),
        encoding="utf-8",
    )

    # Manifest (also stores answer + explanation for FB auto-comment)
    manifest = {
        "layout_type": "quiz_reverse" if is_reverse else "quiz",
        "direction": direction,
        "total_duration": total_duration,
        "question_native": content.question_native,
        "question_target": content.question_target,
        "options": [o.__dict__ for o in options],
        "correct_answer": content.correct_answer,
        "explanation": content.explanation,
        "caption": content.caption,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir


# ═══════════════════════════════════════════════════════════════════════
#  WHATS_THIS layout — visual vocab with AI image gen
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WhatsThisItemTiming:
    """One round in the whats_this video — ~7s per item."""
    idx: int
    target_word: str
    pronunciation: str
    ipa: str
    native_answer: str
    image_filename: str          # "whats_N.png" relative to job dir
    item_kind: str
    voice_question: str          # target-lang "Was ist das?" — shown WHILE voice plays
    start: float                 # absolute scene start (s)
    slot: float                  # this item's duration
    # Sub-cue offsets WITHIN this item (relative to start):
    question_text_start: float   # when question text becomes visible
    question_audio_start: float  # absolute, when voice_question plays
    question_audio_duration: float
    image_start: float           # when image fades in (absolute)
    countdown_start: float       # when 3-2-1 countdown begins (absolute)
    reveal_start: float          # when target_word card appears (absolute)
    reveal_audio_start: float    # absolute
    reveal_audio_duration: float


def build_whats_this_project(
    *,
    content: WhatsThisContent,
    audio_clips: list[AudioClip],     # 20 (q_i + r_i) + 1 outro_vi = 21
    audio_src_dir: Path,
    image_src_dir: Path,              # contains whats_1.png .. whats_10.png
    out_dir: Path,
    target_lang_name: str = "",
    hyperframes_version: str = "0.6.52",
    channel_dir: Path | None = None,
    native_lang: str = "vi",
) -> Path:
    """Materialize a HyperFrames project for the whats_this (visual vocab) layout.

    NO INTRO — video starts directly with item 1 question to maximize retention.

    Expected audio clip names (21 total):
      - q_1, q_2, ..., q_10  (target-lang voice asking question per item)
      - r_1, r_2, ..., r_10  (target-lang voice reading the answer word)
      - outro_vi
    Expected image files in `image_src_dir`: whats_1.png .. whats_10.png
    """
    if len(audio_clips) != 21:
        raise ValueError(f"whats_this expects 21 audio clips, got {len(audio_clips)}")
    if len(content.items) != 10:
        raise ValueError(f"whats_this needs 10 items, got {len(content.items)}")

    clip_by_name = {c.name: c for c in audio_clips}
    outro_clip = clip_by_name["outro_vi"]

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_audio = out_dir / "assets" / "audio"
    assets_audio.mkdir(parents=True, exist_ok=True)
    for clip in audio_clips:
        shutil.copy2(audio_src_dir / clip.file, assets_audio / clip.file)

    # Layered statics (template + channel) + per-item images
    static_dst = out_dir / "static"
    static_dst.mkdir(parents=True, exist_ok=True)
    has_logo = False
    has_chime = False
    ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".mp3", ".wav"}
    for layer in [TEMPLATE_DIR / "static", (channel_dir / "static") if channel_dir else None]:
        if layer is None or not layer.exists():
            continue
        for f in layer.iterdir():
            if f.is_file() and f.suffix.lower() in ASSET_EXTS:
                shutil.copy2(f, static_dst / f.name)
                if f.name.lower() == "logo.png":
                    has_logo = True
                if f.name.lower() == "chime.mp3":
                    has_chime = True

    # Copy 10 generated images
    for i in range(1, 11):
        src = image_src_dir / f"whats_{i}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing AI image: {src}")
        shutil.copy2(src, static_dst / f"whats_{i}.png")

    # ────── TIMING ──────
    # NO INTRO — start with item 1 immediately for max retention.
    intro_slot = 0.0

    # Per item: 7.0s base, but auto-extend if voice clips exceed budget.
    items_timing: list[WhatsThisItemTiming] = []
    cursor = 0.0

    # CEO bug report 2026-06-13: late items (8-10) voice drifts behind countdown.
    # Defense-in-depth fix:
    #   1. tts._probe_duration adds +0.08 s drift comp (global, see tts.py)
    #   2. SUB_IMAGE_HOLD bumped 1.8 → 2.0 so question voice has more headroom
    #      before countdown starts (each item has its own audio file → drift is
    #      per-item, not cumulative on countdown, but margin reduces visible lag).
    #   3. SUB_REVEAL_TAIL bumped 1.0 → 1.3 so reveal voice never bleeds into
    #      next item's question start.
    #
    # Sub-cue spacing within each item (seconds):
    # v4: image now appears SIMULTANEOUSLY with question (no wait for voice to finish).
    SUB_IMAGE_HOLD = 2.0        # was 1.8 — more headroom after question voice
    SUB_COUNTDOWN = 1.5         # 3-2-1 countdown duration (3 beeps × 0.5s)
    SUB_REVEAL_PAD = 0.25       # pad before reveal voice plays
    SUB_REVEAL_TAIL = 1.3       # was 1.0 — protect against drift into next item

    for i, it in enumerate(content.items, start=1):
        q_clip = clip_by_name[f"q_{i}"]
        r_clip = clip_by_name[f"r_{i}"]

        # Question text + question voice + image ALL appear together at the start of the round.
        # Image fades in 0.05s earlier so its animation finishes by the time voice starts.
        image_start = round(cursor + 0.05, 2)
        q_audio_start = round(cursor + 0.15, 2)
        q_audio_dur = q_clip.duration
        # Countdown starts AFTER question voice ends + image-hold pad (viewer can study image).
        countdown_start = round(q_audio_start + q_audio_dur + SUB_IMAGE_HOLD, 2)
        reveal_start = round(countdown_start + SUB_COUNTDOWN, 2)
        reveal_audio_start = round(reveal_start + SUB_REVEAL_PAD, 2)
        reveal_audio_dur = r_clip.duration
        item_end = reveal_audio_start + reveal_audio_dur + SUB_REVEAL_TAIL
        slot = round(item_end - cursor, 2)

        items_timing.append(WhatsThisItemTiming(
            idx=i,
            target_word=it.target_word,
            pronunciation=it.pronunciation,
            ipa=it.ipa,
            native_answer=it.native_answer,
            image_filename=f"whats_{i}.png",
            item_kind=it.item_kind,
            voice_question=it.voice_question,
            start=round(cursor, 2),
            slot=slot,
            question_text_start=round(cursor + 0.05, 2),
            question_audio_start=q_audio_start,
            question_audio_duration=round(q_audio_dur + 0.1, 2),
            image_start=image_start,
            countdown_start=countdown_start,
            reveal_start=reveal_start,
            reveal_audio_start=reveal_audio_start,
            reveal_audio_duration=round(reveal_audio_dur + 0.1, 2),
        ))
        cursor += slot

    items_total_slot = round(cursor, 2)

    # Outro
    outro_voice_start = round(cursor + 0.25, 2)
    outro_voice_dur = outro_clip.duration
    outro_slot = round(outro_voice_dur + 1.4, 2)
    outro = {
        "start": round(cursor, 2),
        "slot": outro_slot,
        "voice_start": outro_voice_start,
        "voice_duration": round(outro_voice_dur + 0.1, 2),
    }
    cursor += outro_slot
    total_duration = round(cursor, 2)

    # v5 cumulative-drift fix: pre-mix all audio (q×10 + r×10 + outro +
    # countdown chimes×30) into ONE master.wav. HF schedules 1 audio
    # element → zero per-clip mixer overhead → zero drift.
    use_master_audio = False
    if has_chime:
        chime_path = static_dst / "chime.mp3"
        premix_clips: list[tuple[Path, float, float]] = []
        for t in items_timing:
            premix_clips.append((assets_audio / f"q_{t.idx}.wav", t.question_audio_start, 1.0))
            # 3 countdown chimes per item at countdown_start + 0/0.5/1.0
            premix_clips.append((chime_path, t.countdown_start, 0.85))
            premix_clips.append((chime_path, round(t.countdown_start + 0.5, 2), 0.85))
            premix_clips.append((chime_path, round(t.countdown_start + 1.0, 2), 0.85))
            premix_clips.append((assets_audio / f"r_{t.idx}.wav", t.reveal_audio_start, 1.0))
        premix_clips.append((assets_audio / outro_clip.file, outro["voice_start"], 1.0))
        master_path = assets_audio / "master.wav"
        if _premix_audio_track(
            out_path=master_path,
            total_duration=total_duration,
            clips=premix_clips,
        ):
            use_master_audio = True

    # Render template
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("whats_this.html.j2")
    html = template.render(
        demo_mode=_demo_mode(),
        total_duration=total_duration,
        items=[t.__dict__ for t in items_timing],
        items_total_slot=items_total_slot,
        outro=outro,
        outro_text=content.outro_native,
        short_title=content.short_title,
        short_title_target=content.short_title_target,
        topic_label=content.topic_label,
        display_question=content.display_question,
        target_lang_name=target_lang_name or "Đức",
        native_lang=native_lang,
        native_flag=_native_flag(native_lang),
        emoji=_pick_emoji(content.topic_label),
        channel_name=os.environ.get("CHANNEL_NAME", DEFAULT_CHANNEL_NAME),
        channel_tagline=os.environ.get("CHANNEL_TAGLINE", DEFAULT_CHANNEL_TAGLINE),
        avatar_emoji=os.environ.get("AVATAR_EMOJI", DEFAULT_AVATAR_EMOJI),
        theme=_theme_from_env(),
        has_logo=has_logo,
        has_chime=has_chime,
        chime_duration=round(CHIME_DURATION, 2),
        use_master_audio=use_master_audio,
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    # HF metadata
    (out_dir / "meta.json").write_text(
        json.dumps({"id": out_dir.name, "name": out_dir.name}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "hyperframes.json").write_text(
        json.dumps({
            "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
            "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
            "paths": {"blocks": "compositions", "components": "compositions/components", "assets": "assets"},
        }, indent=2),
        encoding="utf-8",
    )
    (out_dir / "package.json").write_text(
        json.dumps({
            "name": out_dir.name, "private": True, "type": "module",
            "scripts": {
                "render": f"npx --yes hyperframes@{hyperframes_version} render",
                "lint":   f"npx --yes hyperframes@{hyperframes_version} lint",
            },
        }, indent=2),
        encoding="utf-8",
    )

    # Manifest
    manifest = {
        "layout_type": "whats_this",
        "total_duration": total_duration,
        "items": [
            {
                "idx": t.idx,
                "target_word": t.target_word,
                "pronunciation": t.pronunciation,
                "ipa": t.ipa,
                "native_answer": t.native_answer,
                "item_kind": t.item_kind,
            }
            for t in items_timing
        ],
        "caption": content.caption,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir


# ═══════════════════════════════════════════════════════════════════════
#  WHATS_BOARD layout — 9-grid cheat sheet
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WhatsBoardCellTiming:
    """One cell in the 9-grid. All cells visible throughout; .start/.slot drive highlight."""
    idx: int
    target_word: str
    pronunciation: str
    ipa: str
    native_answer: str
    image_filename: str
    start: float
    slot: float
    voice_start: float
    voice_duration: float


def build_whats_board_project(
    *,
    content: WhatsBoardContent,
    audio_clips: list[AudioClip],     # 9 voice_repeat (v_1..v_9) + 1 outro_vi = 10
    audio_src_dir: Path,
    image_src_dir: Path,              # contains board_1.png .. board_9.png
    out_dir: Path,
    target_lang_name: str = "",
    hyperframes_version: str = "0.6.52",
    channel_dir: Path | None = None,
) -> Path:
    """Materialize a HyperFrames project for the whats_board (9-grid) layout.

    All 9 cells visible from t=0. Active cell highlight rotates through 1..9,
    each with the target voice reading the word twice. NO intro card.
    """
    if len(audio_clips) != 10:
        raise ValueError(f"whats_board expects 10 audio clips, got {len(audio_clips)}")
    if len(content.items) != 9:
        raise ValueError(f"whats_board needs 9 items, got {len(content.items)}")

    clip_by_name = {c.name: c for c in audio_clips}
    outro_clip = clip_by_name["outro_vi"]

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_audio = out_dir / "assets" / "audio"
    assets_audio.mkdir(parents=True, exist_ok=True)
    for clip in audio_clips:
        shutil.copy2(audio_src_dir / clip.file, assets_audio / clip.file)

    static_dst = out_dir / "static"
    static_dst.mkdir(parents=True, exist_ok=True)
    has_logo = False
    has_chime = False
    ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".mp3", ".wav"}
    for layer in [TEMPLATE_DIR / "static", (channel_dir / "static") if channel_dir else None]:
        if layer is None or not layer.exists():
            continue
        for f in layer.iterdir():
            if f.is_file() and f.suffix.lower() in ASSET_EXTS:
                shutil.copy2(f, static_dst / f.name)
                if f.name.lower() == "logo.png":
                    has_logo = True
                if f.name.lower() == "chime.mp3":
                    has_chime = True

    for i in range(1, 10):
        src = image_src_dir / f"board_{i}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing AI image: {src}")
        shutil.copy2(src, static_dst / f"board_{i}.png")

    # ────── TIMING ──────
    # CEO bug report 2026-06-13: late cells (cell 7+) voice drifts behind
    # scene transition. Defense-in-depth fix:
    #   1. tts._probe_duration adds +0.08 s drift comp (global, see tts.py)
    #   2. PAD_AFTER_VOICE bumped 0.35 → 0.55 here (per-layout safety)
    #   3. VOICE_LEAD_IN 0.05 → 0.0 — let voice start exactly at cell switch
    #      so the leading edge of the audio aligns with the visible highlight.
    INTRO_HOLD = 0.8           # brief hold so viewer scans full board
    PAD_AFTER_VOICE = 0.55     # was 0.35 — drifted on cell 7-9
    VOICE_LEAD_IN = 0.0        # was 0.05 — eliminate one source of cumulative gap

    cells_timing: list[WhatsBoardCellTiming] = []
    cursor = INTRO_HOLD

    for i, it in enumerate(content.items, start=1):
        v_clip = clip_by_name[f"v_{i}"]
        voice_start = round(cursor + VOICE_LEAD_IN, 2)
        voice_dur = v_clip.duration
        slot = round(voice_dur + PAD_AFTER_VOICE + 0.1, 2)
        cells_timing.append(WhatsBoardCellTiming(
            idx=i,
            target_word=it.target_word,
            pronunciation=it.pronunciation,
            ipa=it.ipa,
            native_answer=it.native_answer,
            image_filename=f"board_{i}.png",
            start=round(cursor, 2),
            slot=slot,
            voice_start=voice_start,
            voice_duration=round(voice_dur + 0.15, 2),  # was 0.1 — extra timeline room
        ))
        cursor += slot

    cells_total_end = round(cursor, 2)

    outro_voice_start = round(cursor + 0.25, 2)
    outro_voice_dur = outro_clip.duration
    outro_slot = round(outro_voice_dur + 1.4, 2)
    outro = {
        "start": round(cursor, 2),
        "slot": outro_slot,
        "voice_start": outro_voice_start,
        "voice_duration": round(outro_voice_dur + 0.1, 2),
    }
    cursor += outro_slot
    total_duration = round(cursor, 2)

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("whats_board.html.j2")
    html = template.render(
        demo_mode=_demo_mode(),
        total_duration=total_duration,
        title_native=content.title_native,
        title_target=content.title_target,
        cells=[c.__dict__ for c in cells_timing],
        cells_total_end=cells_total_end,
        outro=outro,
        outro_text=content.outro_native,
        short_title=content.short_title,
        short_title_target=content.short_title_target,
        topic_label=content.topic_label,
        target_lang_name=target_lang_name or "Đức",
        emoji=_pick_emoji(content.topic_label),
        channel_name=os.environ.get("CHANNEL_NAME", DEFAULT_CHANNEL_NAME),
        channel_tagline=os.environ.get("CHANNEL_TAGLINE", DEFAULT_CHANNEL_TAGLINE),
        avatar_emoji=os.environ.get("AVATAR_EMOJI", DEFAULT_AVATAR_EMOJI),
        theme=_theme_from_env(),
        has_logo=has_logo,
        has_chime=has_chime,
        chime_duration=round(CHIME_DURATION, 2),
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    (out_dir / "meta.json").write_text(
        json.dumps({"id": out_dir.name, "name": out_dir.name}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "hyperframes.json").write_text(
        json.dumps({
            "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
            "registry": "https://raw.githubusercontent.com/heygen-com/main/registry",
            "paths": {"blocks": "compositions", "components": "compositions/components", "assets": "assets"},
        }, indent=2),
        encoding="utf-8",
    )
    (out_dir / "package.json").write_text(
        json.dumps({
            "name": out_dir.name, "private": True, "type": "module",
            "scripts": {
                "render": f"npx --yes hyperframes@{hyperframes_version} render",
                "lint":   f"npx --yes hyperframes@{hyperframes_version} lint",
            },
        }, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "layout_type": "whats_board",
        "total_duration": total_duration,
        "title_native": content.title_native,
        "title_target": content.title_target,
        "cells": [
            {
                "idx": c.idx,
                "target_word": c.target_word,
                "pronunciation": c.pronunciation,
                "ipa": c.ipa,
                "native_answer": c.native_answer,
            }
            for c in cells_timing
        ],
        "caption": content.caption,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir


# ═══════════════════════════════════════════════════════════════════════
#  DIALOGUE layout — 2-character mini skit
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DialogueTurnTiming:
    """One turn in the dialogue timeline."""
    idx: int
    speaker: str           # "A" | "B"
    target: str
    pronunciation: str
    ipa: str
    native: str
    start: float
    slot: float
    voice_start: float
    voice_duration: float


def build_dialogue_project(
    *,
    content,                          # DialogueContent (avoid forward import)
    audio_clips: list[AudioClip],     # N turns (t_1..t_N) + outro_vi
    audio_src_dir: Path,
    image_src_dir: Path,              # contains char_a.png, char_b.png, scene.png
    out_dir: Path,
    target_lang_name: str = "",
    hyperframes_version: str = "0.6.52",
    channel_dir: Path | None = None,
) -> Path:
    """Materialize a HyperFrames project for the dialogue layout."""
    n_turns = len(content.turns)
    expected_audio = n_turns + 1  # turns + outro_vi
    if len(audio_clips) != expected_audio:
        raise ValueError(f"dialogue expects {expected_audio} audio clips, got {len(audio_clips)}")
    if not (6 <= n_turns <= 8):
        raise ValueError(f"dialogue needs 6-8 turns, got {n_turns}")

    clip_by_name = {c.name: c for c in audio_clips}
    outro_clip = clip_by_name["outro_vi"]

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_audio = out_dir / "assets" / "audio"
    assets_audio.mkdir(parents=True, exist_ok=True)
    for clip in audio_clips:
        shutil.copy2(audio_src_dir / clip.file, assets_audio / clip.file)

    static_dst = out_dir / "static"
    static_dst.mkdir(parents=True, exist_ok=True)
    has_logo = False
    has_chime = False
    ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".mp3", ".wav"}
    for layer in [TEMPLATE_DIR / "static", (channel_dir / "static") if channel_dir else None]:
        if layer is None or not layer.exists():
            continue
        for f in layer.iterdir():
            if f.is_file() and f.suffix.lower() in ASSET_EXTS:
                shutil.copy2(f, static_dst / f.name)
                if f.name.lower() == "logo.png":
                    has_logo = True
                if f.name.lower() == "chime.mp3":
                    has_chime = True

    # Copy the 3 dialogue images
    for name in ("char_a.png", "char_b.png", "scene.png"):
        src = image_src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing dialogue image: {src}")
        shutil.copy2(src, static_dst / name)

    # ────── TIMING ──────
    # CEO feedback 2026-06-12: with PAD_AFTER_TURN=0.45 + voice_start offset
    # 0.1, late turns (6-8) drifted: scene moved to next bubble while voice
    # for current turn was still finishing. Root cause = ffprobe under-
    # reports MP3 duration by ~50-80ms (encoder padding) + Edge TTS adds
    # tiny leading silence — so the 0.45s buffer effectively shrank to
    # ~0.3s, not enough to mask the discrepancy on longer late-turn voice.
    # Fix: bigger headroom (0.75s) + earlier voice start (0.05s offset).
    INTRO_HOLD = 1.0           # show title + characters briefly before first turn
    PAD_AFTER_TURN = 0.75      # was 0.45 — voice lagged scene on late turns
    VOICE_LEAD_IN  = 0.05      # was 0.10 — start voice sooner after scene flip

    turns_timing: list[DialogueTurnTiming] = []
    cursor = INTRO_HOLD
    for i, turn in enumerate(content.turns, start=1):
        t_clip = clip_by_name[f"t_{i}"]
        voice_start = round(cursor + VOICE_LEAD_IN, 2)
        voice_dur = t_clip.duration
        slot = round(voice_dur + PAD_AFTER_TURN + VOICE_LEAD_IN, 2)
        turns_timing.append(DialogueTurnTiming(
            idx=i,
            speaker=turn.speaker,
            target=turn.target,
            pronunciation=turn.pronunciation,
            ipa=turn.ipa,
            native=turn.native,
            start=round(cursor, 2),
            slot=slot,
            voice_start=voice_start,
            voice_duration=round(voice_dur + 0.15, 2),
        ))
        cursor += slot

    dialogue_total_end = round(cursor, 2)

    outro_voice_start = round(cursor + 0.25, 2)
    outro_voice_dur = outro_clip.duration
    outro_slot = round(outro_voice_dur + 1.4, 2)
    outro = {
        "start": round(cursor, 2),
        "slot": outro_slot,
        "voice_start": outro_voice_start,
        "voice_duration": round(outro_voice_dur + 0.1, 2),
    }
    cursor += outro_slot
    total_duration = round(cursor, 2)

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("dialogue.html.j2")
    html = template.render(
        demo_mode=_demo_mode(),
        total_duration=total_duration,
        title_native=content.title_native,
        title_target=content.title_target,
        char_a_name=content.char_a.name,
        char_a_role=content.char_a.role,
        char_b_name=content.char_b.name,
        char_b_role=content.char_b.role,
        turns=[t.__dict__ for t in turns_timing],
        dialogue_total_end=dialogue_total_end,
        outro=outro,
        outro_text=content.outro_native,
        short_title=content.short_title,
        short_title_target=content.short_title_target,
        topic_label=content.topic_label,
        target_lang_name=target_lang_name or "Đức",
        emoji=_pick_emoji(content.topic_label),
        channel_name=os.environ.get("CHANNEL_NAME", DEFAULT_CHANNEL_NAME),
        channel_tagline=os.environ.get("CHANNEL_TAGLINE", DEFAULT_CHANNEL_TAGLINE),
        avatar_emoji=os.environ.get("AVATAR_EMOJI", DEFAULT_AVATAR_EMOJI),
        theme=_theme_from_env(),
        has_logo=has_logo,
        has_chime=has_chime,
        chime_duration=round(CHIME_DURATION, 2),
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    (out_dir / "meta.json").write_text(
        json.dumps({"id": out_dir.name, "name": out_dir.name}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "hyperframes.json").write_text(
        json.dumps({
            "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
            "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
            "paths": {"blocks": "compositions", "components": "compositions/components", "assets": "assets"},
        }, indent=2),
        encoding="utf-8",
    )
    (out_dir / "package.json").write_text(
        json.dumps({
            "name": out_dir.name, "private": True, "type": "module",
            "scripts": {
                "render": f"npx --yes hyperframes@{hyperframes_version} render",
                "lint":   f"npx --yes hyperframes@{hyperframes_version} lint",
            },
        }, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "layout_type": "dialogue",
        "total_duration": total_duration,
        "scenario": content.scenario,
        "char_a": {"name": content.char_a.name, "role": content.char_a.role, "voice_gender": content.char_a.voice_gender},
        "char_b": {"name": content.char_b.name, "role": content.char_b.role, "voice_gender": content.char_b.voice_gender},
        "turns": [
            {"idx": t.idx, "speaker": t.speaker, "target": t.target,
             "pronunciation": t.pronunciation, "ipa": t.ipa, "native": t.native}
            for t in turns_timing
        ],
        "caption": content.caption,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir


# ═══════════════════════════════════════════════════════════════════════
#  FILL_BLANK_QUIZ layout — short photo+sentence fill-in-blank
# ═══════════════════════════════════════════════════════════════════════

def build_fill_blank_project(
    *,
    content: FillBlankContent,
    image_src_dir: Path,              # contains scene.png
    out_dir: Path,
    target_lang_name: str = "",
    hyperframes_version: str = "0.6.52",
    channel_dir: Path | None = None,
    native_lang: str = "vi",
) -> Path:
    """Materialize a HyperFrames project for fill_blank as STATIC IMAGE poster.

    NO audio clips. Renders as 1.5s "video" → ffmpeg extracts first frame as PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_audio = out_dir / "assets" / "audio"
    assets_audio.mkdir(parents=True, exist_ok=True)

    static_dst = out_dir / "static"
    static_dst.mkdir(parents=True, exist_ok=True)
    has_logo = False
    ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".mp3", ".wav"}
    for layer in [TEMPLATE_DIR / "static", (channel_dir / "static") if channel_dir else None]:
        if layer is None or not layer.exists():
            continue
        for f in layer.iterdir():
            if f.is_file() and f.suffix.lower() in ASSET_EXTS:
                shutil.copy2(f, static_dst / f.name)
                if f.name.lower() == "logo.png":
                    has_logo = True

    # Scene image (required)
    scene_src = image_src_dir / "scene.png"
    if not scene_src.exists():
        raise FileNotFoundError(f"Missing scene image: {scene_src}")
    shutil.copy2(scene_src, static_dst / "scene.png")

    # 1.5s static — only need 1 still frame
    total_duration = 1.5

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("fill_blank.html.j2")
    html = template.render(
        demo_mode=_demo_mode(),
        total_duration=total_duration,
        title_native=content.title_native,
        title_target=content.title_target,
        sentence_template=content.sentence_template,
        correct_word=content.correct_word,
        options=content.options,
        correct_index=content.correct_index,
        topic_label=content.topic_label,
        target_lang_name=target_lang_name or "Đức",
        native_lang=native_lang,
        emoji=_pick_emoji(content.topic_label),
        channel_name=os.environ.get("CHANNEL_NAME", DEFAULT_CHANNEL_NAME),
        channel_tagline=os.environ.get("CHANNEL_TAGLINE", DEFAULT_CHANNEL_TAGLINE),
        avatar_emoji=os.environ.get("AVATAR_EMOJI", DEFAULT_AVATAR_EMOJI),
        theme=_theme_from_env(),
        has_logo=has_logo,
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    (out_dir / "meta.json").write_text(
        json.dumps({"id": out_dir.name, "name": out_dir.name}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "hyperframes.json").write_text(
        json.dumps({
            "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
            "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
            "paths": {"blocks": "compositions", "components": "compositions/components", "assets": "assets"},
        }, indent=2),
        encoding="utf-8",
    )
    (out_dir / "package.json").write_text(
        json.dumps({
            "name": out_dir.name, "private": True, "type": "module",
            "scripts": {
                "render": f"npx --yes hyperframes@{hyperframes_version} render",
                "lint":   f"npx --yes hyperframes@{hyperframes_version} lint",
            },
        }, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "layout_type": "fill_blank",
        "total_duration": total_duration,
        "sentence_template": content.sentence_template,
        "sentence_filled": content.sentence_template.replace("___", content.correct_word, 1),
        "correct_word": content.correct_word,
        "options": content.options,
        "correct_index": content.correct_index,
        "native_translation": content.native_translation,
        "explanation": content.explanation,  # Telegram admin only
        "caption": content.caption,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir


# ═══════════════════════════════════════════════════════════════════════
#  VOCAB_TABLE_IMAGE layout — static PNG poster (not video!)
# ═══════════════════════════════════════════════════════════════════════

def build_vocab_table_project(
    *,
    content: VocabTableContent,
    image_src_dir: Path,              # contains icon.png
    out_dir: Path,
    target_lang_name: str = "",
    hyperframes_version: str = "0.6.52",
    channel_dir: Path | None = None,
    native_lang: str = "vi",
) -> Path:
    """Materialize a HyperFrames project for the vocab_table_image layout.

    NO audio clips. Renders as 1.5s "video" → ffmpeg extracts first frame as PNG.
    v4: full PHOTOREALISTIC SCENE BACKGROUND (like fill_blank), title overlay,
    table overlay with dark backdrop for readability, brand row + tagline below.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create empty assets dir to keep HF happy
    assets_audio = out_dir / "assets" / "audio"
    assets_audio.mkdir(parents=True, exist_ok=True)

    static_dst = out_dir / "static"
    static_dst.mkdir(parents=True, exist_ok=True)
    has_logo = False
    ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".mp3", ".wav"}
    for layer in [TEMPLATE_DIR / "static", (channel_dir / "static") if channel_dir else None]:
        if layer is None or not layer.exists():
            continue
        for f in layer.iterdir():
            if f.is_file() and f.suffix.lower() in ASSET_EXTS:
                shutil.copy2(f, static_dst / f.name)
                if f.name.lower() == "logo.png":
                    has_logo = True

    # Full photorealistic scene background (required, like fill_blank)
    scene_src = image_src_dir / "scene.png"
    if not scene_src.exists():
        raise FileNotFoundError(f"Missing scene image: {scene_src}")
    shutil.copy2(scene_src, static_dst / "scene.png")

    # Total duration: 1.5s — we only need 1 still frame from this
    total_duration = 1.5

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("vocab_table.html.j2")
    html = template.render(
        demo_mode=_demo_mode(),
        total_duration=total_duration,
        title_native=content.title_native,
        title_target=content.title_target,
        items=[it.__dict__ for it in content.items],
        topic_label=content.topic_label,
        target_lang_name=target_lang_name or "Đức",
        native_lang=native_lang,
        emoji=_pick_emoji(content.topic_label),
        channel_name=os.environ.get("CHANNEL_NAME", DEFAULT_CHANNEL_NAME),
        channel_tagline=os.environ.get("CHANNEL_TAGLINE", DEFAULT_CHANNEL_TAGLINE),
        avatar_emoji=os.environ.get("AVATAR_EMOJI", DEFAULT_AVATAR_EMOJI),
        theme=_theme_from_env(),
        has_logo=has_logo,
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    (out_dir / "meta.json").write_text(
        json.dumps({"id": out_dir.name, "name": out_dir.name}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "hyperframes.json").write_text(
        json.dumps({
            "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
            "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
            "paths": {"blocks": "compositions", "components": "compositions/components", "assets": "assets"},
        }, indent=2),
        encoding="utf-8",
    )
    (out_dir / "package.json").write_text(
        json.dumps({
            "name": out_dir.name, "private": True, "type": "module",
            "scripts": {
                "render": f"npx --yes hyperframes@{hyperframes_version} render",
                "lint":   f"npx --yes hyperframes@{hyperframes_version} lint",
            },
        }, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "layout_type": "vocab_table",
        "total_duration": total_duration,
        "title_native": content.title_native,
        "title_target": content.title_target,
        "items": [
            {"target_word": it.target_word, "pronunciation": it.pronunciation,
             "native_answer": it.native_answer}
            for it in content.items
        ],
        "caption": content.caption,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir


# ═══════════════════════════════════════════════════════════════════════
#  COMPARE layout — 2-column comparison PNG poster (basic vs fluent, etc)
# ═══════════════════════════════════════════════════════════════════════

def build_compare_project(
    *,
    content: CompareContent,
    out_dir: Path,
    target_lang_name: str = "",
    hyperframes_version: str = "0.6.52",
    channel_dir: Path | None = None,
    native_lang: str = "vi",
) -> Path:
    """Materialize a HyperFrames project for compare 2-column static PNG poster.

    No audio, no AI image gen — pure HTML+CSS table with emoji icons,
    pronunciation guides, and two columns of phrases. Renders as 1.5s "video"
    so ffmpeg can extract the first frame → upload to FB /photos.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Empty assets dir to keep HF happy
    assets_audio = out_dir / "assets" / "audio"
    assets_audio.mkdir(parents=True, exist_ok=True)

    static_dst = out_dir / "static"
    static_dst.mkdir(parents=True, exist_ok=True)
    has_logo = False
    ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".mp3", ".wav"}
    for layer in [TEMPLATE_DIR / "static", (channel_dir / "static") if channel_dir else None]:
        if layer is None or not layer.exists():
            continue
        for f in layer.iterdir():
            if f.is_file() and f.suffix.lower() in ASSET_EXTS:
                shutil.copy2(f, static_dst / f.name)
                if f.name.lower() == "logo.png":
                    has_logo = True

    # 1.5s static — only need 1 still frame
    total_duration = 1.5

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("compare.html.j2")
    html = template.render(
        demo_mode=_demo_mode(),
        total_duration=total_duration,
        title_native=content.title_native,
        title_target=content.title_target,
        mode=content.mode,
        left_header=content.left_header,
        right_header=content.right_header,
        pairs=[p.__dict__ for p in content.pairs],
        topic_label=content.topic_label,
        target_lang_name=target_lang_name or "Đức",
        native_lang=native_lang,
        emoji=_pick_emoji(content.topic_label),
        channel_name=os.environ.get("CHANNEL_NAME", DEFAULT_CHANNEL_NAME),
        channel_tagline=os.environ.get("CHANNEL_TAGLINE", DEFAULT_CHANNEL_TAGLINE),
        avatar_emoji=os.environ.get("AVATAR_EMOJI", DEFAULT_AVATAR_EMOJI),
        theme=_theme_from_env(),
        has_logo=has_logo,
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    (out_dir / "meta.json").write_text(
        json.dumps({"id": out_dir.name, "name": out_dir.name}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "hyperframes.json").write_text(
        json.dumps({
            "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
            "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
            "paths": {"blocks": "compositions", "components": "compositions/components", "assets": "assets"},
        }, indent=2),
        encoding="utf-8",
    )
    (out_dir / "package.json").write_text(
        json.dumps({
            "name": out_dir.name, "private": True, "type": "module",
            "scripts": {
                "render": f"npx --yes hyperframes@{hyperframes_version} render",
                "lint":   f"npx --yes hyperframes@{hyperframes_version} lint",
            },
        }, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "layout_type": "compare",
        "total_duration": total_duration,
        "title_native": content.title_native,
        "title_target": content.title_target,
        "mode": content.mode,
        "left_header": content.left_header,
        "right_header": content.right_header,
        "pairs": [
            {"emoji": p.emoji,
             "left_target": p.left_target, "right_target": p.right_target,
             "left_pron": p.left_pron, "right_pron": p.right_pron,
             "left_native": p.left_native, "right_native": p.right_native}
            for p in content.pairs
        ],
        "caption": content.caption,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir


# ═══════════════════════════════════════════════════════════════════════
#  GUESS_WORD layout — 10-word reveal game with countdown (added 2026-06-15)
# ═══════════════════════════════════════════════════════════════════════

# Part-of-speech short label per Gemini's "n"/"v"/"adj"/"adv" code.
# CEO chose B (compact dot suffix) over full "noun"/"verb" labels
# (2026-06-15). Italic display below target_word.
_POS_LABEL = {
    "n":    "n.",
    "v":    "v.",
    "adj":  "adj.",
    "adv":  "adv.",
}


@dataclass
class GuessWordTiming:
    """Per-word timing for the guess_word layout (v3 — 2026-06-16)."""
    idx: int
    target_word: str
    native_translation: str
    first_letter_hint: str
    part_of_speech: str          # raw "n" | "v" | "adj" | "adv"
    part_of_speech_label: str    # rendered "n." | "v." | "adj." | "adv."
    ipa: str
    example_sentence: str        # one short sentence in target lang
    start: float                 # absolute scene start
    slot: float                  # this word's full duration
    countdown_duration: float    # 3.0s — countdown + hint visible
    reveal_start: float          # absolute, when reveal phase begins
    reveal_slot: float           # how long reveal stays
    reveal_voice_start: float    # absolute, when TTS plays target_word
    reveal_voice_duration: float


def build_guess_word_project(
    *,
    content: GuessWordContent,
    audio_clips: list[AudioClip],
    audio_src_dir: Path,
    out_dir: Path,
    target_lang_name: str = "",
    target_lang: str = "en",
    hyperframes_version: str = "0.6.52",
    channel_dir: Path | None = None,
    native_lang: str = "vi",
    lesson_number: int = 1,
) -> Path:
    """Materialize a HyperFrames project for the guess_word layout.

    Audio clips expected (12 total):
      intro_native, reveal_1, reveal_2, …, reveal_10, outro_native

    Per word timing (one scene):
      0.0s            translate + hint visible
      0.0 → 3.5s      countdown bar shrinks from 100% → 0
      3.5s            hint fades out, target_word + pos label + IPA fade in,
                      target voice plays
      ~5.0s           next word

    NO intro voice or hero card (CEO 2026-06-16) — straight into word 1.
    NO outro hero card — outro_native voice overlays during last 3s after
    word 10.

    CEO 2026-06-16 v3 changes:
    - Dropped intro_native voice clip (was 12 clips, now 11)
    - Hint/reveal use separate HF clips with explicit data-start (NOT CSS
      animation, which HF doesn't honor reliably)
    - Per-word countdown plays chime.mp3 at t=0, 1, 2 (3-2-1 tick suspense)
    - TIME number (3/2/1) shown in card during countdown
    """
    expected = 11
    if len(audio_clips) != expected:
        raise ValueError(f"guess_word expects {expected} audio clips, got {len(audio_clips)}")
    if len(content.words) != 10:
        raise ValueError(f"guess_word expects exactly 10 words, got {len(content.words)}")

    clip_by_name = {c.name: c for c in audio_clips}
    outro_clip = clip_by_name["outro_native"]

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_audio = out_dir / "assets" / "audio"
    assets_audio.mkdir(parents=True, exist_ok=True)
    for clip in audio_clips:
        shutil.copy2(audio_src_dir / clip.file, assets_audio / clip.file)

    # Layered statics (template defaults + channel overrides)
    static_dst = out_dir / "static"
    static_dst.mkdir(parents=True, exist_ok=True)
    has_logo = False
    has_chime = False
    ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".mp3", ".wav"}
    for layer in [TEMPLATE_DIR / "static", (channel_dir / "static") if channel_dir else None]:
        if layer is None or not layer.exists():
            continue
        for f in layer.iterdir():
            if f.is_file() and f.suffix.lower() in ASSET_EXTS:
                shutil.copy2(f, static_dst / f.name)
                if f.name.lower() == "logo.png":
                    has_logo = True
                if f.name.lower() == "chime.mp3":
                    has_chime = True

    # ────── TIMING (no intro voice/card — straight into word 1) ──────
    # CEO 2026-06-16: countdown 3 seconds, integer ticks. Each word:
    #   t=0.0 → hint visible, TIME=3, chime tick
    #   t=1.0 → TIME=2, chime tick
    #   t=2.0 → TIME=1, chime tick
    #   t=3.0 → hint hides, reveal (target+pos+IPA+example) visible, voice plays
    #   t=3.0+voice → small pad → gap → next word
    COUNTDOWN_DURATION = 3.0  # integer seconds for clean 3/2/1
    READ_PAD_AFTER_REVEAL = 1.5  # extra time for example sentence
    GAP_BETWEEN_WORDS = 0.2

    timings: list[GuessWordTiming] = []
    cursor = 0.0
    for i, w in enumerate(content.words, start=1):
        reveal_clip = clip_by_name[f"reveal_{i}"]
        reveal_voice_dur = reveal_clip.duration
        reveal_slot = round(reveal_voice_dur + READ_PAD_AFTER_REVEAL, 2)
        word_slot = round(COUNTDOWN_DURATION + reveal_slot + GAP_BETWEEN_WORDS, 2)

        pos_raw = (w.part_of_speech or "n").lower().strip()
        pos_label = _POS_LABEL.get(pos_raw, f"{pos_raw}.")

        timings.append(GuessWordTiming(
            idx=i,
            target_word=w.target_word,
            native_translation=w.native_translation,
            first_letter_hint=w.first_letter_hint,
            part_of_speech=pos_raw,
            part_of_speech_label=pos_label,
            ipa=w.ipa,
            example_sentence=getattr(w, "example_sentence", "") or "",
            start=round(cursor, 2),
            slot=word_slot,
            countdown_duration=COUNTDOWN_DURATION,
            reveal_start=round(cursor + COUNTDOWN_DURATION, 2),
            reveal_slot=reveal_slot,
            reveal_voice_start=round(cursor + COUNTDOWN_DURATION, 2),
            reveal_voice_duration=round(reveal_voice_dur + 0.1, 2),
        ))
        cursor += word_slot

    # Outro hero card + voice overlay (CEO 2026-06-16: add hero scene).
    # Card appears right after word 10 ends and stays for the full outro window.
    outro_start = round(cursor, 2)
    outro_voice_dur = outro_clip.duration
    outro_voice_start = round(cursor + 0.3, 2)
    outro_voice_duration_render = round(outro_voice_dur + 0.1, 2)
    outro_tail = max(outro_voice_dur + 1.0, 2.5)  # hold the card slightly after voice ends
    outro_slot = round(outro_tail, 2)
    cursor += outro_tail
    total_duration = round(cursor, 2)

    # Flag emojis (use NATIVE_LANG_FLAG for both directions — same mapping)
    native_flag = _native_flag(native_lang)
    target_flag = NATIVE_LANG_FLAG.get((target_lang or "").lower(), "🏳️")

    # Render
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("guess_word.html.j2")
    html = template.render(
        demo_mode=_demo_mode(),
        total_duration=total_duration,
        title_native=content.title_native,
        native_flag=native_flag,
        target_flag=target_flag,
        topic_label=content.topic_label,
        target_lang_name=target_lang_name or "",
        native_lang=native_lang,
        outro_voice_start=outro_voice_start,
        outro_voice_duration=outro_voice_duration_render,
        outro_start=outro_start,
        outro_slot=outro_slot,
        outro_text=content.outro_native,
        words=[t.__dict__ for t in timings],
        words_count=len(timings),
        lesson_number=lesson_number,
        has_chime=has_chime,
        channel_name=os.environ.get("CHANNEL_NAME", DEFAULT_CHANNEL_NAME),
        channel_tagline=os.environ.get("CHANNEL_TAGLINE", DEFAULT_CHANNEL_TAGLINE),
        avatar_emoji=os.environ.get("AVATAR_EMOJI", DEFAULT_AVATAR_EMOJI),
        theme=_theme_from_env(),
        has_logo=has_logo,
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    # HyperFrames project metadata
    (out_dir / "meta.json").write_text(
        json.dumps({"id": out_dir.name, "name": out_dir.name}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "hyperframes.json").write_text(
        json.dumps({
            "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
            "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
            "paths": {"blocks": "compositions", "components": "compositions/components", "assets": "assets"},
        }, indent=2),
        encoding="utf-8",
    )
    (out_dir / "package.json").write_text(
        json.dumps({
            "name": out_dir.name, "private": True, "type": "module",
            "scripts": {
                "render": f"npx --yes hyperframes@{hyperframes_version} render",
                "lint":   f"npx --yes hyperframes@{hyperframes_version} lint",
            },
        }, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "layout_type": "guess_word",
        "total_duration": total_duration,
        "title_native": content.title_native,
        "words": [
            {"target_word": t.target_word,
             "native_translation": t.native_translation,
             "first_letter_hint": t.first_letter_hint,
             "part_of_speech": t.part_of_speech,
             "ipa": t.ipa}
            for t in timings
        ],
        "caption": content.caption,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir
