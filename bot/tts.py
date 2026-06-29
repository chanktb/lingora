"""Generate TTS audio with Microsoft Edge TTS (free, no API key).

Voice selection is dynamic: we query Edge TTS's voice list once and pick
a Neural voice for any (lang_code, gender) the user asks for. Supports
~78 languages (every locale Edge ships with).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import edge_tts

# When a language has multiple regions, prefer this region (else first by name).
PREFERRED_LOCALE = {
    "en": "en-US",
    "es": "es-ES",
    "pt": "pt-BR",
    "zh": "zh-CN",
    "ar": "ar-SA",
    "fr": "fr-FR",
    "de": "de-DE",
    "nl": "nl-NL",
    "it": "it-IT",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "sw": "sw-KE",
    "sr": "sr-RS",
}

# Voices that are explicitly children/cartoon — exclude from adult-only picks.
# (Names verified against Edge TTS voice catalog 2025/2026.)
KNOWN_CHILD_VOICES = {
    "en-US-AnaNeural",        # child female
    "en-GB-MaisieNeural",     # child female
    "es-MX-CecilioNeural",    # child male
}

# ContentCategories tags that indicate non-adult / kids voices
CHILD_CATEGORY_TAGS = {"Cartoon", "Children", "Kids", "Cute"}

# Cache: (lang, gender) -> voice ShortName
_voice_cache: dict[tuple[str, str], str] = {}
_voices_loaded: list[dict] | None = None


@dataclass
class AudioClip:
    name: str
    text: str
    file: str
    duration: float


@dataclass
class AudioSegment:
    """Spec for one piece of audio to synthesize."""
    name: str             # used as filename stem (e.g. "intro", "p1_target", "p1_vi", "outro")
    text: str
    voice: str            # Edge TTS voice short name
    rate: str = "-10%"


def _ffprobe_bin() -> str:
    base = os.environ.get("FFMPEG_BIN", "")
    if base:
        candidate = Path(base) / "ffprobe.exe"
        if candidate.exists():
            return str(candidate)
    return "ffprobe"


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            _ffprobe_bin(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _ffmpeg_bin() -> str:
    base = os.environ.get("FFMPEG_BIN", "")
    if base:
        candidate = Path(base) / "ffmpeg.exe"
        if candidate.exists():
            return str(candidate)
    return "ffmpeg"


def _strip_silence_to_wav(mp3_path: Path) -> Path:
    """Strip both-ends silence AND convert MP3 → WAV PCM.

    CEO bug report 2026-06-15 v3 (whats_this still drifts late-video on
    russianpath even after silence-strip + MP3 re-encode landed earlier):

    Even after stripping leading + trailing silence, LAME MP3 re-encode
    adds 2257 samples (~51 ms at 44.1 kHz) of encoder padding to the
    START of every output file. Chrome `<audio>` schedules data-start
    exactly, but the audible sample doesn't appear until +51 ms in.
    With 10 items in whats_this, cumulative drift = ~510 ms by item 10 —
    exactly matching the "càng về sau càng chậm" complaint.

    Permanent fix: stop re-encoding to MP3. Output WAV PCM (pcm_s16le)
    instead. WAV has ZERO encoder delay → first sample IS first audible
    phoneme → cumulative drift = 0 regardless of item count.

    Trade-off: WAV files ~10× larger than 128 kbps MP3. For typical
    ≤30-clip lessons, total audio < 50 MB; HyperFrames + Chrome handle
    WAV natively, so no decoder change needed.

    Returns the new audio path (.wav on success, original .mp3 on failure
    so the caller has a working file either way).
    """
    src = mp3_path
    dst = mp3_path.with_suffix(".wav")
    try:
        result = subprocess.run(
            [
                _ffmpeg_bin(),
                "-v", "error",
                "-y",
                "-i", str(src),
                "-af",
                "silenceremove="
                "start_periods=1:start_silence=0:start_threshold=-40dB:"
                "stop_periods=-1:stop_silence=0:stop_threshold=-40dB:"
                "detection=peak",
                "-c:a", "pcm_s16le",
                "-ar", "44100",
                "-ac", "1",
                str(dst),
            ],
            capture_output=True,
            timeout=20,
        )
        if result.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            src.unlink(missing_ok=True)
            return dst
        if dst.exists():
            dst.unlink(missing_ok=True)
        sys.stderr.write(
            f"[tts] silence-strip-to-wav failed for {src.name}, keeping mp3\n"
        )
        return src
    except Exception as exc:
        if dst.exists():
            dst.unlink(missing_ok=True)
        sys.stderr.write(f"[tts] silence-strip-to-wav error {src.name}: {exc}\n")
        return src


async def _load_voices() -> list[dict]:
    global _voices_loaded
    if _voices_loaded is None:
        _voices_loaded = await edge_tts.list_voices()
    return _voices_loaded


def _is_adult_voice(v: dict) -> bool:
    """True if the voice is an adult (not a child / cartoon character)."""
    if v.get("ShortName") in KNOWN_CHILD_VOICES:
        return False
    cats = set(v.get("ContentCategories") or [])
    if cats & CHILD_CATEGORY_TAGS:
        return False
    return True


async def pick_voice_async(lang_code: str, gender: str = "any") -> str:
    """Pick a Neural voice for the given language and gender.

    gender:
      "female" → adult female only
      "male"   → adult male only
      "any"    → random pick between adult female and male (default)

    Raises ValueError if Edge TTS has no voice for the requested language.
    """
    lang_code = lang_code.lower()
    gender_in = (gender or "any").lower()
    if gender_in not in {"male", "female", "any"}:
        gender_in = "any"

    # For "any", actually pick random gender per call (don't cache that one).
    if gender_in == "any":
        gender_in = random.choice(["male", "female"])

    gender_title = "Male" if gender_in == "male" else "Female"
    key = (lang_code, gender_title)
    if key in _voice_cache:
        return _voice_cache[key]

    voices = await _load_voices()
    matching = [
        v for v in voices
        if v["Locale"].lower().startswith(f"{lang_code}-") and _is_adult_voice(v)
    ]
    if not matching:
        # Fallback: same language but without adult filter (rare; child-only language)
        matching = [v for v in voices if v["Locale"].lower().startswith(f"{lang_code}-")]
    if not matching:
        raise ValueError(f"Edge TTS has no voice for language '{lang_code}'")

    same_gender = [v for v in matching if v["Gender"] == gender_title]
    pool = same_gender or matching  # fall back to other gender if needed

    preferred = PREFERRED_LOCALE.get(lang_code)
    if preferred:
        in_preferred = [v for v in pool if v["Locale"].lower() == preferred.lower()]
        if in_preferred:
            pool = in_preferred

    # When multiple candidates remain, pick a random one (some langs have 3+ female voices)
    pick = random.choice(pool)["ShortName"]
    _voice_cache[key] = pick
    return pick


def pick_voice(lang_code: str, gender: str = "any") -> str:
    """Sync wrapper for pick_voice_async."""
    return asyncio.run(pick_voice_async(lang_code, gender))


# Text matching this regex has no pronounceable content (only whitespace +
# punctuation/symbols). Edge TTS rejects these inputs with NoAudioReceived
# because the synth pipeline emits no audio chunks. Callers (e.g. Lingora
# composer's quiz label slot) sometimes intentionally pass a single
# punctuation character ("·") as a "silent pause placeholder" — we honour
# that intent by writing a real silent MP3 instead of round-tripping Edge TTS.
_UNPRONOUNCEABLE_RE = re.compile(r"^[\W_]+$", re.UNICODE)

# Default duration for sentinel-silence MP3s. Matches what Edge TTS used to
# emit for "·" before it became unreliable (~100 ms). Composer schedules
# GAP_AFTER_LABEL=0.15s + PAD_AFTER_TARGET=0.45s on top, so 100 ms is enough
# to read as a natural pause between quiz answers.
_SILENCE_MS_DEFAULT = 100


def _write_silence_mp3(out_path: Path, duration_ms: int = _SILENCE_MS_DEFAULT) -> None:
    """Generate a pure-silent MP3 at out_path using ffmpeg anullsrc.

    Used to bypass Edge TTS for "unpronounceable" inputs (single punctuation,
    whitespace-only) where Edge TTS would otherwise raise NoAudioReceived.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _ffmpeg_bin(),
            "-v", "error",
            "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=24000:cl=mono",
            "-t", f"{duration_ms / 1000:.3f}",
            "-c:a", "libmp3lame",
            "-b:a", "64k",
            str(out_path),
        ],
        capture_output=True,
        timeout=10,
        check=True,
    )


async def _synth_one(text: str, voice: str, rate: str, out_path: Path) -> Path:
    """Synthesize a single segment with retry-on-NoAudioReceived.

    Returns the final on-disk audio path. On normal Edge TTS path, the file
    is converted MP3 → WAV by `_strip_silence_to_wav` and the .wav path is
    returned. On the unpronounceable-input fast-path, a silent MP3 is
    written and that path is returned (single-frame silence — no encoder-
    delay cost worth optimising).

    Edge-TTS rate-limits silently when many synth calls land in a tight
    window — symptom is empty stream (NoAudioReceived). Retry with
    increasing backoff (3s, 7s, 15s) covers all observed rate-limit
    windows in practice.
    """
    if not text or _UNPRONOUNCEABLE_RE.match(text):
        _write_silence_mp3(out_path)
        return out_path

    last_exc: Exception | None = None
    backoffs = [3.0, 7.0, 15.0]
    for attempt in range(4):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(out_path))
            # Strip silence + convert to WAV to eliminate LAME encoder padding.
            # Returns new .wav path (or original .mp3 if ffmpeg fails).
            return _strip_silence_to_wav(out_path)
        except edge_tts.exceptions.NoAudioReceived as exc:
            last_exc = exc
            if attempt < 3:
                wait = backoffs[attempt]
                sys.stderr.write(
                    f"[tts] NoAudioReceived attempt {attempt+1}/4 — "
                    f"retrying after {wait}s (voice={voice!r}, "
                    f"text_len={len(text)})\n"
                )
                await asyncio.sleep(wait)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return out_path


async def synth_segments_async(
    segments: list[AudioSegment],
    out_dir: Path,
) -> list[AudioClip]:
    """Synthesize a list of audio segments (each with its own voice).

    Used for v6 video pipeline where intro/outro use native voice
    and phrases use both target + native voices.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[AudioClip] = []
    for i, seg in enumerate(segments):
        out_mp3 = out_dir / f"{seg.name}.mp3"
        final_path = await _synth_one(seg.text, seg.voice, seg.rate, out_mp3)
        dur = _probe_duration(final_path)
        results.append(AudioClip(name=seg.name, text=seg.text, file=final_path.name, duration=dur))
        # Throttle: 250 ms between synth calls to avoid Microsoft Edge-TTS
        # rate limit (~10 req / few-seconds window). Skip after last seg.
        if i < len(segments) - 1:
            await asyncio.sleep(0.25)

    manifest = out_dir / "manifest.json"
    manifest.write_text(
        json.dumps([c.__dict__ for c in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


# Legacy single-voice API (kept for backward compat with old tests)
async def synth_phrases_async(
    *,
    intro_text: str,
    phrase_texts: list[str],
    out_dir: Path,
    voice: str,
    rate: str = "-10%",
) -> list[AudioClip]:
    segments = [AudioSegment("intro", intro_text, voice, rate)] + [
        AudioSegment(f"p{i + 1}", text, voice, rate)
        for i, text in enumerate(phrase_texts)
    ]
    return await synth_segments_async(segments, out_dir)


def synth_phrases(
    *,
    intro_text: str,
    phrase_texts: list[str],
    out_dir: Path,
    voice: str,
    rate: str = "-10%",
) -> list[AudioClip]:
    return asyncio.run(
        synth_phrases_async(
            intro_text=intro_text,
            phrase_texts=phrase_texts,
            out_dir=out_dir,
            voice=voice,
            rate=rate,
        )
    )


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    lang = sys.argv[1] if len(sys.argv) > 1 else "de"
    gender = sys.argv[2] if len(sys.argv) > 2 else "female"
    voice = pick_voice(lang, gender)
    print(f"{lang}/{gender} → {voice}")
