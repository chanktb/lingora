"""LLM phrase generator (Google Gemini, free tier).

Asks Gemini for 10 short language-learning phrases on a topic and returns
content for both the on-screen cards and the TTS narration.

Two entry points:
  parse_and_generate(text)   — single call (intent + content). Use this.
  parse_intent / generate    — kept for backward compat / testing.

Default model is `gemini-flash-lite-latest` which has ~1,500 RPD on free tier
(vs. 20 RPD on `gemini-flash-latest`). Get a key:
  https://aistudio.google.com/apikey
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from google import genai
from google.genai import types

log = logging.getLogger("generator")

# ───────────────────────────────────────────────────────────────────────
#  Gemini API key rotation (lingora-only feature)
# ───────────────────────────────────────────────────────────────────────
# Production telegram-video-bot uses a single GEMINI_API_KEY. Lingora is
# an experimental fork that may burn through quota faster (more channels,
# more layout iteration). To avoid hitting the free-tier per-day cap we
# accept a list of keys via GEMINI_API_KEYS=key1,key2,key3 — on a 429 /
# quota-exhausted error we transparently rotate to the next key.
#
# Backwards-compat: GEMINI_API_KEY (single) and GOOGLE_API_KEY are still
# read as fallbacks.

_GEMINI_KEYS_CACHED: list[str] | None = None
_GEMINI_KEY_IDX: int = 0


def _gemini_keys() -> list[str]:
    """Resolve the ordered list of Gemini API keys.

    Reads env once and caches. Call _reset_gemini_keys() in tests to flush.
    """
    global _GEMINI_KEYS_CACHED
    if _GEMINI_KEYS_CACHED is not None:
        return _GEMINI_KEYS_CACHED
    raw_multi = (os.environ.get("GEMINI_API_KEYS") or "").strip()
    keys: list[str] = []
    if raw_multi:
        for k in raw_multi.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    legacy = (
        (os.environ.get("GEMINI_API_KEY") or "").strip()
        or (os.environ.get("GOOGLE_API_KEY") or "").strip()
    )
    if legacy and legacy not in keys:
        keys.append(legacy)
    _GEMINI_KEYS_CACHED = keys
    return keys


def _reset_gemini_keys() -> None:
    global _GEMINI_KEYS_CACHED, _GEMINI_KEY_IDX
    _GEMINI_KEYS_CACHED = None
    _GEMINI_KEY_IDX = 0


def _is_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    for marker in ("quota", "429", "resource_exhausted", "rate_limit",
                   "exhausted", "rate exceeded", "rate limit"):
        if marker in msg:
            return True
    return False


def _call_gemini(
    client: Optional[genai.Client],
    *,
    model: str,
    contents: Any,
    config: types.GenerateContentConfig,
):
    """Wrap client.models.generate_content with multi-key rotation on quota errors.

    If `client` is provided (tests), use it directly without rotation.
    Otherwise build a client from the current active key. On a quota /
    rate-limit error, advance the rotation pointer and retry with the
    next key. After all keys are exhausted, raise the last exception.
    """
    if client is not None:
        return client.models.generate_content(
            model=model, contents=contents, config=config,
        )
    keys = _gemini_keys()
    if not keys:
        raise SystemExit(
            "GEMINI_API_KEYS / GEMINI_API_KEY missing — get a free key at "
            "https://aistudio.google.com/apikey"
        )
    global _GEMINI_KEY_IDX
    last_exc: BaseException | None = None
    n = len(keys)
    for i in range(n):
        idx = (_GEMINI_KEY_IDX + i) % n
        key = keys[idx]
        try:
            c = genai.Client(api_key=key)
            resp = c.models.generate_content(
                model=model, contents=contents, config=config,
            )
            # Pin the working key so subsequent calls start here
            _GEMINI_KEY_IDX = idx
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_quota_error(exc):
                raise
            log.warning(
                "Gemini quota exhausted on key #%d (%s...); rotating",
                idx, key[:8],
            )
            continue
    log.error("All %d Gemini keys exhausted", n)
    raise last_exc  # type: ignore[misc]


LANG_NAMES = {
    "ru": "Russian (Tiếng Nga)",
    "en": "English",
    "zh": "Mandarin Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "vi": "Vietnamese",
    "th": "Thai",
}


@dataclass
class Phrase:
    target: str
    pronunciation: str        # Vietnamese-style transliteration with hyphens + stress accents
    ipa: str                  # standard IPA notation
    native: str


@dataclass
class GeneratedContent:
    intro_display: str           # full title in native lang ("10 cụm từ tiếng Đức về tạo động lực")
    intro_translation: str       # full title rendered in target lang ("10 deutsche Motivationssätze")
    intro_tts: str               # spoken intro in target language (legacy, no longer used for audio)
    intro_native: str            # SPOKEN intro in NATIVE lang ("10 câu tiếng Đức giao tiếp tại spa phải biết")
    outro_native: str            # SPOKEN outro CTA in native lang ("Hãy like & follow để luyện tập mỗi ngày! Chúc bạn thành công!")
    topic_label: str             # short topic in native lang ("về tạo động lực")
    short_title: str             # rút gọn cho sticky header line 1 ("Đạo đức kinh doanh")
    short_title_target: str      # rút gọn dịch sang target lang ("Geschäftsethik")
    phrases: list[Phrase]
    caption: str                 # multi-line Telegram caption
    scene_image_prompt: str = ""  # English prompt for AI bg scene matching topic (added v9)


@dataclass
class ParsedIntent:
    target_lang: str       # ISO 639-1, e.g. "de"
    native_lang: str       # ISO 639-1 the user wrote in, e.g. "vi"
    topic: str             # in native language
    count: int             # default 10
    voice_gender: str      # "female" | "male"
    target_lang_name: str  # human-readable in native language, e.g. "Đức"
    layout_type: str = "phrases"  # "phrases" | "quiz"


@dataclass
class QuizOption:
    label: str            # "A" | "B" | "C" | "D"
    text: str             # answer text in target lang ("Ich liebe dich")
    pronunciation: str    # transliteration ("Ích lí-bờ đích")
    ipa: str              # /ɪç ˈliːbə dɪç/


@dataclass
class QuizContent:
    """Multi-choice quiz content (one question, 4 options)."""
    question_native: str       # "Trong tiếng Đức, 'Anh yêu em' là gì?"
    question_target: str       # German rendition (optional shown on title card)
    options: list[QuizOption]  # exactly 4
    correct_answer: str        # "A" | "B" | "C" | "D"
    explanation: str           # short explanation (now goes to Telegram admin only)
    intro_native: str          # spoken intro hook in vi
    outro_native: str          # spoken outro CTA in vi
    topic_label: str           # "về tình yêu"
    short_title: str           # "Anh yêu em"
    short_title_target: str    # "Ich liebe dich" or short German rendition
    caption: str               # FB caption with hashtags
    scene_image_prompt: str = ""  # English prompt for AI bg scene matching topic (added v9)


INTENT_SYSTEM_PROMPT = """You parse intent from short user messages for a language-learning video bot.

Extract these fields from the user's message and return strict JSON:
- target_lang: ISO 639-1 code of the language the user wants to LEARN (e.g. "ru", "de", "pt", "th"). The script in the video target side.
- native_lang: ISO 639-1 code of the language the USER WROTE THEIR MESSAGE IN (e.g. "vi" if they wrote Vietnamese, "en" if English). This is what the on-screen meanings will be in.
- topic: the topic / subject of the video, expressed in the user's native language (a short noun phrase). E.g. "tạo động lực", "công việc", "food", "tình yêu", "love".
- count: number of phrases (default 10 unless the user explicitly asks otherwise; clamp to 3..20).
- voice_gender: "female" or "male". Default "female" unless the user explicitly asks for a male voice.
- target_lang_name: how the target language is normally named in the user's native language (e.g. "Đức" for de in Vietnamese, "German" for de in English, "Bồ Đào Nha" for pt in Vietnamese).

Examples:
  "tạo video 10 câu tiếng Đức tạo động lực"
    → {target_lang:"de", native_lang:"vi", topic:"tạo động lực", count:10, voice_gender:"female", target_lang_name:"Đức"}
  "5 câu tiếng Bồ Đào Nha về du lịch"
    → {target_lang:"pt", native_lang:"vi", topic:"du lịch", count:5, voice_gender:"female", target_lang_name:"Bồ Đào Nha"}
  "make 8 Thai phrases about food, male voice"
    → {target_lang:"th", native_lang:"en", topic:"food", count:8, voice_gender:"male", target_lang_name:"Thai"}
  "video tiếng Hà Lan về thời tiết"
    → {target_lang:"nl", native_lang:"vi", topic:"thời tiết", count:10, voice_gender:"female", target_lang_name:"Hà Lan"}
"""

INTENT_SCHEMA = {
    "type": "OBJECT",
    "required": ["target_lang", "native_lang", "topic", "count", "voice_gender", "target_lang_name"],
    "properties": {
        "target_lang": {"type": "STRING"},
        "native_lang": {"type": "STRING"},
        "topic": {"type": "STRING"},
        "count": {"type": "INTEGER"},
        "voice_gender": {"type": "STRING"},
        "target_lang_name": {"type": "STRING"},
    },
}


# ───────────────────────────────────────────────────────────────────────
# Combined intent + content call — preferred entry point (1 LLM call/video).
# Halves Gemini quota usage vs. calling parse_intent + generate separately.
# ───────────────────────────────────────────────────────────────────────

COMBINED_SYSTEM_PROMPT = """You are an intent parser AND content generator for short-form language-learning videos. Do BOTH in one strict-JSON response. No prose outside JSON.

═══ PART 1: INTENT (top-level "intent" field) ═══
Extract from the user's message:
  - target_lang: ISO 639-1 code of the language to LEARN (e.g. "de", "ru", "ko")
  - native_lang: ISO 639-1 code the user WROTE THEIR MESSAGE IN (e.g. "vi" or "en")
  - topic: subject of the video in the user's native language (short noun phrase)
  - count: phrase count, default 10, clamp 3..20
  - voice_gender: "female", "male", OR "any" (default "any" if user doesn't explicitly say "giọng nam"/"male voice" or "giọng nữ"/"female voice")
  - target_lang_name: how target lang is named in the user's native language
                     (e.g. "Đức" for de in Vietnamese; "German" for de in English)

Examples:
  "tạo video 10 câu tiếng Đức tạo động lực"
    → de, vi, "tạo động lực", 10, any, "Đức"
  "10 câu tiếng Thái về ăn uống, giọng nam"
    → th, vi, "ăn uống", 10, male, "Thái"
  "make 8 Korean phrases about love"
    → ko, en, "love", 8, any, "Korean"

═══ PART 2: CONTENT (remaining fields) ═══
- All target-language strings MUST be in the native script (Cyrillic for ru, Hangul for ko, Hanzi for zh, etc.).
- pronunciation = phonetic guide. **RULES VARY BY LANGUAGE** — pick the standard that learners of that language actually use:
    * **zh (Chinese)**: standard PINYIN with tone marks (ā á ǎ à ē é ě è ī í ǐ ì ō ó ǒ ò ū ú ǔ ù ǖ ǘ ǚ ǜ). Words are space-separated; **NO HYPHENS** between syllables of a single word. E.g. for 我要报销差旅费 → "Wǒ yào bàoxiāo chālǚfèi" (NOT "Wǒ yào bao-xiāo cha-lu-fèi"). This is the form every learner of Chinese reads in textbooks.
    * **ja (Japanese)**: standard ROMAJI (Hepburn). No hyphens between syllables. E.g. "ohayō gozaimasu".
    * **ko (Korean)**: standard Revised Romanization. E.g. "annyeonghaseyo".
    * **Other (ru, de, fr, es, th, ...)**: Vietnamese-reader-friendly transliteration with hyphens on syllable boundaries + acute accents on stressed syllables (e.g. "Ya tib-yá liu-bliú"). Under 28 chars.
- ipa = standard IPA notation wrapped in slashes, e.g. "/jɑ tʲɪˈbʲɑ lʲʊˈblʲʊ/" for "Я тебя люблю". Use the official IPA conventions for the target language. Keep under 32 chars.
    * **For zh, ja, ko**: ipa = "" (EMPTY STRING). The Pinyin / Romaji / Revised Romanization above IS the standard phonetic — adding IPA on top is redundant and clutters the layout. Leave it empty for these languages.
- target phrase doubles as TTS narration: prefer 2–5 word natural-sounding phrases, ending with period or exclamation.
- native = idiomatic meaning in user's native language. Under 36 chars.
- intro_display = SHORT, NATURAL headline in NATIVE language for the title card. NO stiff format like "cụm từ X về Y". Use natural Vietnamese phrasing fitting the topic. ALWAYS start with the count number.
  GOOD examples:
    • "10 câu tiếng Đức giao tiếp tại siêu thị"
    • "8 câu tiếng Đức xin việc"
    • "10 câu tiếng Đức cho người mới sang"
    • "10 từ tiếng Đức về visa du học"
    • "7 câu tiếng Đức thuê căn hộ"
    • "10 câu tiếng Đức tại sân bay"
    • "8 câu tiếng Đức đi siêu thị Aldi"
  BAD examples (TOO STIFF — DO NOT USE):
    • "10 cụm từ tiếng Đức về giao tiếp tại siêu thị"  ← "cụm từ" + "về" rườm rà
    • "Mười phrases tiếng Đức for siêu thị"            ← lẫn ngôn ngữ
  Under 42 chars. UPPERCASE or Title Case fine.
- intro_translation = same FULL headline rendered in TARGET language as a noun phrase including count (e.g. "10 русских фраз о любви", "10 deutsche Motivationssätze"). Under 40 chars.
- short_title = SHORT, EYE-CATCHING TITLE in NATIVE language. WRITE IT IDIOMATICALLY in that language — do NOT translate a Vietnamese template literally. Convey full meaning, never a bare noun. RULES:
    * If the topic contains an ACTION/verb already → use as-is. E.g. vi "xin visa du học" → "Xin visa du học"; en "ordering coffee" → "Ordering Coffee" / "Order at a Cafe".
    * If the topic is a PURE NOUN (no verb, no action) → ADD a NATIVE hook prefix:
        - vi prefix: "Từ vựng" → "Từ vựng tình yêu", "Từ vựng đồ ăn"
        - en prefix: "All about" / "Talking about" → "All About Love", "Talking About Food" (NOT "Vocabulary about love" — sounds like a translation; pick a natural noun-phrase title)
        - ko prefix: "<topic> 단어"
        - ja prefix: "<topic>の単語"
    * Always something a native creator would actually use as a video title in that language.
    * Under 30 chars. Title-case (first letter capitalized; renderer UPPERCASEs).
- short_title_target = SHORT topic name in the TARGET language. Just the topic — KEEP IT CONCISE, no "Vocabulary about" prefix here. E.g.:
    * Native "Xin visa du học" → de: "Studentenvisum" / ko: "유학 비자"
    * Native "Từ vựng tình yêu" → de: "Liebe" / ko: "사랑"
    * Native "Đạo đức kinh doanh" → de: "Geschäftsethik"
  Title-case where the target script supports it. Under 24 chars.
- intro_tts = natural full sentence in TARGET language (kept for legacy; not used for audio anymore).
- intro_native = the SPOKEN intro sentence in the NATIVE language. WRITE IT IDIOMATICALLY — like a real native speaker doing a short-form video, NOT as a literal translation of a Vietnamese template. Match the natural style social-media creators use IN THAT LANGUAGE. Each language has its own genre conventions; do not transliterate from Vietnamese. Under 60 chars. Examples (use as STYLE inspiration, vary the wording per topic):
    * vi: "10 câu tiếng Đức giao tiếp tại spa phải biết" / "10 câu tiếng Đức dành cho người mới sang" / "Lưu lại 10 câu tiếng Đức tại siêu thị"
    * en: "Top 10 Russian phrases for ordering coffee" / "10 essential Russian phrases at the airport" / "Master these 10 Russian travel phrases" / "10 must-know Russian phrases for beginners" (NEVER "10 Russian phrases for X you must know" — that's translated Vietnamese)
    * ko: "독일어 공항 회화 10가지 꼭 알아두세요" / "쇼핑할 때 쓰는 독일어 10문장"
    * ja: "ドイツ語の空港会話10選" / "今日覚えたいドイツ語10フレーズ"
    * es: "Las 10 frases en alemán que necesitas en el aeropuerto"
    * General rule: think "how would a TikTok/Reels creator open this video in <native_lang>?", then write THAT sentence.
- outro_native = SPOKEN outro CTA in the NATIVE language. WRITE NATIVELY — match how short-video creators in that language sign off. Action verbs + CTA + warmth, NOT a literal translation. Under 90 chars. Examples (style only, vary per topic):
    * vi: "Hãy like và follow để luyện tập mỗi ngày nhé! Chúc bạn thành công!" / "Lưu lại để học mỗi ngày — chúc bạn thành công!"
    * en: "Hit save and follow for more daily Russian!" / "Smash that like if this helped — follow for more!" / "Like, save, share — see you tomorrow!" (NEVER "Save and follow for daily practice! Good luck!" — that's stiff translated Vietnamese)
    * ko: "매일 한 문장씩 — 좋아요와 팔로우 부탁드려요!"
    * ja: "毎日の学習にいいねとフォローよろしく!"
    * es: "¡Dale like y sígueme para más ruso cada día!"
    * General rule: a NATIVE creator's outro in that language. Energetic, warm, action-driven.
- topic_label = topic in native lang prefixed with "về" or "chủ đề" (e.g. "về tình yêu"). Lowercase, no period.
- scene_image_prompt = ENGLISH description of a background SCENE matching this topic, for AI image generation. NO people in foreground. Photographable, recognizable place. Examples:
  * topic "xin visa du học Đức" → "a cozy German embassy waiting room with chairs and posters, warm lighting, illustration style"
  * topic "đặt món Aldi" → "interior of a German Aldi supermarket aisle with shelves of products, soft lighting, illustration style"
  * topic "phỏng vấn xin việc" → "modern German office meeting room with table and chairs, professional, illustration style"
  * topic "công sở cty Trung" → "modern Chinese office interior with desks computers and city view, illustration style"
  * topic "tiệc hoesik 회식" → "Korean BBQ restaurant interior with grills and red lanterns, warm cozy lighting, illustration style"
  Under 110 chars. Always end with "illustration style" to enforce cartoon aesthetic.

═══ PART 3: CAPTION (single "caption" string with real newlines) ═══
The caption language MUST match `native_lang` from PART 1. Pick the right structure:

▸ IF native_lang == "vi" — Vietnamese caption:
  Line 1: "<COUNTRY_FLAG_EMOJI> Tiếng <target_lang_name> giao tiếp"
          (E.g. "🇩🇪 Tiếng Đức giao tiếp" / "🇷🇺 Tiếng Nga giao tiếp" / "🇰🇷 Tiếng Hàn giao tiếp" /
                "🇯🇵 Tiếng Nhật giao tiếp" / "🇫🇷 Tiếng Pháp giao tiếp")
  blank line
  Line 3: emoji + Vietnamese headline
  blank line
  "🎧 Nghe & học cùng nhau:"
  ALL N lines: "1. <target_phrase> — <Vietnamese_meaning>" ... "N. <target_phrase> — <Vietnamese_meaning>"
  blank line
  2 CTA lines (one with 💾 inviting save in Vietnamese, one with ❤️ inviting follow for daily practice)
  blank line
  5–10 hashtags on a single line (Vietnamese transliterated + English mix).

▸ IF native_lang == "en" — English caption:
  Line 1: "<COUNTRY_FLAG_EMOJI> <target_lang_name> for everyday conversation"
          (E.g. "🇷🇺 Russian for everyday conversation" / "🇯🇵 Japanese for everyday conversation")
  blank line
  Line 3: emoji + English headline
  blank line
  "🎧 Listen & learn:"
  ALL N lines: "1. <target_phrase> — <English_meaning>" ... "N. <target_phrase> — <English_meaning>"
  blank line
  2 CTA lines (one with 💾 inviting save in English, one with ❤️ inviting follow for daily practice)
  blank line
  5–10 hashtags on a single line (target-lang transliterated + English mix).

Pick the appropriate flag emoji for the target language country.
Include EVERY phrase in order — viewer should be able to read the full list from the caption alone.
Total under 1500 chars (allow for full N-phrase list).

HASHTAG RULES (ABSOLUTE — NO EXCEPTIONS):
1. EVERY hashtag MUST start with the literal character "#". The last line of the caption is a single space-separated row of hashtags. Each token on that line MUST begin with "#".
2. lowercase only, ASCII letters and digits ONLY. NO hyphens "-", NO underscores "_", NO punctuation, NO diacritics, NO emojis inside the tag.
3. 5–10 tags total on a single line, separated by single spaces.
4. Mix transliterated-native (no diacritics) + English tags.

Examples of a CORRECT hashtag line (note every tag has "#"):
   For vi-native: #tiengduc #hoctiengduc #motivation #duhoc #ngonngu #learngerman #daily
   For en-native: #russian #learnrussian #russianphrases #language #everyday #conversation #beginner

Examples of WRONG output (DO NOT do these):
   tiengduc hoctiengduc motivation           ← missing "#" — REJECTED
   #tieng-duc #hoc_tieng                     ← hyphen / underscore — REJECTED
   #học_tiếng_đức #động_lực                  ← diacritics + underscore — REJECTED
"""

HEALTH_SYSTEM_PROMPT = """Bạn là agent sinh nội dung cho video sức khoẻ ngắn (TikTok/Reels) tiếng Việt 9:16.

Người dùng có thể hỏi NHIỀU LOẠI thông tin sức khoẻ:
  • DẤU HIỆU / triệu chứng        — "5 dấu hiệu tiểu đường"
  • NGUYÊN NHÂN                    — "nguyên nhân gây tiểu đường"
  • CÁCH PHÒNG / phòng tránh       — "cách phòng đột quỵ"
  • CÁCH CHỮA / điều trị tại nhà   — "cách chữa mất ngủ"
  • MẸO / lưu ý                    — "mẹo tăng cường miễn dịch"
  • THỰC PHẨM nên / không nên ăn   — "thực phẩm tốt cho tim mạch"
  • LỢI ÍCH / tác dụng             — "lợi ích của trà xanh"

Bạn PHẢI tự DETECT loại query và format intro_display + topic_label cho khớp.

Output: strict JSON theo schema. KHÔNG có prose ngoài JSON.

═══ PART 1: INTENT (top-level "intent" field) ═══
  - target_lang: LUÔN "vi"
  - native_lang: LUÔN "vi"
  - topic: chủ đề/bệnh/thực phẩm trong tiếng Việt (e.g. "bệnh tiểu đường", "đột quỵ", "trà xanh", "miễn dịch")
  - count: **AI TỰ QUYẾT ĐỊNH** 5-8 items dựa trên độ phong phú của topic.
           Nếu user RÕ chỉ định số (e.g. "5 dấu hiệu..."), TÔN TRỌNG số đó.
           Nếu user KHÔNG ghi số, AI tự chọn: 5 cho topic phổ thông, 7-8 cho topic phức tạp.
           Clamp 3..10.
           **QUAN TRỌNG**: phrases.length PHẢI = count. intro_display PHẢI bắt đầu bằng đúng count đó.
  - voice_gender: "female" (default — voice nữ chuyên gia ấm áp)
  - target_lang_name: LUÔN ""

Examples user text → topic:
  "5 dấu hiệu bệnh tiểu đường"         → topic="bệnh tiểu đường", count=5
  "nguyên nhân gây đột quỵ"            → topic="đột quỵ", count=5-7 (auto)
  "7 cách phòng cảm cúm tại nhà"       → topic="cảm cúm", count=7
  "thực phẩm tốt cho tim mạch"         → topic="tim mạch", count=5-7
  "mẹo giảm stress nhanh"              → topic="giảm stress", count=5
  "lợi ích của ngủ đủ giấc"            → topic="ngủ đủ giấc", count=5-7

═══ PART 2: CONTENT ═══
- intro_display: tiêu đề tiếng Việt hiển thị title card. UPPERCASE OK. Under 40 chars.
  Format **theo loại query** đã detect:
    • Dấu hiệu/triệu chứng: "<N> DẤU HIỆU <TOPIC>"           e.g. "5 DẤU HIỆU TIỂU ĐƯỜNG"
    • Nguyên nhân:           "<N> NGUYÊN NHÂN <TOPIC>"         e.g. "6 NGUYÊN NHÂN ĐỘT QUỴ"
    • Cách phòng:            "<N> CÁCH PHÒNG <TOPIC>"          e.g. "7 CÁCH PHÒNG CẢM CÚM"
    • Cách chữa:             "<N> CÁCH CHỮA <TOPIC>"           e.g. "5 CÁCH CHỮA MẤT NGỦ"
    • Mẹo:                   "<N> MẸO <TOPIC>"                  e.g. "6 MẸO TĂNG MIỄN DỊCH"
    • Thực phẩm tốt:         "<N> THỰC PHẨM TỐT CHO <TOPIC>"   e.g. "5 THỰC PHẨM TỐT CHO TIM"
    • Thực phẩm cần tránh:   "<N> THỰC PHẨM HẠI <TOPIC>"
    • Lợi ích:               "<N> LỢI ÍCH CỦA <TOPIC>"
    • Catch-all:             "<N> ĐIỀU CẦN BIẾT VỀ <TOPIC>"
- intro_translation: SAME as intro_display (tiếng Việt only).
- intro_tts: SAME as intro_native (legacy field).
- intro_native: HOOK spoken trong tiếng Việt — urgency + actionable, **THU HÚT** ngay 2 giây đầu.
  Match hook variant với loại query:
    • Dấu hiệu:   "Lưu video này — <intro_display>" / "Bạn có dấu hiệu này không?"
    • Nguyên nhân: "<intro_display> — số 3 90% người Việt mắc"
    • Cách phòng:  "Lưu lại — <intro_display> ai cũng nên biết"
    • Cách chữa:   "<intro_display> không cần thuốc"
    • Mẹo:         "<intro_display> không tốn tiền"
    • Thực phẩm:   "<intro_display> — bạn ăn chưa?"
    • Lợi ích:     "<intro_display> — bất ngờ luôn"
  Under 65 chars.
- outro_native: CTA spoken in Vietnamese, ấm áp + disclaimer nhẹ.
  Variants (xen kẽ tránh lặp):
    - "Lưu lại và chia sẻ cho người thân nhé! Tham khảo bác sĩ để chẩn đoán chính xác."
    - "Theo dõi kênh để biết thêm thông tin sức khoẻ. Chúc bạn luôn khoẻ!"
    - "Đừng quên lưu lại — thông tin này có thể giúp bạn hoặc người thân. Chúc khoẻ!"
    - "Lưu video lại — chia sẻ cho ba mẹ và người thân yêu nhé!"
  Under 100 chars.
- topic_label: ngắn, lowercase, no period. Match loại query:
    • Dấu hiệu:    "về dấu hiệu <topic>"
    • Nguyên nhân: "về nguyên nhân <topic>"
    • Cách phòng:  "về cách phòng <topic>"
    • Mẹo:         "về mẹo <topic>"
    • Catch-all:   "về <topic>"
- short_title: KEYWORD ngắn in Title Case — CHỈ topic gọn (không có "dấu hiệu/nguyên nhân/cách phòng"). E.g. "Tiểu đường", "Đột quỵ", "Cảm cúm", "Tim mạch", "Mất ngủ". Under 24 chars.
- short_title_target: SAME as short_title.
- phrases: list of N items, each {target, pronunciation, ipa, native}:
    - target = TIÊU ĐỀ điểm chính in Vietnamese, brief noun phrase. Format theo loại:
        • Dấu hiệu: "Khát nước liên tục"
        • Nguyên nhân: "Ăn nhiều đồ ngọt"
        • Cách phòng: "Tập thể dục đều đặn"
        • Mẹo: "Uống đủ 2 lít nước"
        • Thực phẩm: "Cá hồi" / "Trà xanh"
      2-6 từ. Under 30 chars.
    - pronunciation = "" (EMPTY STRING — KHÔNG có)
    - ipa = "" (EMPTY STRING — KHÔNG có)
    - native = MÔ TẢ NGẮN/CHI TIẾT in Vietnamese (giải thích cụ thể). Under 70 chars.
      Examples:
        • Dấu hiệu: "Uống 3-4 lít/ngày vẫn khát do đường huyết cao"
        • Nguyên nhân: "Đường huyết tăng nhanh, tụy phải tiết nhiều insulin"
        • Cách phòng: "30 phút mỗi ngày — giảm 40% nguy cơ"
        • Thực phẩm: "Giàu Omega-3, giảm viêm và bảo vệ tim mạch"

═══ PART 3: CAPTION ═══
Multi-line tiếng Việt, structure:
  Line 1: 🌿 emoji + intro_display
  Blank line
  "📋 Dấu hiệu cần lưu ý:"
  3 lines: "1. <target1>: <native1>" / "2. <target2>: <native2>" / "3. <target3>: <native3>"
  Blank line
  "💾 Lưu video để chia sẻ với người thân"
  "⚕️ Thông tin tham khảo — vui lòng gặp bác sĩ để chẩn đoán chính xác"
  Blank line
  5-8 hashtags single line.

HASHTAG RULES (TUYỆT ĐỐI):
1. Mỗi hashtag PHẢI bắt đầu với "#"
2. Lowercase ASCII only — NO diacritics, NO hyphens, NO underscores
3. Mix Vietnamese transliterated + English
4. Example đúng: #suckhoe #dauhieu #benhtieuduong #healthtips #wellness #medical #vietnam
5. Example SAI: suckhoe (no #), #sức_khoẻ (diacritics + underscore), #health-tips (hyphen)

NỘI DUNG Y HỌC (CRITICAL):
- Dùng kiến thức dựa trên WHO, CDC, Mayo Clinic, Bộ Y Tế VN
- KHÔNG alarmist — informative + recommendations nhẹ nhàng
- Triệu chứng phải REALISTIC, được biết đến rộng rãi
- Mỗi target = 1 triệu chứng/dấu hiệu CỤ THỂ, KHÔNG vague
- Tránh đoán bệnh hiếm — focus vào bệnh phổ biến và dấu hiệu CỔ ĐIỂN
"""


COMBINED_SCHEMA = {
    "type": "OBJECT",
    "required": ["intent", "intro_display", "intro_translation", "intro_tts",
                 "intro_native", "outro_native",
                 "topic_label", "short_title", "short_title_target",
                 "scene_image_prompt", "phrases", "caption"],
    "properties": {
        "intent": {
            "type": "OBJECT",
            "required": ["target_lang", "native_lang", "topic", "count",
                         "voice_gender", "target_lang_name"],
            "properties": {
                "target_lang": {"type": "STRING"},
                "native_lang": {"type": "STRING"},
                "topic": {"type": "STRING"},
                "count": {"type": "INTEGER"},
                "voice_gender": {"type": "STRING"},
                "target_lang_name": {"type": "STRING"},
            },
        },
        "intro_display": {"type": "STRING"},
        "intro_translation": {"type": "STRING"},
        "intro_tts": {"type": "STRING"},
        "intro_native": {"type": "STRING"},
        "outro_native": {"type": "STRING"},
        "topic_label": {"type": "STRING"},
        "short_title": {"type": "STRING"},
        "short_title_target": {"type": "STRING"},
        "scene_image_prompt": {"type": "STRING"},
        "phrases": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["target", "pronunciation", "ipa", "native"],
                "properties": {
                    "target": {"type": "STRING"},
                    "pronunciation": {"type": "STRING"},
                    "ipa": {"type": "STRING"},
                    "native": {"type": "STRING"},
                },
            },
        },
        "caption": {"type": "STRING"},
    },
}


# ─────────────────────────────────────────────────────────────────────────
# QUIZ LAYOUT — language-only (engagement-bait, multi-choice + auto-comment)
# ─────────────────────────────────────────────────────────────────────────

LANGUAGE_QUIZ_SYSTEM_PROMPT = """You generate MULTIPLE-CHOICE QUIZ content for a short-form vertical (9:16) language-learning video. Output strict JSON, no prose.

═══ PART 1: INTENT (top-level "intent" field) ═══
- target_lang: ISO 639-1 of the language being LEARNED (e.g. "de", "ru", "ko")
- native_lang: ISO 639-1 the user wrote in (e.g. "vi", "en")
- topic: short phrase in user's native lang (e.g. "tình yêu", "ăn uống")
- count: ALWAYS 4 (exactly 4 options)
- voice_gender: "female" | "male" | "any" (default "any")
- target_lang_name: lang name in user's native lang (e.g. "Đức")
- layout_type: ALWAYS "quiz"

═══ PART 2: QUIZ CONTENT ═══
- question_native: question in user's native language. WRITE NATIVELY for that language — DO NOT translate a Vietnamese template literally. Under 60 chars. Examples per language (style only — vary the wording):
    * vi: "Trong tiếng Đức, 'Anh yêu em' là gì?" / "'Anh yêu em' trong tiếng Đức nói thế nào?"
    * en: "How do you say 'I love you' in German?" / "What's 'I love you' in German?" / "'I love you' in German — how?" (NEVER "In German, 'I love you' is what?" — translated VN)
    * ko: "독일어로 '사랑해'는 뭐예요?"
    * ja: "ドイツ語で「愛してる」は何と言いますか?"
    * es: "¿Cómo se dice 'te quiero' en alemán?"
- question_target: SAME question rendered NATIVELY in TARGET language. E.g. "Wie sagt man 'I love you' auf Deutsch?" / "Wie sagt man 'Cảm ơn' auf Deutsch?". Under 60 chars.
- options: EXACTLY 4 items {label, text, pronunciation, ipa}:
    - label: "A", "B", "C", "D" in order
    - text: option in TARGET language script (Cyrillic for ru, Hangul for ko, Hanzi for zh, native German "Ich liebe dich")
    - pronunciation: **RULES VARY BY TARGET LANGUAGE**:
        * **zh**: standard Pinyin with tone marks, NO hyphens between syllables, e.g. "Wǒ ài nǐ" (NOT "Wo-aì-nỉ")
        * **ja**: standard Romaji (Hepburn), no hyphens, e.g. "ohayō"
        * **ko**: standard Revised Romanization, e.g. "annyeong"
        * **Other (ru, de, fr, es, th, ...)**: Vietnamese-reader-friendly Latin transliteration with hyphens + stress accents (under 24 chars)
    - ipa: standard IPA in slashes, e.g. "/ɪç ˈliːbə dɪç/" (under 28 chars)
        * **For zh, ja, ko**: ipa = "" (EMPTY — Pinyin/Romaji is already the standard phonetic; redundant IPA clutters the card)
  CONSTRAINTS:
    - Exactly 1 option is CORRECT (the right translation)
    - 3 distractors must be COMMON, well-known words in the target lang (not nonsense)
    - Distractors are DIFFERENT enough that learners must think (not all greetings if topic is love)
    - Shuffle so correct answer is not always A
- correct_answer: which label is correct: "A" | "B" | "C" | "D"
- explanation: 1-2 sentence explanation in NATIVE language. Write natively for the user's language (not translated VN). E.g. vi: "Đáp án A. 'Ich liebe dich' = 'Anh yêu em'. 'Liebe' = tình yêu." | en: "Answer A. 'Ich liebe dich' means 'I love you' — 'Liebe' is the German word for love.". Under 120 chars. Used for the FB pinned comment.
- intro_native: spoken question in native language. Write NATIVELY in that language. SAME natural phrasing as `question_native` above — re-use or paraphrase. Under 50 chars.
- outro_native: spoken outro CTA in NATIVE language. Idiomatic short-video closer in that language — NOT translated VN. Under 60 chars. Examples per language (style only):
    * vi: "Bạn chọn đáp án nào? Comment ngay bên dưới nhé!" / "Đáp án của bạn là gì? Comment đi nào!"
    * en: "What's your answer? Drop it below!" / "Comment your guess below!" / "Which one did you pick?"
    * ko: "정답이 뭐예요? 댓글로 알려주세요!"
    * ja: "あなたの答えはどれ?コメントで教えて!"
    * es: "¿Cuál es tu respuesta? ¡Déjala en los comentarios!"
- topic_label: in native lang prefixed with "về" or "chủ đề" (e.g. "về tình yêu"). Lowercase.
- short_title: SHORT topic noun phrase in NATIVE lang. Just the topic itself — DO NOT append the target language name (e.g. "I love you" NOT "I love you in Russian"; "Ordering coffee" NOT "Ordering coffee in Russian"; "Cảm ơn" NOT "Cảm ơn tiếng Đức"). The surrounding template already shows "in <target_lang>" so adding it again is redundant. Title Case. Under 24 chars.
- scene_image_prompt: ENGLISH description of a background SCENE matching this topic, for AI image gen. NO people in foreground. Photographable place fitting the question's context. E.g.:
  * "Anh yêu em" → "a romantic Paris cafe with Eiffel Tower view, warm sunset light, illustration style"
  * "đặt món Aldi" → "interior of German Aldi supermarket aisle, soft lighting, illustration style"
  * "phỏng vấn" → "modern office meeting room with table chairs, professional, illustration style"
  Under 110 chars. Always end with "illustration style".
- short_title_target: TARGET-LANG rendition (e.g. "Ich liebe dich", "Danke"). Under 24 chars.

═══ PART 3: CAPTION (single "caption" string with real newlines) ═══
The caption language MUST match `native_lang` from PART 1. Pick the right structure.

▸ IF native_lang == "vi" — Vietnamese caption:
  Line 1: "<COUNTRY_FLAG_EMOJI> Tiếng <target_lang_name> giao tiếp"
          (E.g. "🇩🇪 Tiếng Đức giao tiếp" / "🇷🇺 Tiếng Nga giao tiếp" / "🇰🇷 Tiếng Hàn giao tiếp")
          ALWAYS use "giao tiếp" for SEO.
  blank line
  Line 3: emoji + question_native
  blank line
  "🎯 ĐÁP ÁN BẠN CHỌN?"
  "A) <opt A text>"
  "B) <opt B text>"
  "C) <opt C text>"
  "D) <opt D text>"
  blank line
  "💬 Bạn chọn đáp án nào? Comment ngay bên dưới!"
  blank line
  5-8 hashtags single space-separated line.

▸ IF native_lang == "en" — English caption (natively idiomatic, not translated VN):
  Line 1: "<COUNTRY_FLAG_EMOJI> <target_lang_name> for everyday conversation"
          (E.g. "🇷🇺 Russian for everyday conversation" / "🇯🇵 Japanese for everyday conversation")
  blank line
  Line 3: emoji + question_native
  blank line
  "🎯 PICK YOUR ANSWER:"
  "A) <opt A text>"
  "B) <opt B text>"
  "C) <opt C text>"
  "D) <opt D text>"
  blank line
  "💬 Drop your answer in the comments!"
  blank line
  5-8 hashtags (transliterated target + English mix).

▸ For OTHER native_lang (ko / ja / es / …) — same structure, all native-language strings IDIOMATIC for that language. Translate the section headers natively, never copy Vietnamese verbatim.

CRITICAL: DO NOT reveal the correct answer anywhere in the caption — no "đáp án đúng là…" / "answer below" / "the right one is …" hints. Pure suspense.

HASHTAG RULES (TUYỆT ĐỐI):
1. Mỗi hashtag bắt đầu với "#"
2. Lowercase ASCII only — NO diacritics, NO hyphens, NO underscores
3. Mix transliterated + English
4. vi example: #tiengduc #quiz #hocngoaingu #germanquiz #language
5. en example: #russian #russianquiz #learnrussian #language #polyglot #studygram
"""


# ──────────────────────────────────────────────────────────────────────
# REVERSE QUIZ (target → native) — shows a target-lang phrase + 4 VN options
# ──────────────────────────────────────────────────────────────────────

LANGUAGE_QUIZ_REVERSE_SYSTEM_PROMPT = """You generate REVERSE MULTIPLE-CHOICE QUIZ content for a short-form vertical (9:16) language-learning video.

Reverse quiz = the question shows a TARGET-LANGUAGE PHRASE, and the 4 options are translations in the user's NATIVE language (the language they speak). User has to guess which option is the correct meaning.

The user message gives a CATEGORY (e.g. "công sở & sếp" / "office & boss"). You MUST:
  1. Pick ONE common, useful phrase in the target language that fits this category.
  2. Generate 4 native-language meaning options where exactly ONE is the correct translation.

Output strict JSON, no prose.

═══ PART 1: INTENT (top-level "intent" field) ═══
- target_lang: ISO 639-1 of the source-side language (the phrase shown to viewers, e.g. "ru", "de", "ja")
- native_lang: ISO 639-1 of the USER's spoken language — INFERRED FROM THE USER MESSAGE LANGUAGE. If the user wrote in English (e.g. "Reverse quiz on the topic 'travel'…") then native_lang="en". If they wrote in Vietnamese (e.g. "Reverse quiz về chủ đề 'thuê nhà'…") then native_lang="vi". Other languages follow the same rule.
- topic: the category as given by the user, IN THEIR NATIVE LANGUAGE
- count: ALWAYS 4
- voice_gender: "female" | "male" | "any" (default "any")
- target_lang_name: lang name in the user's NATIVE language (e.g. for native=vi: "Đức"; for native=en: "German" / "Russian"; for native=ko: "독일어")
- layout_type: ALWAYS "quiz_reverse"

═══ PART 2: REVERSE QUIZ CONTENT ═══
All native-language fields MUST be in the user's NATIVE language (NOT hardcoded Vietnamese).

- question_target: the chosen target-language phrase, IN NATIVE SCRIPT (Cyrillic for ru, Hangul for ko, Hanzi for zh, e.g. "Счёт, пожалуйста", "请假", "アルバイト"). This is what viewers SEE big on screen. Under 30 chars. Pick a USEFUL, common phrase — not obscure.
- question_native: question in the USER's NATIVE language wrapping the target phrase. Idiomatic native phrasing. Under 60 chars.
    * vi: "Trong tiếng Việt, '<question_target>' nghĩa là gì?"
    * en: "What does '<question_target>' mean?" / "'<question_target>' means what in English?"
    * ko: "'<question_target>'는 한국어로 무슨 뜻이에요?"
    * ja: "「<question_target>」は日本語で何という意味?"
- options: EXACTLY 4 items {label, text, pronunciation, ipa}:
    * label: "A", "B", "C", "D" in order
    * text: translation in the user's NATIVE language (for native=vi: "Cho tôi thực đơn"; for native=en: "The bill, please"; etc). Under 30 chars.
    * pronunciation: ALWAYS "" (empty string — no pronunciation needed for native-lang options)
    * ipa: ALWAYS "" (empty)
  CONSTRAINTS:
    - Exactly 1 option is the CORRECT meaning
    - 3 distractors must be COMMON phrases in the native lang related to the same topic domain
    - Distractors are CLOSE but distinct meanings
    - Shuffle so correct answer is NOT always A
- correct_answer: "A" | "B" | "C" | "D"
- explanation: 1-2 sentence explanation IN NATIVE LANG. E.g. en: "Answer A. 'Счёт, пожалуйста' literally = 'bill, please' — what you say to ask for the check.". Under 140 chars.
- intro_native: short hook in NATIVE language (do NOT include the target phrase — narrator says only this, then a SEPARATE target voice reads question_target right after). Under 55 chars. Energetic.
    * vi: "Cụm tiếng <target_lang_name> này nghĩa gì? Đoán xem!"
    * en: "What does this <target_lang_name> phrase mean? Take a guess!"
    * ko: "이 <target_lang_name> 표현, 무슨 뜻일까요?"
- outro_native: native-language CTA, idiomatic short-video closer. Under 60 chars.
    * vi: "Bạn chọn đáp án nào? Comment ngay bên dưới!"
    * en: "What's your guess? Drop your answer below!"
    * ko: "정답이 뭐예요? 댓글로 알려주세요!"
- topic_label: in NATIVE language with native-language prefix (vi: "về", en: "about", ko: "에 관한"). Lowercase.
- short_title: short NATIVE-LANG noun phrase from the category. Under 24 chars. Title Case.
- scene_image_prompt: ENGLISH description of a background SCENE matching the category. NO people in foreground. Photographable place. E.g.:
  * "office" → "modern office interior with desks and computers, illustration style"
  * "travel" → "airport departure hall with flight info board, illustration style"
  Under 110 chars. Always end with "illustration style".
- short_title_target: SAME as short_title (we keep short_title visible only).

═══ PART 3: CAPTION ═══
The caption language MUST match `native_lang` from PART 1.

▸ IF native_lang == "vi" — Vietnamese caption:
  Line 1: "<COUNTRY_FLAG> Tiếng <target_lang_name> giao tiếp"
  blank line
  Line 3: "🤔 '<question_target>' tiếng Việt nghĩa là gì?"
  blank line
  "🎯 ĐÁP ÁN BẠN CHỌN?"
  "A) <opt A text VN>"  …  "D) <opt D text VN>"
  blank line
  "💬 Bạn chọn đáp án nào? Comment ngay bên dưới!"
  blank line
  5-8 hashtags single line.

▸ IF native_lang == "en" — English caption:
  Line 1: "<COUNTRY_FLAG> <target_lang_name> for everyday conversation"
  blank line
  Line 3: "🤔 What does '<question_target>' mean in English?"
  blank line
  "🎯 PICK YOUR ANSWER:"
  "A) <opt A text EN>"  …  "D) <opt D text EN>"
  blank line
  "💬 Drop your guess in the comments!"
  blank line
  5-8 hashtags single line (transliterated target + English mix).

▸ OTHER native_lang (ko / ja / es / …) — same structure, all native strings IDIOMATIC for that language.

CRITICAL: DO NOT reveal the correct answer anywhere in the caption. Pure suspense.

HASHTAG RULES (TUYỆT ĐỐI):
1. Mỗi hashtag bắt đầu với "#"
2. Lowercase ASCII only — NO diacritics, NO hyphens, NO underscores
3. Mix transliterated + English
4. vi: #tiengduc #quiz #hocngoaingu #vocab #dichthuat #language
5. en: #russian #russianquiz #learnrussian #language #polyglot
"""


# Same schema shape as COMBINED_SCHEMA but quiz fields. We reuse the loader logic.
QUIZ_SCHEMA = {
    "type": "OBJECT",
    "required": ["intent", "question_native", "question_target", "options",
                 "correct_answer", "explanation", "intro_native", "outro_native",
                 "topic_label", "short_title", "short_title_target",
                 "scene_image_prompt", "caption"],
    "properties": {
        "intent": {
            "type": "OBJECT",
            "required": ["target_lang", "native_lang", "topic", "count",
                         "voice_gender", "target_lang_name"],
            "properties": {
                "target_lang": {"type": "STRING"},
                "native_lang": {"type": "STRING"},
                "topic": {"type": "STRING"},
                "count": {"type": "INTEGER"},
                "voice_gender": {"type": "STRING"},
                "target_lang_name": {"type": "STRING"},
            },
        },
        "question_native": {"type": "STRING"},
        "question_target": {"type": "STRING"},
        "options": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["label", "text", "pronunciation", "ipa"],
                "properties": {
                    "label": {"type": "STRING"},
                    "text": {"type": "STRING"},
                    "pronunciation": {"type": "STRING"},
                    "ipa": {"type": "STRING"},
                },
            },
        },
        "correct_answer": {"type": "STRING"},
        "explanation": {"type": "STRING"},
        "intro_native": {"type": "STRING"},
        "outro_native": {"type": "STRING"},
        "topic_label": {"type": "STRING"},
        "short_title": {"type": "STRING"},
        "short_title_target": {"type": "STRING"},
        "scene_image_prompt": {"type": "STRING"},
        "caption": {"type": "STRING"},
    },
}


# Layout detection from user text — keyword-based, fast, no extra LLM call.
LAYOUT_KEYWORDS = {
    "quiz": ["quiz", "câu hỏi", "đáp án", "trắc nghiệm", "fill in", "test tiếng"],
}


def detect_layout(text: str, default: str = "phrases") -> str:
    """Pick a layout based on user text keywords."""
    t = (text or "").lower()
    for layout, keywords in LAYOUT_KEYWORDS.items():
        if any(k in t for k in keywords):
            return layout
    return default


def parse_and_generate_quiz(
    user_text: str,
    *,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> tuple[ParsedIntent, QuizContent]:
    """Generate a multi-choice quiz (4 options) for the language niche."""
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    resp = _call_gemini(client,
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=LANGUAGE_QUIZ_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=QUIZ_SCHEMA,
            temperature=0.7,
        ),
    )
    data = json.loads(resp.text)

    intent_data = data["intent"]
    intent = ParsedIntent(
        target_lang=intent_data["target_lang"].lower(),
        native_lang=intent_data["native_lang"].lower(),
        topic=intent_data["topic"],
        count=4,
        voice_gender=(intent_data.get("voice_gender") or "any").lower(),
        target_lang_name=intent_data["target_lang_name"],
        layout_type="quiz",
    )
    options = [QuizOption(**o) for o in data["options"]]
    if len(options) != 4:
        raise ValueError(f"Quiz must have exactly 4 options, got {len(options)}")
    if data["correct_answer"] not in {"A", "B", "C", "D"}:
        raise ValueError(f"correct_answer must be A/B/C/D, got {data['correct_answer']!r}")

    caption = _sanitize_hashtags(data["caption"])
    caption = _ensure_seo_hashtag(caption, intent.target_lang)
    content = QuizContent(
        question_native=data["question_native"],
        question_target=data["question_target"],
        options=options,
        correct_answer=data["correct_answer"],
        explanation=data["explanation"],
        intro_native=data["intro_native"],
        outro_native=data["outro_native"],
        topic_label=data["topic_label"],
        short_title=data["short_title"],
        short_title_target=data["short_title_target"],
        caption=caption,
        scene_image_prompt=data.get("scene_image_prompt", ""),
    )
    return intent, content


def parse_and_generate_quiz_reverse(
    user_text: str,
    *,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> tuple[ParsedIntent, QuizContent]:
    """Generate a REVERSE quiz (target phrase shown → 4 VN options).

    Returns the same QuizContent shape as forward quiz BUT semantics:
      - question_target = target-language phrase (e.g. "Arbeitserlaubnis")
      - question_native = wrapper VN sentence including question_target
      - options[].text = Vietnamese translations
      - options[].pronunciation / .ipa = "" (empty)

    The auto-post pipeline detects this by inspecting intent.layout_type == "quiz_reverse"
    OR by passing direction explicitly. Voice routing differs from forward:
      - intro_target audio uses TARGET voice (reads question_target alone)
      - opt_X audio uses NATIVE voice (reads VN translation)
    """
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    resp = _call_gemini(client,
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=LANGUAGE_QUIZ_REVERSE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=QUIZ_SCHEMA,
            temperature=0.75,
        ),
    )
    data = json.loads(resp.text)

    intent_data = data["intent"]
    intent = ParsedIntent(
        target_lang=intent_data["target_lang"].lower(),
        native_lang=intent_data.get("native_lang", "vi").lower(),
        topic=intent_data["topic"],
        count=4,
        voice_gender=(intent_data.get("voice_gender") or "any").lower(),
        target_lang_name=intent_data["target_lang_name"],
        layout_type="quiz_reverse",
    )
    options = [QuizOption(**o) for o in data["options"]]
    if len(options) != 4:
        raise ValueError(f"Reverse quiz must have exactly 4 options, got {len(options)}")
    if data["correct_answer"] not in {"A", "B", "C", "D"}:
        raise ValueError(f"correct_answer must be A/B/C/D, got {data['correct_answer']!r}")

    caption = _sanitize_hashtags(data["caption"])
    caption = _ensure_seo_hashtag(caption, intent.target_lang)
    content = QuizContent(
        question_native=data["question_native"],
        question_target=data["question_target"],
        options=options,
        correct_answer=data["correct_answer"],
        explanation=data["explanation"],
        intro_native=data["intro_native"],
        outro_native=data["outro_native"],
        topic_label=data["topic_label"],
        short_title=data["short_title"],
        short_title_target=data["short_title_target"],
        caption=caption,
        scene_image_prompt=data.get("scene_image_prompt", ""),
    )
    return intent, content


# ═══════════════════════════════════════════════════════════════════════
#  WHATS_THIS layout — visual vocab "Đây là gì?" with AI image gen
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WhatsThisItem:
    """Single round in the visual vocab video."""
    target_word: str        # target language word: "Trinken" / "吃饭" / "먹다"
    pronunciation: str      # phonetic per language standard
    ipa: str                # IPA (empty for zh/ja/ko)
    native_answer: str      # Vietnamese: "Uống" / "Ăn"
    image_prompt: str       # English subject for FLUX: "a person drinking water from a glass"
    voice_question: str     # Question in target lang: "Was ist das?" / "他在做什么?"
    voice_reveal: str       # Just the word, read alone (often == target_word)
    item_kind: str          # "noun" | "verb" | "adj"


@dataclass
class WhatsThisContent:
    intro_display: str          # title card text (no longer rendered — kept for caption fallback)
    intro_native: str           # spoken intro (no longer used — intro removed)
    outro_native: str           # spoken outro CTA in native lang
    topic_label: str            # short label native ("hành động hàng ngày")
    short_title: str            # rút gọn cho header line ("Hành động hàng ngày")
    short_title_target: str     # rút gọn dịch target lang ("Alltagshandlungen")
    display_question: str       # theme-aware on-screen question, e.g. "Đây là nghề gì?", "Đây là gì?"
    items: list[WhatsThisItem]  # exactly 10
    caption: str                # FB caption with all 10 + pronunciation


LANGUAGE_WHATS_THIS_SYSTEM_PROMPT = """You generate a "What's this?" visual vocabulary lesson for a short-form language-learning video. The video shows 10 AI-generated cartoon illustrations one by one. For each item the learner hears a question in the target language, sees the picture, then sees the answer in the target language + pronunciation + Vietnamese meaning.

Return ONE strict JSON object. NO prose outside JSON.

═══ PART 1: INTENT (top-level "intent" field) ═══
- target_lang: ISO 639-1 (e.g. "de", "zh", "ko")
- native_lang: ISO 639-1 (always "vi" for these channels)
- topic: theme in native language ("hành động hàng ngày", "đồ ăn Hàn Quốc"...)
- count: ALWAYS 10
- voice_gender: "any"
- target_lang_name: ("Đức" / "Trung" / "Hàn")

═══ PART 2: HEADERS ═══
- intro_display: SHORT title for caption only (not rendered on video). Start with "10 ".
  Examples: "10 hành động hàng ngày bằng tiếng Đức" / "10 món Hàn ai cũng nên biết" / "10 đồ vật cty Trung văn phòng". Under 42 chars.
- intro_native: short Vietnamese title line (legacy, not used at runtime — keep for compatibility). Under 60 chars.
- outro_native: SPOKEN outro CTA in NATIVE lang, encouraging follow + practice.
  vi: "Hãy like và follow để học mỗi ngày nhé! Chúc bạn thành công!" or variants.
  Under 90 chars.
- topic_label: topic in native lang prefixed with "về" or "chủ đề" ("về hành động hàng ngày"). Lowercase.
- short_title: just the topic keyword as noun phrase NATIVE lang. Title-case. Under 28 chars.
- short_title_target: same noun phrase in TARGET language. Under 28 chars.
- display_question: SHORT theme-aware question SHOWN ON SCREEN. Vietnamese. Must MATCH the theme noun. Pick the most natural phrasing:
  * Theme nghề/professions → "Đây là nghề gì?"
  * Theme đồ ăn/món/thức ăn → "Đây là món gì?"
  * Theme hành động/động từ/verb → "Đây là hành động gì?"
  * Theme cảm xúc/trạng thái → "Đây là cảm xúc gì?"
  * Theme phương tiện/đi lại → "Đây là phương tiện gì?"
  * Theme cơ thể/bộ phận → "Đây là bộ phận gì?"
  * Theme thời tiết/mùa → "Đây là thời tiết gì?"
  * Theme đồ vật/nhà cửa/văn phòng → "Đây là gì?"
  * Theme lễ hội → "Đây là lễ hội gì?"
  * Default / mixed / unknown → "Đây là gì?"
  Under 26 chars. Match the singular noun naturally.

═══ PART 3: 10 ITEMS (the "items" array) ═══
Pick 10 CONCRETE, COMMON, EVERYDAY items fitting the theme. Each item is a SINGLE word (noun, verb, or adjective). Prefer items with very visual / iconic representations.

For each item:
- target_word: SINGLE word in target language native script. Lemma form for verbs (e.g. de: "trinken" not "trinkt"; zh: "吃" or "吃饭"; ko: "먹다"). Capitalize properly for German nouns ("Apfel", "Auto"). Keep it short (1-3 syllables ideal). Under 18 chars.
- pronunciation: phonetic per LANGUAGE STANDARD:
  * **zh**: standard PINYIN with tone marks (NO hyphens between syllables of a single word). E.g. "chī fàn" / "wǒ ài nǐ".
  * **ja**: standard ROMAJI (Hepburn). E.g. "tabemasu" / "neko".
  * **ko**: Revised Romanization. E.g. "meokda" / "gae".
  * **de/fr/es/ru/...**: Vietnamese-reader-friendly transliteration with hyphens between syllables + acute on stressed syllable. E.g. de "trinken" → "trinh-cờn"; "Apfel" → "áp-fờl".
  Under 24 chars.
- ipa: IPA in slashes (e.g. de "/ˈapfəl/"). **For zh, ja, ko: ipa = ""** (empty).
- native_answer: idiomatic Vietnamese translation of the target word. SINGLE word or short phrase, like the target. Under 18 chars. Match the kind: verb→verb, noun→noun.
- image_prompt: English description for AI image generator (FLUX). Be CONCRETE and ICONIC. Format: "<subject> <action/state> <minimal context>". Keep under 100 chars. Examples:
  * "Apfel" → "a single shiny red apple on a wooden table"
  * "trinken" → "a person drinking water from a clear glass"
  * "吃饭" → "a person eating rice with chopsticks from a bowl"
  * "먹다" → "a person eating bibimbap from a stone bowl with spoon"
  Avoid abstract concepts. Be photographable.
- voice_question: SHORT question in TARGET language asking "what is this?" or "what is the action?".
  * **For noun items** use "what is this?" form:
    * de: "Was ist das?"
    * zh: "这是什么?"
    * ko: "이게 뭐예요?"
  * **For verb items** use "what is he/she doing?" form:
    * de: "Was macht er?" / "Was passiert?"
    * zh: "他在做什么?"
    * ko: "뭐 하고 있어요?"
  * **For adjective items** use "how is he/she?" form:
    * de: "Wie ist er?" / "Wie fühlt er sich?"
    * zh: "他怎么样?"
    * ko: "어떤 기분이에요?"
- voice_reveal: just the target_word read alone (usually identical to target_word; for German nouns include the article: "der Apfel" / "die Katze" / "das Auto"). Under 24 chars.
- item_kind: "noun" | "verb" | "adj".

═══ PART 4: CAPTION (single "caption" string with REAL newlines) ═══
Structure:
  Line 1: "<FLAG_EMOJI> Tiếng <target_lang_name> mỗi ngày"
  blank
  Line 3: emoji + intro_display
  blank
  "📚 10 từ vựng hôm nay:"
  10 lines: "1. <target_word> — <pronunciation> — <native_answer>" ... "10. ..."
  blank
  "👉 Lưu lại học mỗi ngày!"
  "📌 Follow để không bỏ lỡ video mới."
  blank
  "#tieng<targetname> #hocngoaingu #tuvung"
Country flags: 🇩🇪 de · 🇨🇳 zh · 🇰🇷 ko · 🇯🇵 ja.
Under 1200 chars total.

═══ RULES ═══
- 10 items, exactly. Diverse subjects within the theme — don't repeat similar items.
- target_word in NATIVE SCRIPT (Hanzi/Hangul/etc.), never Latinized.
- pronunciation per the LANGUAGE STANDARD above. NO mixing standards.
- image_prompt in ENGLISH, no commas listing many things — keep it ONE clear subject.
- All items must be appropriate for general audience (no NSFW, no controversial).
"""


WHATS_THIS_SCHEMA = {
    "type": "OBJECT",
    "required": ["intent", "intro_display", "intro_native", "outro_native",
                 "topic_label", "short_title", "short_title_target",
                 "display_question", "items", "caption"],
    "properties": {
        "intent": {
            "type": "OBJECT",
            "required": ["target_lang", "native_lang", "topic", "count",
                         "voice_gender", "target_lang_name"],
            "properties": {
                "target_lang":      {"type": "STRING"},
                "native_lang":      {"type": "STRING"},
                "topic":            {"type": "STRING"},
                "count":            {"type": "INTEGER"},
                "voice_gender":     {"type": "STRING"},
                "target_lang_name": {"type": "STRING"},
            },
        },
        "intro_display":       {"type": "STRING"},
        "intro_native":        {"type": "STRING"},
        "outro_native":        {"type": "STRING"},
        "topic_label":         {"type": "STRING"},
        "short_title":         {"type": "STRING"},
        "short_title_target":  {"type": "STRING"},
        "display_question":    {"type": "STRING"},
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["target_word", "pronunciation", "ipa",
                             "native_answer", "image_prompt",
                             "voice_question", "voice_reveal", "item_kind"],
                "properties": {
                    "target_word":     {"type": "STRING"},
                    "pronunciation":   {"type": "STRING"},
                    "ipa":             {"type": "STRING"},
                    "native_answer":   {"type": "STRING"},
                    "image_prompt":    {"type": "STRING"},
                    "voice_question":  {"type": "STRING"},
                    "voice_reveal":    {"type": "STRING"},
                    "item_kind":       {"type": "STRING"},
                },
            },
        },
        "caption":             {"type": "STRING"},
    },
}


def parse_and_generate_whats_this(
    user_text: str,
    *,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> tuple[ParsedIntent, WhatsThisContent]:
    """Generate a "What's this?" visual vocab video content from theme request."""
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    resp = _call_gemini(client,
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=LANGUAGE_WHATS_THIS_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=WHATS_THIS_SCHEMA,
            temperature=0.85,  # slightly higher for diverse item picks
        ),
    )
    data = json.loads(resp.text)

    intent_data = data["intent"]
    intent = ParsedIntent(
        target_lang=intent_data["target_lang"].lower(),
        native_lang=intent_data.get("native_lang", "vi").lower(),
        topic=intent_data["topic"],
        count=10,
        voice_gender=(intent_data.get("voice_gender") or "any").lower(),
        target_lang_name=intent_data["target_lang_name"],
        layout_type="whats_this",
    )
    items = [WhatsThisItem(**it) for it in data["items"]]
    if len(items) != 10:
        raise ValueError(f"whats_this needs exactly 10 items, got {len(items)}")

    caption = _sanitize_hashtags(data["caption"])
    caption = _ensure_seo_hashtag(caption, intent.target_lang)
    content = WhatsThisContent(
        intro_display=data["intro_display"],
        intro_native=data["intro_native"],
        outro_native=data["outro_native"],
        topic_label=data["topic_label"],
        short_title=data["short_title"],
        short_title_target=data["short_title_target"],
        display_question=data["display_question"],
        items=items,
        caption=caption,
    )
    return intent, content


# ═══════════════════════════════════════════════════════════════════════
#  WHATS_BOARD layout — 9-grid cheat sheet vocab with AI image gen
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WhatsBoardItem:
    """One cell in the 9-grid board."""
    target_word: str        # target language noun: "고양이" / "猫" / "die Katze"
    pronunciation: str      # phonetic per language standard
    ipa: str                # IPA (empty for zh/ja/ko)
    native_answer: str      # Vietnamese: "Mèo"
    image_prompt: str       # English subject for FLUX: "a cute cartoon orange tabby cat sitting"
    voice_repeat: str       # Target word read ONCE (field name kept for back-compat): "고양이"


@dataclass
class WhatsBoardContent:
    title_native: str            # "ĐỘNG VẬT TRONG TIẾNG HÀN" — top header VN (UPPERCASE)
    title_target: str            # "동물" — title in target script, shown below VN
    outro_native: str            # spoken outro CTA in native lang
    topic_label: str             # short label native ("động vật")
    short_title: str             # short topic name native ("Động vật")
    short_title_target: str      # short topic name target ("동물")
    items: list[WhatsBoardItem]  # exactly 9
    caption: str                 # FB caption with all 9 + pronunciation


LANGUAGE_WHATS_BOARD_SYSTEM_PROMPT = """You generate a 9-grid VISUAL VOCABULARY CHEAT SHEET for a short-form language-learning video. The video shows ALL 9 AI-generated illustrations at once in a 3×3 grid. A voice reads each target word ONCE while highlighting that cell. Format inspired by canthoground.com Korean-vocab grids.

Return ONE strict JSON object. NO prose outside JSON.

═══ PART 1: INTENT ═══
- target_lang: ISO 639-1 (e.g. "de", "zh", "ko")
- native_lang: ISO 639-1 INFERRED from user message language ("vi" if user wrote Vietnamese, "en" if English, etc.). ALL native-language fields below MUST be written IDIOMATICALLY in this native_lang — never translate Vietnamese examples verbatim. for these channels
- topic: theme in native language ("động vật", "trái cây"...)
- count: ALWAYS 9
- voice_gender: "any"
- target_lang_name: ("Đức" / "Trung" / "Hàn")

═══ PART 2: HEADERS ═══
- title_native: TOP header text in NATIVE language. UPPERCASE-ready (Gemini writes in title-case; CSS will UPPERCASE it). Format: "<THEME> TRONG TIẾNG <LANG>". E.g. "Động vật trong tiếng Hàn" / "Trái cây trong tiếng Đức" / "Đồ uống trong tiếng Trung". Under 36 chars.
- title_target: SHORT theme in TARGET language native script. E.g. ko: "동물" / zh: "动物" / de: "Tiere". Under 12 chars (just the bare noun).
- outro_native: SPOKEN outro CTA in NATIVE lang. vi: "Hãy like và follow để học mỗi ngày nhé! Chúc bạn thành công!". Under 90 chars.
- topic_label: topic in native lang prefixed with "về" ("về động vật"). Lowercase.
- short_title: topic noun phrase native lang. Title-case. Under 24 chars.
- short_title_target: same noun in target lang. Under 16 chars.

═══ PART 3: 9 ITEMS ═══
Pick 9 CONCRETE, HIGHLY IMAGEABLE nouns from the theme. Skip anything ambiguous to draw (emotions, abstract concepts). Prefer items kids learn first.

For each item:
- target_word: SINGLE noun in target language native script. For German: include article ("die Katze" / "der Hund" / "das Auto"). Under 18 chars.
- pronunciation: phonetic per LANGUAGE STANDARD:
  * **zh**: standard PINYIN with tone marks (NO hyphens between syllables of a single word). E.g. "māo" / "píngguǒ".
  * **ja**: Hepburn romaji. E.g. "neko" / "ringo".
  * **ko**: Revised Romanization. E.g. "go-yang-i" / "sa-gwa".
  * **de/fr/es/ru**: Vietnamese-reader-friendly with hyphens + acute on stressed syllable. E.g. de "die Katze" → "đi ka-tsơ".
  Under 22 chars.
- ipa: IPA in slashes. **For zh, ja, ko: ipa = ""** (empty).
- native_answer: Vietnamese translation, SINGLE noun. Title-case. Under 16 chars. E.g. "Mèo" / "Quả táo" / "Ô tô".
- image_prompt: English description for AI image generator (FLUX). Be CONCRETE and ICONIC for a flat-cartoon kawaii style:
  * "a cute cartoon orange tabby cat sitting"
  * "a shiny red apple on a wooden surface"
  * "a yellow school bus parked"
  * "a wooden chair viewed from front"
  Avoid: people in suggestive poses, body parts, weapons, anything ambiguous. Single subject, plain background. Under 90 chars.
- voice_repeat: target_word read ONCE in the target's native script (Edge TTS handles CJK/Cyrillic/etc.). E.g.:
  * ko: "고양이"
  * zh: "猫"
  * ja: "猫"
  * de: "die Katze"
  * ru: "кошка"
  Format: exactly "<target_word>" — single occurrence, NO repetition, NO trailing period.
  CRITICAL: do NOT repeat the word. Reading it once keeps voice in sync with the cell highlight transition. Under 22 chars.

═══ PART 4: CAPTION ═══
Structure:
  Line 1: "<FLAG> Tiếng <target_lang_name> mỗi ngày"
  blank
  Line 3: emoji + title_native
  blank
  "📚 9 từ vựng hôm nay:"
  9 lines: "1. <target_word> — <pronunciation> — <native_answer>" ... "9. ..."
  blank
  "👉 Lưu lại học mỗi ngày!"
  "📌 Follow để không bỏ lỡ video mới."
  blank
  "#tieng<targetname> #hocngoaingu #tuvung"

═══ RULES ═══
- EXACTLY 9 items. Diverse subjects within the theme.
- target_word in NATIVE SCRIPT (Hangul/Hanzi/etc.), never just Latinized.
- pronunciation per the LANGUAGE STANDARD above. NO mixing.
- image_prompt in ENGLISH, single concrete subject.
- voice_repeat = target_word read ONCE in native script. NEVER repeat the word.
- ALL items kid-friendly, draw-able by cartoon AI.
"""


WHATS_BOARD_SCHEMA = {
    "type": "OBJECT",
    "required": ["intent", "title_native", "title_target", "outro_native",
                 "topic_label", "short_title", "short_title_target",
                 "items", "caption"],
    "properties": {
        "intent": {
            "type": "OBJECT",
            "required": ["target_lang", "native_lang", "topic", "count",
                         "voice_gender", "target_lang_name"],
            "properties": {
                "target_lang":      {"type": "STRING"},
                "native_lang":      {"type": "STRING"},
                "topic":            {"type": "STRING"},
                "count":            {"type": "INTEGER"},
                "voice_gender":     {"type": "STRING"},
                "target_lang_name": {"type": "STRING"},
            },
        },
        "title_native":        {"type": "STRING"},
        "title_target":        {"type": "STRING"},
        "outro_native":        {"type": "STRING"},
        "topic_label":         {"type": "STRING"},
        "short_title":         {"type": "STRING"},
        "short_title_target":  {"type": "STRING"},
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["target_word", "pronunciation", "ipa",
                             "native_answer", "image_prompt", "voice_repeat"],
                "properties": {
                    "target_word":   {"type": "STRING"},
                    "pronunciation": {"type": "STRING"},
                    "ipa":           {"type": "STRING"},
                    "native_answer": {"type": "STRING"},
                    "image_prompt":  {"type": "STRING"},
                    "voice_repeat":  {"type": "STRING"},
                },
            },
        },
        "caption":             {"type": "STRING"},
    },
}


def parse_and_generate_whats_board(
    user_text: str,
    *,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> tuple[ParsedIntent, WhatsBoardContent]:
    """Generate a 9-grid cheat-sheet vocab video content from theme request."""
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    resp = _call_gemini(client,
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=LANGUAGE_WHATS_BOARD_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=WHATS_BOARD_SCHEMA,
            temperature=0.85,
        ),
    )
    data = json.loads(resp.text)

    intent_data = data["intent"]
    intent = ParsedIntent(
        target_lang=intent_data["target_lang"].lower(),
        native_lang=intent_data.get("native_lang", "vi").lower(),
        topic=intent_data["topic"],
        count=9,
        voice_gender=(intent_data.get("voice_gender") or "any").lower(),
        target_lang_name=intent_data["target_lang_name"],
        layout_type="whats_board",
    )
    items = [WhatsBoardItem(**it) for it in data["items"]]
    if len(items) != 9:
        raise ValueError(f"whats_board needs exactly 9 items, got {len(items)}")

    caption = _sanitize_hashtags(data["caption"])
    caption = _ensure_seo_hashtag(caption, intent.target_lang)
    content = WhatsBoardContent(
        title_native=data["title_native"],
        title_target=data["title_target"],
        outro_native=data["outro_native"],
        topic_label=data["topic_label"],
        short_title=data["short_title"],
        short_title_target=data["short_title_target"],
        items=items,
        caption=caption,
    )
    return intent, content


# ═══════════════════════════════════════════════════════════════════════
#  DIALOGUE layout — 2-character mini skit with speech bubble + sub bar
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DialogueCharacter:
    """One character in the dialogue. Voice + portrait."""
    label: str          # "A" or "B"
    name: str           # display name in native lang ("Chị Anna" / "Anh Hùng")
    role: str           # role in scene ("phục vụ" / "khách hàng")
    voice_gender: str   # "female" or "male"
    image_prompt: str   # English: "a friendly young waiter in apron, cartoon portrait"


@dataclass
class DialogueTurn:
    """One turn of the dialogue."""
    speaker: str            # "A" or "B"
    target: str             # target-language sentence
    pronunciation: str      # phonetic per language standard
    ipa: str                # IPA (empty for zh/ja/ko)
    native: str             # Vietnamese translation


@dataclass
class DialogueContent:
    scenario: str               # native: "đặt món tại nhà hàng Đức"
    title_native: str           # UPPERCASE-ready: "Đặt món tại nhà hàng"
    title_target: str           # short target: "Im Restaurant"
    char_a: DialogueCharacter
    char_b: DialogueCharacter
    scene_image_prompt: str     # English background prompt
    turns: list[DialogueTurn]   # 6-8 turns
    outro_native: str           # CTA outro
    topic_label: str
    short_title: str
    short_title_target: str
    caption: str                # FB caption


LANGUAGE_DIALOGUE_SYSTEM_PROMPT = """You generate a 2-CHARACTER MINI DIALOGUE for a short-form language-learning video. The video shows: a background scene, 2 character portraits at bottom corners, a speech bubble pops near the active speaker per turn, and a sub bar at the bottom shows the target text + phonetic + Vietnamese translation.

Return ONE strict JSON object. NO prose outside JSON.

═══ PART 1: INTENT ═══
- target_lang: ISO 639-1 (e.g. "de", "zh", "ko")
- native_lang: ISO 639-1 INFERRED from user message language ("vi" if user wrote Vietnamese, "en" if English, etc.). ALL native-language fields below MUST be written IDIOMATICALLY in this native_lang — never translate Vietnamese examples verbatim.
- topic: scenario in native language ("đặt món tại nhà hàng Đức")
- count: number of turns (6, 7, or 8)
- voice_gender: "any"
- target_lang_name: ("Đức" / "Trung" / "Hàn")

═══ PART 2: SCENARIO & TITLES ═══

CRITICAL CONSISTENCY RULE — the title fields and the FIRST TURN of the dialogue MUST match. If the dialogue opens with "Where is the metro?" then the title cannot advertise "Where is the street?" — the viewer reads the title before hearing the dialogue and any mismatch breaks trust. Workflow:
  1. Decide the SPECIFIC scenario detail first (e.g. "asking where the metro is").
  2. Write title_target as the natural QUESTION/PHRASE the learner needs (e.g. ru: "Где метро?").
  3. Write title_native as the native-language headline matching that same specific scenario.
  4. Make sure turn 1 of the dialogue actually USES that question or a clearly equivalent one.

Fields:
- scenario: short native-lang label of the SPECIFIC scenario (not a broad category). vi: "Hỏi đường ra ga tàu điện ngầm"; en: "Asking where the metro is". Title-case. Under 32 chars.
- title_native: TOP header text in NATIVE language for the video card. UPPERCASE-ready. Idiomatic — DO NOT translate Vietnamese template literally. Under 36 chars. Examples:
    * vi: "Hỏi đường ra ga metro" / "Đặt món nhà hàng Đức"
    * en: "Asking Where The Metro Is" / "Ordering at a German Cafe"
    * ko: "지하철역 위치 묻기" / "독일 식당에서 주문하기"
    * ja: "メトロの場所を聞く" / "ドイツ食堂で注文"
- title_target: SHORT NATURAL phrase in TARGET language that MATCHES turn 1's question. Should be a phrase the viewer will recognise once they hear the dialogue. Under 22 chars. Examples:
    * de: "Im Restaurant" only for a general-restaurant scenario; for "asking the metro" use "Wo ist die U-Bahn?"
    * ru: "Где метро?" (not "Где улица?" if dialogue is about metro)
    * ko: "지하철역이 어디?" (not just "지하철")
    * zh: "地铁在哪?"
    * ja: "メトロはどこ?"
- outro_native: SPOKEN outro CTA in NATIVE lang, idiomatic. Under 90 chars. Examples:
    * vi: "Hãy like và follow để học mỗi ngày nhé!"
    * en: "Hit save and follow for more daily Russian — see you tomorrow!"
- topic_label: topic prefixed natively (vi: "về", en: "about", ko: "에 관한"). Lowercase.
- short_title: short noun phrase in NATIVE lang. SAME content as title_native but trimmed. Under 28 chars.
- short_title_target: SAME natural phrase as title_target. Under 22 chars.

═══ PART 3: 2 CHARACTERS ═══
char_a and char_b: the 2 people in the conversation. CASTING (strict): char_a = the LOCAL person native to the TARGET country (e.g. a Russian / German / Chinese / Korean staff member, employer, or official); char_b = the learner (the person who speaks the channel's NATIVE language — see PART 1 native_lang). Their appearance, clothing and role MUST fit this casting and the scenario, and BOTH characters MUST be depicted INSIDE the scenario's setting (never a random or foreign place).

For each character:
- label: "A" (typically the LOCAL native — waiter, employer, official) or "B" (typically the NATIVE-LANG-speaking learner — customer, candidate, student. For vi-native channels: Vietnamese-speaking; for en-native channels: English-speaking; etc.).
- name: short display name in NATIVE language ("Chị Anna" / "Anh Hùng" / "Cô Müller" / "Anh Park"). Under 16 chars.
- role: role label in NATIVE language ("phục vụ" / "khách hàng" / "nhà tuyển dụng" / "ứng viên"). Under 18 chars.
- voice_gender: "female" or "male". **CRITICAL: voice_gender MUST match the gender implied by the name.**
  Vietnamese prefixes are deterministic:
    • Male prefixes → voice_gender="male": "Anh ", "Bác ", "Ông ", "Chú ", "Cậu "
    • Female prefixes → voice_gender="female": "Chị ", "Cô ", "Bà ", "Dì ", "Em " (when clearly female)
    • Neutral prefixes ("Bạn ") → pick consistent with the FIRST NAME's gender.
  Foreign names: use the REAL-WORLD gender of the name (Hans/Müller/Bjorn/Park = male; Anna/Lan/Mei = female).
  Prefer opposite genders for A and B (better audio contrast) BUT if scenario realism requires same gender, that's fine — the renderer will vary TTS rate to keep voices distinct. NEVER sacrifice name-gender consistency for contrast.
- image_prompt: English description for AI image gen. ONE PERSON, head-and-shoulders portrait. It MUST contain, in this order: (1) the person's role + nationality/appearance fitting the casting above (char_a = LOCAL of the target country, char_b = the learner), (2) a short facial expression, (3) the SAME setting as scene_image_prompt but blurred — so the portrait clearly looks taken INSIDE that scene, NOT a random or foreign location. Both characters MUST share the SAME setting as each other and the scene. Do NOT name an art style (the renderer applies it). Avoid full body, sexual poses, weapons. Under 150 chars. E.g. (scene = Russian cafe):
  * "a friendly young Russian barista in an apron, smiling, inside a cozy Russian cafe interior, blurred background"
  * "a young foreign student customer smiling, seated inside the same cozy Russian cafe interior, blurred background"

═══ PART 4: SCENE BACKGROUND ═══
- scene_image_prompt: English description of the background SETTING (no people). Be specific to the scenario + target country (this is the SAME setting the two character portraits must be placed in). Do NOT name an art style (the renderer applies it). E.g.:
  * "a cozy Russian cafe interior with wooden tables and a samovar"
  * "a modern German office with desks and computers"
  Under 110 chars.

═══ PART 5: TURNS (the "turns" array, 6-8 items) ═══
Build a NATURAL conversation flow. A starts (usually local/native speaker), B responds (usually learner). Alternating A → B → A → B... but flexibility allowed.

For each turn:
- speaker: "A" or "B"
- target: full target-language sentence. NATURAL, conversational. Under 80 chars.
- pronunciation: phonetic per LANGUAGE STANDARD:
  * **zh**: PINYIN with tone marks, no hyphens between syllables of a word.
  * **ja**: Hepburn romaji.
  * **ko**: Revised Romanization.
  * **de/fr/es/ru**: Vietnamese-reader-friendly with hyphens + acute on stress. Under 90 chars.
- ipa: standard IPA in slashes. **For zh/ja/ko: ipa = ""**.
- native: idiomatic Vietnamese translation. Natural conversational tone. Under 70 chars.

═══ PART 6: CAPTION ═══
Structure:
  Line 1: "<FLAG> Tiếng <target_lang_name> mỗi ngày"
  blank
  Line 3: emoji + title_native
  blank
  "🎭 Hội thoại:"
  Each turn: "<A/B>: <target> — <native>" (limit to 6 for caption length)
  blank
  "👉 Lưu lại học mỗi ngày!"
  blank
  "#tieng<targetname> #hocngoaingu #hoithoai"

═══ RULES ═══
- 6 to 8 turns total, alternating A/B mostly.
- All turns CONVERSATIONAL (not textbook).
- Characters have CONSISTENT voice gender throughout.
- image_prompts: single person, head-and-shoulders, depicted INSIDE the scene's setting (blurred background). Never name an art style — the renderer applies it.
- char_a and char_b MUST be in the SAME setting (the scenario's place); char_a = local of the target country, char_b = the learner.
- scene_image_prompt is NO-PEOPLE setting (same place the portraits sit in).
- Scenario must be SAFE-FOR-WORK (no romance, no politics, no NSFW).
"""


DIALOGUE_SCHEMA = {
    "type": "OBJECT",
    "required": ["intent", "scenario", "title_native", "title_target",
                 "char_a", "char_b", "scene_image_prompt",
                 "turns", "outro_native", "topic_label",
                 "short_title", "short_title_target", "caption"],
    "properties": {
        "intent": {
            "type": "OBJECT",
            "required": ["target_lang", "native_lang", "topic", "count",
                         "voice_gender", "target_lang_name"],
            "properties": {
                "target_lang":      {"type": "STRING"},
                "native_lang":      {"type": "STRING"},
                "topic":            {"type": "STRING"},
                "count":            {"type": "INTEGER"},
                "voice_gender":     {"type": "STRING"},
                "target_lang_name": {"type": "STRING"},
            },
        },
        "scenario":           {"type": "STRING"},
        "title_native":       {"type": "STRING"},
        "title_target":       {"type": "STRING"},
        "outro_native":       {"type": "STRING"},
        "topic_label":        {"type": "STRING"},
        "short_title":        {"type": "STRING"},
        "short_title_target": {"type": "STRING"},
        "scene_image_prompt": {"type": "STRING"},
        "char_a": {
            "type": "OBJECT",
            "required": ["label", "name", "role", "voice_gender", "image_prompt"],
            "properties": {
                "label":         {"type": "STRING"},
                "name":          {"type": "STRING"},
                "role":          {"type": "STRING"},
                "voice_gender":  {"type": "STRING"},
                "image_prompt":  {"type": "STRING"},
            },
        },
        "char_b": {
            "type": "OBJECT",
            "required": ["label", "name", "role", "voice_gender", "image_prompt"],
            "properties": {
                "label":         {"type": "STRING"},
                "name":          {"type": "STRING"},
                "role":          {"type": "STRING"},
                "voice_gender":  {"type": "STRING"},
                "image_prompt":  {"type": "STRING"},
            },
        },
        "turns": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["speaker", "target", "pronunciation", "ipa", "native"],
                "properties": {
                    "speaker":       {"type": "STRING"},
                    "target":        {"type": "STRING"},
                    "pronunciation": {"type": "STRING"},
                    "ipa":           {"type": "STRING"},
                    "native":        {"type": "STRING"},
                },
            },
        },
        "caption":            {"type": "STRING"},
    },
}


def parse_and_generate_dialogue(
    user_text: str,
    *,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> tuple[ParsedIntent, DialogueContent]:
    """Generate a 2-character mini-dialogue video content from scenario request."""
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    resp = _call_gemini(client,
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=LANGUAGE_DIALOGUE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=DIALOGUE_SCHEMA,
            temperature=0.85,
        ),
    )
    data = json.loads(resp.text)

    intent_data = data["intent"]
    intent = ParsedIntent(
        target_lang=intent_data["target_lang"].lower(),
        native_lang=intent_data.get("native_lang", "vi").lower(),
        topic=intent_data["topic"],
        count=int(intent_data.get("count", 6)),
        voice_gender=(intent_data.get("voice_gender") or "any").lower(),
        target_lang_name=intent_data["target_lang_name"],
        layout_type="dialogue",
    )
    turns = [DialogueTurn(**t) for t in data["turns"]]
    if not (6 <= len(turns) <= 8):
        raise ValueError(f"dialogue needs 6-8 turns, got {len(turns)}")

    char_a = DialogueCharacter(**data["char_a"])
    char_b = DialogueCharacter(**data["char_b"])

    caption = _sanitize_hashtags(data["caption"])
    caption = _ensure_seo_hashtag(caption, intent.target_lang)
    content = DialogueContent(
        scenario=data["scenario"],
        title_native=data["title_native"],
        title_target=data["title_target"],
        char_a=char_a,
        char_b=char_b,
        scene_image_prompt=data["scene_image_prompt"],
        turns=turns,
        outro_native=data["outro_native"],
        topic_label=data["topic_label"],
        short_title=data["short_title"],
        short_title_target=data["short_title_target"],
        caption=caption,
    )
    return intent, content


# ═══════════════════════════════════════════════════════════════════════
#  FILL_BLANK_QUIZ layout — short photo+sentence fill-in-blank
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FillBlankContent:
    title_native: str           # "Ngữ pháp giới từ tiếng Đức" (CSS uppercase)
    title_target: str           # short topic in target ("Präpositionen")
    sentence_template: str      # target lang with ___ blank: "Ich gehe ___ Hause."
    correct_word: str           # the word that fills the blank: "nach"
    options: list[str]          # 3 options: ["nach", "zu", "bei"] (1 correct, 2 distractors)
    correct_index: int          # 0-based index of correct option in `options`
    native_translation: str     # VN translation of the FILLED sentence (Telegram admin only)
    explanation: str            # short VN explanation (Telegram admin only)
    scene_image_prompt: str     # English: "young person walking home on a city street, photorealistic"
    topic_label: str
    short_title: str
    short_title_target: str
    caption: str


LANGUAGE_FILL_BLANK_SYSTEM_PROMPT = """You generate a short 1-sentence fill-in-blank STATIC IMAGE POSTER (not a video). Format inspired by English Canbe / Grammar Goat. The image shows a photorealistic background (person doing an action), the sentence with ___ blank at top, 3 option chips below. NO countdown, NO reveal — viewers guess in comments.

Return ONE strict JSON object. NO prose outside JSON.

═══ PART 1: INTENT ═══
- target_lang: ISO 639-1 (e.g. "de", "zh", "ko")
- native_lang: ISO 639-1 INFERRED from user message language ("vi" if user wrote Vietnamese, "en" if English, etc.). ALL native-language fields below MUST be written IDIOMATICALLY in this native_lang — never translate Vietnamese examples verbatim.
- topic: grammar topic in native lang ("giới từ chỉ vị trí")
- count: 1
- voice_gender: "any"
- target_lang_name: ("Đức" / "Trung" / "Hàn")

═══ PART 2: HEADERS ═══
- title_native: 2-part header in NATIVE separated by colon ":". Format MANDATORY: "<CATEGORY tiếng <lang>>: <SPECIFIC TOPIC>". Line 1 = generic category, line 2 = specific topic. Examples:
  * "Ngữ pháp tiếng Đức: Giới từ"
  * "Trợ từ tiếng Hàn: 조사 chủ ngữ"
  * "Lượng từ tiếng Trung: 量词 cơ bản"
  * "Ngữ pháp tiếng Đức: Tính từ vị ngữ"
  Under 50 chars total. The colon ":" is REQUIRED — it tells the template to split into 2 lines.
- title_target: short topic in target lang. E.g. "Präpositionen" / "조사" / "量词" / "Prädikative Adjektive". Under 22 chars.
- topic_label: prefixed "về" ("về giới từ"). Lowercase.
- short_title: same as title_native.
- short_title_target: same as title_target.

═══ PART 3: SENTENCE + BLANK ═══
- sentence_template: ONE target-language sentence with EXACTLY ONE blank marked as `___` (three underscores). Natural, conversational, common. Under 60 chars. Examples:
  * de: "Ich gehe ___ Hause."  (blank for "nach")
  * de: "Sie wohnt ___ Berlin."  (blank for "in")
  * zh: "我在 ___ 看书。"  (blank for "家")
  * ko: "저는 ___ 갑니다."  (blank for "학교에")
- correct_word: the word that fills the blank EXACTLY (1-3 chars typical).
- options: ARRAY of EXACTLY 3 strings. RANDOM order (don't always put correct first). Include `correct_word` + 2 plausible distractors (same grammatical category). Similar length.
- correct_index: 0-based index of `correct_word` in `options` (0, 1, or 2).
- native_translation: VN translation of FILLED sentence (with correct word). Used for Telegram admin notify ONLY — NOT shown to FB viewers. Under 60 chars.
- explanation: 1 sentence VN explanation why correct. Telegram admin only. E.g. "Đáp án 'nach' dùng cho hướng đi với địa danh không có mạo từ (nach Hause, nach Berlin)." Under 140 chars.

═══ PART 4: SCENE IMAGE ═══
- scene_image_prompt: English PHOTOREALISTIC scene matching the sentence's action. NO text. Person doing action naturally. E.g.:
  * "Ich gehe nach Hause" → "young woman walking home on a city street at evening, smiling, photorealistic"
  * "Ich spiele Tennis" → "young person playing tennis on a court, photorealistic"
  * "我在家看书" → "young person reading book at home on sofa, cozy, photorealistic"
  Under 130 chars. Always end with "photorealistic".

═══ PART 5: CAPTION ═══
Structure:
  Line 1: "<FLAG> Tiếng <target_lang_name> giao tiếp"
  blank
  Line 3: "🤔 <sentence_template>"
  blank
  "🎯 Bạn chọn đáp án nào?"
  3 lines: "▫️ <opt 1>" / "▫️ <opt 2>" / "▫️ <opt 3>"
  blank
  "💬 Comment ngay bên dưới!"
  blank
  5-8 hashtags single line.
CRITICAL: DO NOT reveal correct answer in caption. NO "đáp án đúng là...". Total under 500 chars.

═══ RULES ═══
- 1 sentence with 1 blank.
- 3 options, randomly ordered.
- scene_image_prompt PHOTOREALISTIC (real photo style).
- NO voice/audio fields — this is a STATIC IMAGE poster.
"""


FILL_BLANK_SCHEMA = {
    "type": "OBJECT",
    "required": ["intent", "title_native", "title_target", "sentence_template",
                 "correct_word", "options", "correct_index", "native_translation",
                 "explanation", "scene_image_prompt",
                 "topic_label", "short_title", "short_title_target", "caption"],
    "properties": {
        "intent": {
            "type": "OBJECT",
            "required": ["target_lang", "native_lang", "topic", "count",
                         "voice_gender", "target_lang_name"],
            "properties": {
                "target_lang":      {"type": "STRING"},
                "native_lang":      {"type": "STRING"},
                "topic":            {"type": "STRING"},
                "count":            {"type": "INTEGER"},
                "voice_gender":     {"type": "STRING"},
                "target_lang_name": {"type": "STRING"},
            },
        },
        "title_native":        {"type": "STRING"},
        "title_target":        {"type": "STRING"},
        "sentence_template":   {"type": "STRING"},
        "correct_word":        {"type": "STRING"},
        "options": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "correct_index":       {"type": "INTEGER"},
        "native_translation":  {"type": "STRING"},
        "explanation":         {"type": "STRING"},
        "scene_image_prompt":  {"type": "STRING"},
        "topic_label":         {"type": "STRING"},
        "short_title":         {"type": "STRING"},
        "short_title_target":  {"type": "STRING"},
        "caption":             {"type": "STRING"},
    },
}


def parse_and_generate_fill_blank(
    user_text: str,
    *,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> tuple[ParsedIntent, FillBlankContent]:
    """Generate a 1-sentence fill-in-blank quiz from grammar topic request."""
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    resp = _call_gemini(client,
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=LANGUAGE_FILL_BLANK_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=FILL_BLANK_SCHEMA,
            temperature=0.85,
        ),
    )
    data = json.loads(resp.text)

    intent_data = data["intent"]
    intent = ParsedIntent(
        target_lang=intent_data["target_lang"].lower(),
        native_lang=intent_data.get("native_lang", "vi").lower(),
        topic=intent_data["topic"],
        count=1,
        voice_gender=(intent_data.get("voice_gender") or "any").lower(),
        target_lang_name=intent_data["target_lang_name"],
        layout_type="fill_blank",
    )
    options = list(data["options"])
    if len(options) != 3:
        raise ValueError(f"fill_blank needs exactly 3 options, got {len(options)}")
    correct_idx = int(data["correct_index"])
    if not (0 <= correct_idx <= 2):
        raise ValueError(f"correct_index must be 0/1/2, got {correct_idx}")

    caption = _sanitize_hashtags(data["caption"])
    caption = _ensure_seo_hashtag(caption, intent.target_lang)
    content = FillBlankContent(
        title_native=data["title_native"],
        title_target=data["title_target"],
        sentence_template=data["sentence_template"],
        correct_word=data["correct_word"],
        options=options,
        correct_index=correct_idx,
        native_translation=data["native_translation"],
        explanation=data["explanation"],
        scene_image_prompt=data["scene_image_prompt"],
        topic_label=data["topic_label"],
        short_title=data["short_title"],
        short_title_target=data["short_title_target"],
        caption=caption,
    )
    return intent, content


# ═══════════════════════════════════════════════════════════════════════
#  VOCAB_TABLE_IMAGE layout — static PNG poster
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VocabTableItem:
    """One row in the vocab table."""
    target_word: str      # target language: "der Topf" / "锅" / "냄비"
    pronunciation: str    # phonetic per language standard
    native_answer: str    # Vietnamese: "Cái nồi"


@dataclass
class VocabTableContent:
    title_native: str           # banner top: "TỪ VỰNG ĐỒ BẾP TRONG TIẾNG ĐỨC"
    title_target: str           # banner sub: "In der Küche"
    items: list[VocabTableItem]  # 8 items
    scene_image_prompt: str      # English: FULL photorealistic SCENE matching theme (like fill_blank scene). E.g. "young person cooking in a modern kitchen with pot on stove, photorealistic"
    topic_label: str
    short_title: str
    short_title_target: str
    caption: str                # FB caption with all 8 + pronunciation


LANGUAGE_VOCAB_TABLE_SYSTEM_PROMPT = """You generate a 8-row VOCABULARY TABLE for a STATIC PNG poster (not a video). The poster shows a 3-column table (VN | Target | Pronunciation) + a character mascot illustrating the theme + brand watermark. Format inspired by viral pages like Everyday Polish, English Canbe vocab posters.

Return ONE strict JSON object. NO prose outside JSON.

═══ PART 1: INTENT ═══
- target_lang: ISO 639-1
- native_lang: ISO 639-1 INFERRED from user message language ("vi" if user wrote Vietnamese, "en" if English, etc.). ALL native-language fields below MUST be written IDIOMATICALLY in this native_lang — never translate Vietnamese examples verbatim.
- topic: theme in native ("đồ vật nhà bếp")
- count: 8
- voice_gender: "any"
- target_lang_name: ("Đức" / "Trung" / "Hàn")

═══ PART 2: HEADERS ═══
- title_native: BANNER TOP text in NATIVE language. MANDATORY FORMAT (must follow exactly): "Từ vựng <topic noun phrase> trong tiếng <target_lang_name>". Title case (CSS will UPPERCASE). Examples:
    * "Từ vựng đồ bếp trong tiếng Đức"
    * "Từ vựng sân bay trong tiếng Hàn"
    * "Từ vựng khách sạn trong tiếng Trung"
    * "Từ vựng quần áo trong tiếng Đức"
  Under 50 chars. Keep topic noun phrase concise (1-3 words).
- title_target: SUB-BANNER text in TARGET lang. Natural location/topic phrase. E.g. de: "In der Küche" / zh: "在厨房" / ko: "주방에서" / de: "Am Flughafen" / ko: "공항에서". Under 18 chars.
- topic_label: prefixed "về" ("về đồ bếp"). Lowercase.
- short_title: short noun phrase native. Title Case. Under 24 chars.
- short_title_target: same in target lang. Under 16 chars.

═══ PART 3: 8 ITEMS ═══
Pick 8 CONCRETE common items from the theme. Easy-to-translate nouns.

Per item:
- target_word: target language word in NATIVE SCRIPT. German: include article (der/die/das + noun). Under 22 chars.
- pronunciation: phonetic per LANGUAGE STANDARD:
  * **zh**: PINYIN with tone marks, no hyphens.
  * **ja**: Hepburn romaji.
  * **ko**: Revised Romanization.
  * **de/fr/es/ru**: Vietnamese-reader-friendly with hyphens + acute. E.g. "der Topf" → "đơ tốp".
  Under 18 chars.
- native_answer: VN translation. Single noun phrase. Title Case. Under 16 chars.

═══ PART 4: SCENE BACKGROUND ═══
- scene_image_prompt: English description of a FULL PHOTOREALISTIC SCENE matching the theme. Used as full-screen background. Should have a person or activity that connects emotionally to the theme. Photorealistic, natural lighting, real-world setting (NOT flat illustration). NO text on image. Examples:
  * Kitchen theme → "young person cooking in a modern kitchen with pot on stove, photorealistic, natural light"
  * Hotel theme → "elegant hotel lobby with reception desk and luggage cart, photorealistic"
  * Airport theme → "passenger walking through bright airport terminal with luggage, photorealistic"
  * Restaurant theme → "person ordering food at a cozy restaurant with menu, photorealistic"
  * Clothes theme → "person browsing clothes in a modern boutique store, photorealistic"
  * Office theme → "young professional working at modern office desk with laptop, photorealistic"
  * Train station theme → "passenger waiting on platform at modern train station, photorealistic"
  * Coffee shop theme → "barista serving coffee in a warm coffee shop, photorealistic"
  Under 140 chars. NO text on image. Always include "photorealistic, natural light".

═══ PART 5: CAPTION ═══
Structure:
  Line 1: "<FLAG> Tiếng <target_lang_name> mỗi ngày"
  blank
  Line 3: emoji + title_native
  blank
  "📚 8 từ vựng hôm nay:"
  8 lines: "1. <target_word> — <pronunciation> — <native_answer>" ... "8. ..."
  blank
  "💾 Lưu ảnh để học mỗi ngày!"
  blank
  5-8 hashtags single line.

═══ RULES ═══
- EXACTLY 8 items.
- target_word in NATIVE SCRIPT.
- character matches theme naturally.
- All items kid-friendly, common.
"""


VOCAB_TABLE_SCHEMA = {
    "type": "OBJECT",
    "required": ["intent", "title_native", "title_target",
                 "topic_label", "short_title", "short_title_target",
                 "scene_image_prompt", "items", "caption"],
    "properties": {
        "intent": {
            "type": "OBJECT",
            "required": ["target_lang", "native_lang", "topic", "count",
                         "voice_gender", "target_lang_name"],
            "properties": {
                "target_lang":      {"type": "STRING"},
                "native_lang":      {"type": "STRING"},
                "topic":            {"type": "STRING"},
                "count":            {"type": "INTEGER"},
                "voice_gender":     {"type": "STRING"},
                "target_lang_name": {"type": "STRING"},
            },
        },
        "title_native":            {"type": "STRING"},
        "title_target":            {"type": "STRING"},
        "topic_label":             {"type": "STRING"},
        "short_title":             {"type": "STRING"},
        "short_title_target":      {"type": "STRING"},
        "scene_image_prompt":      {"type": "STRING"},
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["target_word", "pronunciation", "native_answer"],
                "properties": {
                    "target_word":   {"type": "STRING"},
                    "pronunciation": {"type": "STRING"},
                    "native_answer": {"type": "STRING"},
                },
            },
        },
        "caption":                 {"type": "STRING"},
    },
}


def parse_and_generate_vocab_table(
    user_text: str,
    *,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> tuple[ParsedIntent, VocabTableContent]:
    """Generate 8-row vocab table content for static PNG poster."""
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    resp = _call_gemini(client,
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=LANGUAGE_VOCAB_TABLE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=VOCAB_TABLE_SCHEMA,
            temperature=0.85,
        ),
    )
    data = json.loads(resp.text)

    intent_data = data["intent"]
    intent = ParsedIntent(
        target_lang=intent_data["target_lang"].lower(),
        native_lang=intent_data.get("native_lang", "vi").lower(),
        topic=intent_data["topic"],
        count=8,
        voice_gender=(intent_data.get("voice_gender") or "any").lower(),
        target_lang_name=intent_data["target_lang_name"],
        layout_type="vocab_table",
    )
    items = [VocabTableItem(**it) for it in data["items"]]
    if len(items) != 8:
        raise ValueError(f"vocab_table needs 8 items, got {len(items)}")

    caption = _sanitize_hashtags(data["caption"])
    caption = _ensure_seo_hashtag(caption, intent.target_lang)
    content = VocabTableContent(
        title_native=data["title_native"],
        title_target=data["title_target"],
        items=items,
        scene_image_prompt=data["scene_image_prompt"],
        topic_label=data["topic_label"],
        short_title=data["short_title"],
        short_title_target=data["short_title_target"],
        caption=caption,
    )
    return intent, content


# ═══════════════════════════════════════════════════════════════════════
#  COMPARE layout — 2-column basic vs fluent / don't say vs say
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ComparePair:
    """One row in the compare table — two phrasings side by side."""
    emoji: str            # 🤝 / 👍 / 😊 — small icon hinting at the situation
    left_target: str      # left side phrase in target lang (casual / basic / wrong)
    right_target: str     # right side phrase in target lang (formal / fluent / right)
    left_pron: str        # pronunciation guide for left (pinyin/romaji/romanization/IPA)
    right_pron: str       # pronunciation guide for right
    left_native: str      # VN translation of left phrase
    right_native: str     # VN translation of right phrase


@dataclass
class CompareContent:
    title_native: str       # banner top in VN: "Cách nói cảm ơn lịch sự hơn"
    title_target: str       # banner sub in target: "更礼貌的感谢" / "Höflicher danken"
    mode: str               # "basic_vs_fluent" | "casual_vs_formal" | "dont_vs_say" | "mistake_vs_correct"
    left_header: str        # column 1 label in VN: "Cơ bản" / "Đừng nói" / "Bình thường" / "Sai"
    right_header: str       # column 2 label in VN: "Lịch sự hơn" / "Nói thay vào" / "Trang trọng" / "Đúng"
    pairs: list[ComparePair]  # 8 pairs
    topic_label: str        # for picker rotation tracking
    short_title: str        # nav title VN
    short_title_target: str # nav title target lang
    caption: str            # FB caption: full 8 pairs + VN cheatsheet


LANGUAGE_COMPARE_SYSTEM_PROMPT = """You generate an 8-pair 2-COLUMN COMPARISON for a STATIC IMAGE poster (not a video). Format inspired by viral pages like "Basic vs Fluent English", "Don't say / Say", "Cách nói tinh tế hơn".

Audience: native-{native_lang} learners of {target_lang_name}. Caption + table headers are in the USER's NATIVE language (inferred from the user message — vi if Vietnamese, en if English, etc.); the actual phrases being compared are in {target_lang_name} (with pronunciation guide).

Return ONE strict JSON object. NO prose outside JSON.

═══ PART 1: INTENT ═══
- target_lang: ISO 639-1 ("de" / "zh" / "ko" / "ja" / "fr" / "ru" / "es")
- native_lang: ISO 639-1 INFERRED from user message language ("vi" if user wrote Vietnamese, "en" if English, etc.). ALL native-language fields below MUST be written IDIOMATICALLY in this native_lang — never translate Vietnamese examples verbatim.
- topic: theme in native ("cách nói cảm ơn lịch sự")
- count: 8
- voice_gender: "any"
- target_lang_name: ("Đức" / "Trung" / "Hàn" / ...)

═══ PART 2: MODE ═══
Pick ONE mode based on the request theme:
- "basic_vs_fluent": casual learner phrase → advanced native-speaker phrase
- "casual_vs_formal": informal/friendly → polite/professional (Du vs Sie, 你 vs 您, 반말 vs 존댓말)
- "dont_vs_say": rude/awkward phrase → polite alternative
- "mistake_vs_correct": common learner mistake → correct version

═══ PART 3: HEADERS & TITLE ═══
- title_native: short VN headline describing the comparison. Title Case. Under 40 chars. Examples:
    * "Cảm ơn tinh tế hơn"
    * "Casual → Lịch sự (Sie / 您 / 존댓말)"
    * "Đừng nói thế — nói thay vào"
    * "Lỗi sai phổ biến → câu đúng"
- title_target: short phrase in TARGET lang reflecting the mode. Under 22 chars. Examples:
    * zh: "更礼貌的说法"
    * de: "Höflicher sagen"
    * ko: "더 자연스럽게"
- left_header: VN column 1 label. Under 18 chars. Mode-aware:
    * basic_vs_fluent → "Cơ bản" / "Phổ thông"
    * casual_vs_formal → "Thân mật" / "Bạn bè"
    * dont_vs_say → "Đừng nói"
    * mistake_vs_correct → "Sai"
- right_header: VN column 2 label. Under 18 chars. Mode-aware:
    * basic_vs_fluent → "Tinh tế hơn" / "Như người bản xứ"
    * casual_vs_formal → "Lịch sự" / "Trang trọng"
    * dont_vs_say → "Nói thay vào"
    * mistake_vs_correct → "Đúng"
- topic_label: prefixed "về" ("về cách nói cảm ơn"). Lowercase.
- short_title: same as title_native, under 24 chars.
- short_title_target: same as title_target.

═══ PART 4: 8 PAIRS ═══
Generate EXACTLY 8 pairs. Each pair shows the SAME communicative intent expressed two ways.

Per pair:
- emoji: ONE relevant emoji per situation. 🤝 cảm ơn, 👋 chào, 🍽️ ăn uống, 💼 công việc, 😊 cảm xúc, ⏰ thời gian, 🤔 thắc mắc, 💪 động viên, 🙏 xin lỗi, 🎉 chúc mừng. Single emoji char only.
- left_target: target-lang phrase (CASUAL / BASIC / WRONG side). NATIVE SCRIPT. Under 25 chars.
- right_target: target-lang phrase (FORMAL / FLUENT / CORRECT side). Same meaning, more advanced. Under 30 chars.
- left_pron: pronunciation guide. PER LANGUAGE STANDARD:
    * zh: PINYIN with tone marks. "xiè xie"
    * ko: Revised Romanization. "annyeonghaseyo"
    * ja: Hepburn romaji. "arigatou"
    * de/fr/es/ru: Vietnamese-reader-friendly with hyphens + acute. "đan-kê"
  Under 22 chars.
- right_pron: same scheme. Under 28 chars.
- left_native: VN translation of left_target. Under 30 chars.
- right_native: VN translation of right_target. Under 36 chars.

Pair quality rules:
- Both sides MEAN THE SAME THING (just expressed differently).
- Right side should be NOTICEABLY BETTER for the mode (more polite/fluent/correct).
- Variety of emoji + situations across the 8 pairs.
- Common practical phrases natives actually use daily.
- **CRITICAL: left_target MUST DIFFER from right_target — character-by-character, no exceptions.**
  If you cannot think of a credible mistake / casual / basic variant for a particular
  meaning, DO NOT output the same phrase on both sides. Instead PICK A DIFFERENT
  scenario where the comparison is meaningful. Identical pairs ruin the entire post.
- Examples of GOOD pairs (mode=mistake_vs_correct, target=ru):
    * left "Это для ты"      / right "Это для тебя"        (wrong case → correct accusative)
    * left "Звоню друг"      / right "Звоню другу"          (missing dative ending)
    * left "Книга брат"      / right "Книга брата"          (missing genitive ending)
- Examples of BAD pairs (NEVER do this):
    * left "Я иду на работу" / right "Я иду на работу"     ← IDENTICAL, useless
    * left "Я в магазине"    / right "Я в магазине"        ← IDENTICAL, useless

═══ PART 5: CAPTION ═══
Structure:
  Line 1: "<FLAG> Tiếng <target_lang_name> mỗi ngày"
  blank
  Line 3: "✨ <title_native>"
  blank
  "📚 8 cặp hôm nay:"
  For each pair i in 1..8:
    "{i}. {emoji} {left_native} → {right_native}"
    "   {left_target} ({left_pron}) → {right_target} ({right_pron})"
    blank
  "💾 Lưu ảnh để dùng khi cần!"
  blank
  5-8 hashtags single line.

═══ RULES ═══
- EXACTLY 8 pairs.
- target phrases in NATIVE SCRIPT (汉字, 한글, Umlauts, etc).
- Right side ALWAYS the better/more advanced version.
- Pronunciation MUST follow per-language standard exactly.
"""


COMPARE_SCHEMA = {
    "type": "OBJECT",
    "required": ["intent", "title_native", "title_target", "mode",
                 "left_header", "right_header",
                 "topic_label", "short_title", "short_title_target",
                 "pairs", "caption"],
    "properties": {
        "intent": {
            "type": "OBJECT",
            "required": ["target_lang", "native_lang", "topic", "count",
                         "voice_gender", "target_lang_name"],
            "properties": {
                "target_lang":      {"type": "STRING"},
                "native_lang":      {"type": "STRING"},
                "topic":            {"type": "STRING"},
                "count":            {"type": "INTEGER"},
                "voice_gender":     {"type": "STRING"},
                "target_lang_name": {"type": "STRING"},
            },
        },
        "title_native":         {"type": "STRING"},
        "title_target":         {"type": "STRING"},
        "mode":                 {"type": "STRING"},
        "left_header":          {"type": "STRING"},
        "right_header":         {"type": "STRING"},
        "topic_label":          {"type": "STRING"},
        "short_title":          {"type": "STRING"},
        "short_title_target":   {"type": "STRING"},
        "pairs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["emoji", "left_target", "right_target",
                             "left_pron", "right_pron",
                             "left_native", "right_native"],
                "properties": {
                    "emoji":        {"type": "STRING"},
                    "left_target":  {"type": "STRING"},
                    "right_target": {"type": "STRING"},
                    "left_pron":    {"type": "STRING"},
                    "right_pron":   {"type": "STRING"},
                    "left_native":  {"type": "STRING"},
                    "right_native": {"type": "STRING"},
                },
            },
        },
        "caption":              {"type": "STRING"},
    },
}


def _normalize_compare_text(s: str) -> str:
    """Whitespace + punctuation insensitive compare for left/right equality check."""
    import re as _re
    return _re.sub(r"[\s\.,!?;:'\"\-—]+", "", s.strip().lower())


def parse_and_generate_compare(
    user_text: str,
    *,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> tuple[ParsedIntent, CompareContent]:
    """Generate 8-pair 2-column comparison for static PNG poster.

    CEO bug 2026-06-13 (Russian Path "I иду на работу" appeared identically on
    BOTH columns row 1+2): Gemini sometimes lazy-outputs same string on both
    sides when it cannot think of a credible slip-up. Validation post-process
    detects identical pairs and retries ONCE with explicit feedback.
    """
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    current_user_text = user_text
    data = None
    pairs: list[ComparePair] = []

    for attempt in range(2):  # initial + 1 retry max
        resp = _call_gemini(client,
            model=model,
            contents=current_user_text,
            config=types.GenerateContentConfig(
                system_instruction=LANGUAGE_COMPARE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=COMPARE_SCHEMA,
                temperature=0.85 + (0.05 * attempt),  # bump temp on retry
            ),
        )
        data = json.loads(resp.text)
        pairs = [ComparePair(**p) for p in data["pairs"]]
        if len(pairs) != 8:
            if attempt == 0:
                current_user_text = user_text + "\n\nMust return EXACTLY 8 pairs."
                continue
            raise ValueError(f"compare needs 8 pairs, got {len(pairs)}")

        # Detect identical left/right pairs (the bug)
        identical_idx = [
            i for i, p in enumerate(pairs, start=1)
            if _normalize_compare_text(p.left_target) == _normalize_compare_text(p.right_target)
        ]
        if not identical_idx:
            break  # all pairs differ — good

        if attempt == 0:
            # Retry with explicit feedback citing offending row numbers
            current_user_text = (
                f"{user_text}\n\nPREVIOUS ATTEMPT FAILED: pairs {identical_idx} had "
                f"left_target IDENTICAL to right_target — comparison is meaningless. "
                f"Regenerate ALL 8 pairs. Every pair MUST have left and right phrases "
                f"that differ at the character level. If you can't think of a credible "
                f"slip-up for a phrase, choose a DIFFERENT scenario where the mistake "
                f"is common."
            )
            continue
        # Second attempt also produced identical pairs — fail loud so admin
        # notices instead of shipping a broken poster.
        raise ValueError(
            f"compare layout has identical left==right at pair(s) {identical_idx} "
            f"after retry. Gemini cannot generate meaningful comparison for this topic."
        )

    assert data is not None and pairs  # for type checker

    intent_data = data["intent"]
    intent = ParsedIntent(
        target_lang=intent_data["target_lang"].lower(),
        native_lang=intent_data.get("native_lang", "vi").lower(),
        topic=intent_data["topic"],
        count=8,
        voice_gender=(intent_data.get("voice_gender") or "any").lower(),
        target_lang_name=intent_data["target_lang_name"],
        layout_type="compare",
    )

    caption = _sanitize_hashtags(data["caption"])
    caption = _ensure_seo_hashtag(caption, intent.target_lang)
    content = CompareContent(
        title_native=data["title_native"],
        title_target=data["title_target"],
        mode=data["mode"],
        left_header=data["left_header"],
        right_header=data["right_header"],
        pairs=pairs,
        topic_label=data["topic_label"],
        short_title=data["short_title"],
        short_title_target=data["short_title_target"],
        caption=caption,
    )
    return intent, content


# Map of niche → system prompt. Schema reused across all niches.
NICHE_PROMPTS = {
    "language": COMBINED_SYSTEM_PROMPT,
    "health":   HEALTH_SYSTEM_PROMPT,
}


def parse_and_generate(
    user_text: str,
    *,
    niche: str = "language",
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> tuple[ParsedIntent, GeneratedContent]:
    """One Gemini call that does both intent parsing AND content generation.

    niche: "language" (default) | "health" | future niches.
           Selects the system prompt — schema stays the same.
    """
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    system_prompt = NICHE_PROMPTS.get(niche, COMBINED_SYSTEM_PROMPT)

    resp = _call_gemini(client,
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=COMBINED_SCHEMA,
            temperature=0.7,
        ),
    )
    data = json.loads(resp.text)

    intent_data = data["intent"]
    count = max(3, min(20, int(intent_data.get("count", 10))))
    gender = intent_data.get("voice_gender", "any").lower()
    if gender not in {"male", "female", "any"}:
        gender = "any"
    intent = ParsedIntent(
        target_lang=intent_data["target_lang"].lower(),
        native_lang=intent_data["native_lang"].lower(),
        topic=intent_data["topic"],
        count=count,
        voice_gender=gender,
        target_lang_name=intent_data["target_lang_name"],
    )

    phrases = [Phrase(**p) for p in data["phrases"]]
    # Post-validation: force intent.count to match actual phrases length.
    # Reason: AI may auto-pick a different count for non-language niches.
    intent.count = len(phrases)
    caption = _sanitize_hashtags(data["caption"])
    caption = _ensure_seo_hashtag(caption, intent.target_lang)
    content = GeneratedContent(
        intro_display=data["intro_display"],
        intro_translation=data["intro_translation"],
        intro_tts=data["intro_tts"],
        intro_native=data["intro_native"],
        outro_native=data["outro_native"],
        topic_label=data["topic_label"],
        short_title=data["short_title"],
        short_title_target=data["short_title_target"],
        phrases=phrases,
        caption=caption,
        scene_image_prompt=data.get("scene_image_prompt", ""),
    )
    return intent, content


# ───────────────────────────────────────────────────────────────────────
# Legacy individual calls (still useful for testing parse_intent alone)
# ───────────────────────────────────────────────────────────────────────

def parse_intent(
    user_text: str,
    *,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> ParsedIntent:
    """Use Gemini to extract structured intent from a free-form message."""
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    resp = _call_gemini(client,
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=INTENT_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=INTENT_SCHEMA,
            temperature=0.0,
        ),
    )
    data = json.loads(resp.text)
    count = max(3, min(20, int(data.get("count", 10))))
    gender = data.get("voice_gender", "female").lower()
    if gender not in {"male", "female"}:
        gender = "female"
    return ParsedIntent(
        target_lang=data["target_lang"].lower(),
        native_lang=data["native_lang"].lower(),
        topic=data["topic"],
        count=count,
        voice_gender=gender,
        target_lang_name=data["target_lang_name"],
    )


SYSTEM_PROMPT = """You generate language-learning content for short-form vertical videos.
Output strict JSON. No prose outside the JSON.

Rules:
- All target-language strings MUST be in the script of the target language (Cyrillic for Russian, Hangul for Korean, Hanzi for Chinese, etc.).
- Pronunciation = Latin transliteration ("Vietnamese-reader-friendly" style) with hyphens on syllable boundaries and accent marks on stressed syllables (e.g. "Ya tib-yá liu-bliú"). Keep it under 28 chars when possible.
- ipa = standard IPA notation wrapped in slashes (e.g. "/jɑ tʲɪˈbʲɑ lʲʊˈblʲʊ/"). Use the official IPA conventions for the target language. Under 32 chars.
- Each "target" phrase is also for TTS narration — prefer 2–5 word phrases that sound natural spoken aloud. No quotes around the phrase. End with a period or exclamation.
- "native" = idiomatic meaning in the user's native language (not literal). Keep under 36 chars so it fits one line.
- "intro_display" is the headline shown on the title card in the native language (e.g. "10 cụm từ tiếng Nga về tình yêu"). Format: "<N> <native-word-for-phrases> tiếng <native-lang-name> <topic-label>" where N matches phrase count.
- "intro_translation" is the same headline rendered in the TARGET language as a short noun phrase (e.g. "10 русских фраз о любви", "10 deutsche Motivationssätze"). Used as the sticky header subtitle on every phrase card. Keep under 40 chars.
- "intro_tts" is a natural full sentence in the TARGET language the narrator will speak (e.g. "Десять русских фраз о любви.").
- "topic_label" is the topic in the native language as a short noun phrase prefixed with "về" or "chủ đề" (e.g. "về tình yêu", "về công việc"). Lowercase, no period.
- "caption" is a MULTI-LINE social-media caption in the native language with this exact structure (use real newlines, not literal "\\n"):
    Line 1: emoji + headline in native lang
    Blank line
    "🎧 Nghe & học cùng nhau:"
    Three lines, each one of "1. <target_phrase> — <native_meaning>" (pick phrases 1, 2, 3)
    Blank line
    Two CTA lines, one inviting to save and one inviting to follow for daily practice (use 💾 and ❤️ emoji)
    Blank line
    5-10 hashtags space-separated on a single line.
    Total caption length under 700 chars.
- HASHTAG RULES: lowercase, ASCII letters and digits only — NEVER contain hyphens "-", underscores "_", or any punctuation. Examples valid: "#tiengduc", "#hoctiengduc", "#motivation". Examples invalid: "#tieng-duc", "#hoc_tiengduc", "#học_tiếng_đức".
- Hashtags should mix native-language transliteration (no diacritics) and English. 5–10 total.
"""

SCHEMA = {
    "type": "OBJECT",
    "required": ["intro_display", "intro_translation", "intro_tts", "topic_label",
                 "short_title", "short_title_target", "phrases", "caption"],
    "properties": {
        "intro_display": {"type": "STRING"},
        "intro_translation": {"type": "STRING"},
        "intro_tts": {"type": "STRING"},
        "intro_native": {"type": "STRING"},
        "outro_native": {"type": "STRING"},
        "topic_label": {"type": "STRING"},
        "short_title": {"type": "STRING"},
        "short_title_target": {"type": "STRING"},
        "phrases": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["target", "pronunciation", "ipa", "native"],
                "properties": {
                    "target": {"type": "STRING"},
                    "pronunciation": {"type": "STRING"},
                    "ipa": {"type": "STRING"},
                    "native": {"type": "STRING"},
                },
            },
        },
        "caption": {"type": "STRING"},
    },
}


_HASHTAG_RE = re.compile(r"(#)([^\s#]+)")


# Target lang code → Vietnamese transliterated slug (for #tiengXgiaotiep SEO hashtag)
LANG_SLUG = {
    "de": "duc", "ru": "nga", "en": "anh", "fr": "phap",
    "ja": "nhat", "ko": "han", "es": "tbn", "zh": "trung",
    "th": "thai", "id": "indo", "pt": "bodaonha", "it": "y",
    "nl": "halan", "pl": "balan", "tr": "thonhiky", "vi": "viet",
    "sv": "thuydien", "da": "danmach", "no": "nauy", "fi": "phanlan",
}


def _ensure_seo_hashtag(caption: str, target_lang: str) -> str:
    """Inject the SEO anchor hashtag if missing from caption.

    Lingora-aware: when the caption is in English (native_lang=en) the
    Vietnamese "#tiengXgiaotiep" anchor would look out of place, so we
    skip injection — the caption already has English #learnrussian etc.
    A caption is treated as English when none of the lines start with a
    Vietnamese SEO line "Tiếng X giao tiếp" (the in-prompt anchor).
    """
    slug = LANG_SLUG.get((target_lang or "").lower(), "")
    if not slug:
        return caption
    # Heuristic: a Vietnamese caption ALWAYS contains "tiếng" lowercase in
    # the title line. If absent, treat caption as English and skip injection.
    if "tiếng" not in caption.lower() and "tieng" not in caption.lower():
        return caption
    tag = f"#tieng{slug}giaotiep"
    if tag in caption.lower():
        return caption
    lines = caption.splitlines()
    last_idx = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)
    if last_idx is not None and _looks_like_hashtag_line(lines[last_idx]):
        lines[last_idx] = lines[last_idx].rstrip() + " " + tag
    else:
        lines.append(tag)
    return "\n".join(lines)


def _normalize_tag(token: str) -> str:
    """Strip leading '#', diacritics and non-alphanumerics; lowercase; re-add '#'."""
    import unicodedata
    clean = token.lstrip("#").strip()
    nfd = unicodedata.normalize("NFD", clean)
    ascii_only = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    cleaned = re.sub(r"[^A-Za-z0-9]", "", ascii_only).lower()
    return "#" + cleaned if cleaned else ""


def _looks_like_hashtag_line(line: str) -> bool:
    """Heuristic: a hashtag line is space-separated short tokens, mostly with '#',
    no sentence-ending punctuation."""
    tokens = line.split()
    if len(tokens) < 3:
        return False
    if any(t.endswith((":", "!", "?", "。", ".")) for t in tokens):
        return False
    # Most tokens should be short (< 30 chars) — emojis can be longer in bytes but
    # we just look at char count
    if not all(len(t) < 32 for t in tokens):
        return False
    # At least one token already has '#', OR every token is alphanumeric-ish (no spaces in tokens)
    has_any_hash = any(t.startswith("#") for t in tokens)
    all_alnum_like = all(re.fullmatch(r"[#A-Za-z0-9_\-]+", t) for t in tokens)
    return has_any_hash or all_alnum_like


def _sanitize_hashtags(caption: str) -> str:
    """Enforce '#' prefix on the hashtag line + sanitize every inline hashtag.

    The hashtag line is typically the last non-empty line. We detect it
    heuristically and re-format ALL tokens to ensure they start with '#'
    and contain only lowercase ASCII letters/digits.
    """
    lines = caption.splitlines()

    # Find last non-empty line
    last_idx = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)

    if last_idx is not None and _looks_like_hashtag_line(lines[last_idx]):
        new_tokens = []
        for tok in lines[last_idx].split():
            normed = _normalize_tag(tok)
            if normed:
                new_tokens.append(normed)
        lines[last_idx] = " ".join(new_tokens)

    # Also sanitize any inline hashtags in other lines (they already have '#')
    for i, line in enumerate(lines):
        if i == last_idx:
            continue
        lines[i] = _HASHTAG_RE.sub(
            lambda m: _normalize_tag(m.group(0)),
            line,
        )

    return "\n".join(lines)


def generate(
    topic: str,
    target_lang: str = "ru",
    native_lang: str = "vi",
    count: int = 10,
    *,
    target_lang_name: Optional[str] = None,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> GeneratedContent:
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    if target_lang_name is None:
        target_lang_name = LANG_NAMES.get(target_lang, target_lang)
    native_lang_name = LANG_NAMES.get(native_lang, native_lang)

    user_prompt = (
        f"Topic: {topic}\n"
        f"Target language: {target_lang_name}\n"
        f"Native language: {native_lang_name}\n"
        f"Phrase count: {count}\n\n"
        f"Generate exactly {count} phrases following the JSON schema."
    )

    resp = _call_gemini(client,
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=SCHEMA,
            temperature=0.7,
        ),
    )
    data = json.loads(resp.text)

    phrases = [Phrase(**p) for p in data["phrases"]]
    if len(phrases) != count:
        raise ValueError(f"LLM returned {len(phrases)} phrases, expected {count}")

    return GeneratedContent(
        intro_display=data["intro_display"],
        intro_translation=data["intro_translation"],
        intro_tts=data["intro_tts"],
        intro_native=data["intro_native"],
        outro_native=data["outro_native"],
        topic_label=data["topic_label"],
        short_title=data["short_title"],
        short_title_target=data["short_title_target"],
        phrases=phrases,
        caption=_sanitize_hashtags(data["caption"]),
    )


# ═══════════════════════════════════════════════════════════════════════════
# GUESS_WORD LAYOUT — single-word reveal with countdown (added 2026-06-15)
# Engagement-bait template #10. NO image — pure typography + countdown.
# Show native translation → first-letter hint → countdown 0 → reveal target
# word (with TTS voice) + part-of-speech chip.
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GuessWord:
    """One round in the guess_word video — 10 per video."""
    target_word: str          # full word in target lang, e.g. "Vergebung"
    native_translation: str   # meaning in native lang, e.g. "sự tha thứ"
    first_letter_hint: str    # one-letter hint + dashes, e.g. "V _ _ _ _ _ _ _ _"
    part_of_speech: str       # "n" | "v" | "adj" | "adv"
    ipa: str                  # IPA in slashes, e.g. "/fɛɐˈɡeːbʊŋ/"
    example_sentence: str = ""  # short sentence using target word (≤ 70 chars)


@dataclass
class GuessWordContent:
    """10-word guess game content. Pure typography template (no Pexels/AI image)."""
    title_native: str          # "Thử đoán xem" in user's native lang
    words: list[GuessWord]     # exactly 10
    intro_native: str          # spoken intro in native ("Cùng đoán 10 từ tiếng Đức nhé!")
    outro_native: str          # spoken outro CTA
    topic_label: str           # short topic in native ("về tình yêu")
    short_title: str           # rút gọn sticky header line 1
    short_title_target: str    # rút gọn dịch sang target
    caption: str               # multi-line caption — MUST start with "Luyện từ vựng..." pattern


# ─────────────────────────────────────────────────────────────────────────
# title_native ("Thử đoán xem") translation table per native_lang
# ─────────────────────────────────────────────────────────────────────────
_GUESS_TITLE_BY_NATIVE = {
    "vi": "Thử đoán xem",
    "en": "Guess the word",
    "ko": "단어 맞히기",
    "ja": "単語を当ててみよう",
    "zh": "猜单词",
    "de": "Errate das Wort",
    "fr": "Devinez le mot",
    "es": "Adivina la palabra",
    "ru": "Угадайте слово",
}

# ─────────────────────────────────────────────────────────────────────────
# Caption opener — "Luyện từ vựng" pattern per native lang.
# vi is SPECIAL: "Luyện từ vựng IELTS" (CEO direction 2026-06-15).
# ─────────────────────────────────────────────────────────────────────────
_GUESS_CAPTION_OPENER_BY_NATIVE = {
    "vi": "Luyện từ vựng IELTS",
    "en": "Vocabulary practice",
    "ko": "어휘 연습",
    "ja": "語彙練習",
    "zh": "词汇练习",
    "de": "Vokabeltraining",
    "fr": "Pratique du vocabulaire",
    "es": "Práctica de vocabulario",
    "ru": "Тренировка лексики",
}


GUESS_WORD_SCHEMA = {
    "type": "OBJECT",
    "required": [
        "intent", "title_native", "words", "intro_native", "outro_native",
        "topic_label", "short_title", "short_title_target", "caption",
    ],
    "properties": {
        "intent": {
            "type": "OBJECT",
            "required": [
                "target_lang", "native_lang", "topic", "count",
                "voice_gender", "target_lang_name",
            ],
            "properties": {
                "target_lang": {"type": "STRING"},
                "native_lang": {"type": "STRING"},
                "topic": {"type": "STRING"},
                "count": {"type": "INTEGER"},
                "voice_gender": {"type": "STRING"},
                "target_lang_name": {"type": "STRING"},
            },
        },
        "title_native": {"type": "STRING"},
        "words": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["target_word", "native_translation", "first_letter_hint", "part_of_speech", "ipa", "example_sentence"],
                "properties": {
                    "target_word": {"type": "STRING"},
                    "native_translation": {"type": "STRING"},
                    "first_letter_hint": {"type": "STRING"},
                    "part_of_speech": {"type": "STRING"},
                    "ipa": {"type": "STRING"},
                    "example_sentence": {"type": "STRING"},
                },
            },
        },
        "intro_native": {"type": "STRING"},
        "outro_native": {"type": "STRING"},
        "topic_label": {"type": "STRING"},
        "short_title": {"type": "STRING"},
        "short_title_target": {"type": "STRING"},
        "caption": {"type": "STRING"},
    },
}


LANGUAGE_GUESS_WORD_SYSTEM_PROMPT = """You generate GUESS-THE-WORD content for a short-form vertical (9:16) language-learning video. Output strict JSON, no prose.

═══ PART 1: INTENT (top-level "intent" field) ═══
- target_lang: ISO 639-1 of the language being LEARNED (e.g. "de", "ru", "ko", "ja", "en")
- native_lang: ISO 639-1 the user wrote in (e.g. "vi", "en")
- topic: short phrase in user's native lang (e.g. "tình yêu", "cảm xúc", "công việc")
- count: ALWAYS 10
- voice_gender: "female" | "male" | "any" (default "any")
- target_lang_name: lang name in user's native lang (e.g. "Đức", "Nga", "Hàn")
- layout_type: ALWAYS "guess_word" (auto-injected by parser)

═══ PART 2: TITLE ═══
- title_native: the phrase "Guess the word" in user's NATIVE lang. Use EXACTLY one of:
    * vi: "Thử đoán xem"
    * en: "Guess the word"
    * ko: "단어 맞히기"
    * ja: "単語を当ててみよう"
    * zh: "猜单词"
    * de: "Errate das Wort"
    * fr: "Devinez le mot"
    * es: "Adivina la palabra"
    * ru: "Угадайте слово"
  If native_lang is not in the list above, write the equivalent natural phrasing in that language.

═══ PART 3: WORDS (exactly 10) ═══
Each word object has:
- target_word: the single word in TARGET language script (Cyrillic for ru, Hangul for ko, Hanzi for zh, native German etc.). SINGLE WORDS only — no phrases. Examples: "Vergebung" (de), "счастье" (ru), "사랑" (ko), "幸福" (zh).
- native_translation: meaning of target_word in user's native lang. SHORT (under 25 chars). Examples: "sự tha thứ", "hạnh phúc", "tình yêu".
- first_letter_hint: ONLY the first letter of target_word, followed by underscores (one per remaining letter) separated by spaces.
    * Example for "Vergebung" (9 letters): "V _ _ _ _ _ _ _ _"
    * Example for "счастье" (7 letters): "с _ _ _ _ _ _"
    * Example for "사랑" (2 syllables): "사 _"
    * Example for "幸福" (2 chars): "幸 _"
  USE EXACTLY THE FIRST CHARACTER of target_word (preserve case, preserve Cyrillic/CJK script).
- part_of_speech: ONE of "n" (noun), "v" (verb), "adj" (adjective), "adv" (adverb). Pick the most common usage for that word.
- ipa: standard IPA in slashes, e.g. "/fɛɐˈɡeːbʊŋ/". For zh/ja/ko, ipa = "" (empty — script is already phonetic).
- example_sentence: ONE short sentence (≤ 70 chars) in TARGET LANGUAGE that uses target_word naturally. Example: for "Hindernis": "Jedes Hindernis ist eine neue Chance." For "Achieve" (en): "Students who study hard can achieve their goals." Keep it B1-B2 friendly — common verbs + simple clause structure. NO multi-clause complex sentences.

WORD SELECTION RULES:
1. EXACTLY 10 words. No more, no less.
2. All words MUST relate to the topic.
3. **DIFFICULTY: B1+ / IELTS intermediate to advanced.** This is an engagement
   game for learners who ALREADY know basic vocab. CEO direction 2026-06-15:
   - ❌ FORBIDDEN: A1 baby words. NO "color" basics ("rot", "blau"); NO daily
     animals ("Hund", "Katze"); NO simple greetings ("Hallo", "Danke"); NO
     numbers ("eins", "zwei"); NO single-letter body parts ("Auge", "Mund").
   - ✓ TARGET: B1-C1 vocabulary that an intermediate learner would PAUSE to
     guess. Examples by topic for German:
     · emotions   → Vergebung, Mitgefühl, Sehnsucht, Erleichterung
     · obstacles  → Hindernis, Herausforderung, Widerstand, Niederlage
     · work       → Verantwortung, Verhandlung, Engagement, Beförderung
     · nature     → Dämmerung, Wildnis, Strömung, Lichtung
   - Mix difficulty WITHIN B1+ band: 4 medium-B1, 4 medium-B2, 2 hard-C1.
   - If topic is very basic (e.g. "colors", "animals", "food"), still PICK
     advanced/uncommon terms in that domain (e.g. for "colors" use
     "Karminrot/Türkis/Senfgelb", not "rot/blau").
4. Mix parts of speech: aim for 4-6 nouns, 2-4 verbs, 1-3 adjectives, 0-2 adverbs.
5. NO duplicate first_letter_hint (each word should start with a different letter when possible).
6. SAFE FOR ALL AUDIENCES — no politics, no NSFW, no slurs.

═══ PART 4: NARRATION (in native_lang) ═══
- intro_native: short spoken hook in user's native lang. 1-2 short sentences.
- outro_native: short CTA in native lang.

═══ PART 5: LABELS ═══
- topic_label: short topic in native lang ("về tình yêu", "about love")
- short_title: 1-3 word version of topic in native lang ("Tình yêu", "Love")
- short_title_target: same translated to target lang ("Liebe", "Любовь")

═══ PART 6: CAPTION ═══
Multi-line caption for social posts. STRUCTURE STRICTLY:
- Line 1 MUST start with the EXACT opener for native_lang:
    * vi: "Luyện từ vựng IELTS" — IMPORTANT: ALWAYS this exact phrase for vi natives (CEO direction)
    * en: "Vocabulary practice"
    * ko: "어휘 연습"
    * ja: "語彙練習"
    * zh: "词汇练习"
    * de: "Vokabeltraining"
    * fr: "Pratique du vocabulaire"
    * es: "Práctica de vocabulario"
    * ru: "Тренировка лексики"
  Then append " — " + topic. E.g. "Luyện từ vựng IELTS — tình yêu".
- Line 2-3: catchy hook in native lang inviting the user to play.
- Line 4: 3-5 hashtags (e.g. "#hocTiengDuc #IELTS #tuvung").
- NO emojis on line 1.
- Total under 280 chars.
"""


def parse_and_generate_guess_word(
    user_text: str,
    *,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> tuple[ParsedIntent, GuessWordContent]:
    """Generate guess_word content (10 words to guess) for a language video."""
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    resp = _call_gemini(client,
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=LANGUAGE_GUESS_WORD_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=GUESS_WORD_SCHEMA,
            temperature=0.7,
        ),
    )
    data = json.loads(resp.text)

    intent_data = data["intent"]
    native_lang = intent_data["native_lang"].lower()
    intent = ParsedIntent(
        target_lang=intent_data["target_lang"].lower(),
        native_lang=native_lang,
        topic=intent_data["topic"],
        count=10,
        voice_gender=(intent_data.get("voice_gender") or "any").lower(),
        target_lang_name=intent_data["target_lang_name"],
        layout_type="guess_word",
    )

    words_raw = data.get("words") or []
    if len(words_raw) != 10:
        raise ValueError(f"guess_word must have exactly 10 words, got {len(words_raw)}")
    words = [GuessWord(**w) for w in words_raw]

    # Defensive: enforce title_native against the lookup if Gemini misspells.
    title_native = data.get("title_native") or _GUESS_TITLE_BY_NATIVE.get(native_lang, "Guess the word")

    # Defensive: rebuild caption opener — Gemini sometimes drifts off the EXACT
    # opener (especially the special vi → "Luyện từ vựng IELTS" rule). Force
    # the first line to be the canonical opener + topic.
    raw_caption = data.get("caption") or ""
    opener = _GUESS_CAPTION_OPENER_BY_NATIVE.get(native_lang, "Vocabulary practice")
    topic_clean = (intent.topic or "").strip()
    caption_lines = raw_caption.split("\n")
    rebuilt_first_line = f"{opener} — {topic_clean}" if topic_clean else opener
    if caption_lines and caption_lines[0].lower().startswith(opener.lower()):
        caption_lines[0] = rebuilt_first_line
    else:
        caption_lines.insert(0, rebuilt_first_line)
    caption = _sanitize_hashtags("\n".join(caption_lines))
    caption = _ensure_seo_hashtag(caption, intent.target_lang)

    content = GuessWordContent(
        title_native=title_native,
        words=words,
        intro_native=data.get("intro_native") or "",
        outro_native=data.get("outro_native") or "",
        topic_label=data.get("topic_label") or "",
        short_title=data.get("short_title") or "",
        short_title_target=data.get("short_title_target") or "",
        caption=caption,
    )
    return intent, content


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    topic = " ".join(sys.argv[1:]) or "tình yêu"
    content = generate(topic)
    print(json.dumps(
        {
            "intro_display": content.intro_display,
            "intro_tts": content.intro_tts,
            "topic_label": content.topic_label,
            "phrases": [p.__dict__ for p in content.phrases],
            "caption": content.caption,
        },
        ensure_ascii=False,
        indent=2,
    ))
