# 🌍 Lingora

**Turn a topic into a polished 9:16 language-learning video — one click, one MP4.**
*Biến một chủ đề thành video dạy ngoại ngữ 9:16 chỉn chu — một cú click, một file MP4.*

10 layouts (phrases, dialogue, quiz, vocab grid, fill-blank, compare, what's-this, what's-on-the-board, guess-word, board), Edge TTS voiceover, optional Cloudflare FLUX scene/character images, multi-language (de / ru / zh / ja / ko / en / vi / fr / es / …).

> 🎬 **Web demo (slow, runs on a shared VPS):** https://app.khuetran.com/lingora
> For full speed and unlimited use, clone this and run locally with your own free API keys.

---

## English

### Quick start (non-tech: just double-click)

**Windows:**
1. Install [Python 3.12+](https://www.python.org/), [Node.js 22+](https://nodejs.org/) and [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) (or `winget install ffmpeg`).
2. Download this repo (green **Code** button → **Download ZIP**) and unzip it.
3. Double-click **`SETUP.bat`** — it checks everything and installs deps.
4. Double-click **`RUN.bat`** — wizard asks layout, language, topic, then renders.

**Mac / Linux:**
```bash
git clone https://github.com/chanktb/lingora.git
cd lingora
./SETUP.sh
./RUN.sh
```

First run creates `channels/myvideo/.env` from the template — open it, paste your **Gemini API key** (free: https://aistudio.google.com/apikey), then run again. The MP4 lands under `channels/myvideo/jobs/<auto-…>/output.mp4`.

### CLI (non-interactive)
```bash
python run.py --layout phrases --lang de --topic "ordering coffee"
python run.py --layout dialogue --lang ja --topic "asking directions"
python run.py --layout quiz --lang ko --topic "food vocab" --voice female
```

### Requirements
- **Python 3.12+** and **Node 22+** on PATH (renderer shells out to `npx hyperframes`).
- **ffmpeg** on PATH.
- **Free Gemini API key** — https://aistudio.google.com/apikey.
- **Optional:** Cloudflare Workers AI account (`CLOUDFLARE_ACCOUNTS`) for AI scene/character images. Layouts that don't need images still render fine without it.

### License
MIT — see [LICENSE](LICENSE). Free to use, modify and share.

### Author
Built by **Khue Tran** · https://khuetran.com
Need auto-render + auto-post (FB Reels / Stories) at scale? The engine supports it but is intentionally not wired up in this build. Reach out via the site.

---

## Tiếng Việt

### Bắt đầu nhanh (không cần biết code)

**Windows:**
1. Cài [Python 3.12+](https://www.python.org/), [Node.js 22+](https://nodejs.org/) và [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) (hoặc `winget install ffmpeg`).
2. Tải repo này (nút xanh **Code** → **Download ZIP**) rồi giải nén.
3. Click đôi **`SETUP.bat`** — script kiểm tra môi trường và cài deps.
4. Click đôi **`RUN.bat`** — wizard hỏi layout, ngôn ngữ, chủ đề, rồi render.

**Mac / Linux:**
```bash
git clone https://github.com/chanktb/lingora.git
cd lingora
./SETUP.sh
./RUN.sh
```

Lần đầu chạy sẽ tạo `channels/myvideo/.env` từ template — mở file ra, dán **Gemini API key** (miễn phí: https://aistudio.google.com/apikey), rồi chạy lại. File MP4 nằm trong `channels/myvideo/jobs/<auto-…>/output.mp4`.

### CLI (không tương tác)
```bash
python run.py --layout phrases --lang de --topic "gọi cà phê"
python run.py --layout dialogue --lang ja --topic "hỏi đường"
python run.py --layout quiz --lang ko --topic "đồ ăn" --voice female
```

### Yêu cầu
- **Python 3.12+** và **Node 22+** trong PATH (renderer dùng `npx hyperframes`).
- **ffmpeg** trong PATH.
- **Gemini API key miễn phí** — https://aistudio.google.com/apikey.
- **Tuỳ chọn:** tài khoản Cloudflare Workers AI (`CLOUDFLARE_ACCOUNTS`) để sinh ảnh AI. Các layout không cần ảnh vẫn render được nếu thiếu.

### Giấy phép
MIT — xem [LICENSE](LICENSE). Tự do dùng, sửa, chia sẻ.

### Tác giả
**Khue Tran** · https://khuetran.com
Cần bản tự render + tự đăng (FB Reels / Stories) theo lịch? Engine có hỗ trợ nhưng cố tình không bật trong bản này. Liên hệ qua website nếu cần setup.

---

> 🌐 Web demo (chạy chậm, host VPS chung): https://app.khuetran.com/lingora
