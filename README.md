# 🌍 Lingora

**The auto-pilot content studio for language schools, centers & their social channels.**
*Xưởng content tự động cho trường, trung tâm ngoại ngữ và các kênh social của họ.*

Pick a content type, a language pair, and a topic — Lingora writes the copy, generates the visuals, adds a native-sounding voiceover, and renders a ready-to-post **9:16 MP4** for Facebook Reels, TikTok, Instagram, YouTube Shorts or Zalo. One click, one video.

*Chọn một loại content, một cặp ngôn ngữ và một chủ đề — Lingora viết lời, tạo hình ảnh, lồng tiếng bản ngữ và xuất ra **video dọc 9:16** đăng được ngay lên Facebook Reels, TikTok, Instagram, YouTube Shorts hay Zalo. Một cú click, một video.*

> 🎬 **Web demo (slow, runs on a shared VPS):** https://app.khuetran.com/lingora
> For full speed and unlimited use, clone this and run locally with your own free API keys.

---

## Who it's for / Dành cho ai

- **Language schools & tutoring centers** that need a steady stream of lesson clips but have no in-house video team.
- **Social channel / marketing staff** at a language center who spend hours scripting, designing and editing a single Reel.
- **Independent language teachers & tutors** building an audience one short video at a time.

> **The problem:** everyone knows social content drives enrollment — but between teaching, admin and marketing, nobody has hours a day to script, design, voice and edit a daily video. Lingora is that missing production team.
>
> **Vấn đề:** ai cũng biết content social giúp tuyển sinh — nhưng giữa việc dạy, quản lý và marketing, không ai có vài tiếng mỗi ngày để viết kịch bản, thiết kế, lồng tiếng và dựng một video. Lingora chính là "đội sản xuất" còn thiếu đó.

---

## Content types you can make / Các loại content tạo được

**11 ready-to-post formats.** Mỗi format là một video dọc 9:16, có lồng tiếng, xuất ra 1 file MP4.

### 📚 Lesson clips / Clip bài giảng

| Format | What it makes / Tạo ra gì |
|---|---|
| **Phrases** | 8–10 short sentences on a topic, with voiceover · *8–10 câu ngắn theo chủ đề, có lồng tiếng* |
| **Dialogue** | a 6–8 turn mini-skit between two characters · *hội thoại 6–8 lượt giữa 2 nhân vật (kịch ngắn)* |
| **Vocab card** | one keyword + illustration + a multi-language translation grid · *1 từ khoá + hình minh hoạ + bảng dịch đa ngôn ngữ* |
| **What's this** | 10 vocab items, each with an AI illustration · *10 từ vựng, mỗi từ 1 ảnh minh hoạ AI* |
| **Vocab table** | an 8-item vocab sheet with a character mascot (static poster) · *bảng 8 từ kèm nhân vật mascot (poster tĩnh)* |
| **What's on the board** | a 9-cell cheat-sheet grid · *cheat-sheet dạng lưới 9 ô* |
| **Compare** | 8 side-by-side pairs — great for grammar & false friends (static poster) · *8 cặp so sánh cạnh nhau, hợp cho ngữ pháp / từ dễ nhầm (poster tĩnh)* |

### 🎯 Engagement clips / Clip tương tác (quiz — drive comments & saves · kéo comment & lượt lưu)

| Format | What it makes / Tạo ra gì |
|---|---|
| **Quiz** | one phrase, three options, reveal the answer · *1 câu, 3 lựa chọn, lật đáp án* |
| **Reverse quiz** | show the phrase, guess its meaning · *hiện câu, đoán nghĩa* |
| **Fill-in-the-blank** | a sentence missing one word + 3 options · *câu thiếu 1 từ + 3 lựa chọn* |
| **Guess the word** | reveal the target word letter by letter · *lật từ khoá từng chữ cái* |

**Languages / Ngôn ngữ:** German, Russian, Chinese, Japanese, Korean, English, French, Spanish, Vietnamese (de / ru / zh / ja / ko / en / fr / es / vi) — teach any of them to speakers of any other. *Dạy bất kỳ ngôn ngữ nào cho người nói ngôn ngữ khác.*

---

## English

### Quick start (non-tech: just double-click)

No coding needed — your marketing staff can run it.

**Windows:**
1. Install [Python 3.12+](https://www.python.org/), [Node.js 22+](https://nodejs.org/) and [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) (or `winget install ffmpeg`).
2. Download this repo (green **Code** button → **Download ZIP**) and unzip it.
3. Double-click **`SETUP.bat`** — it checks everything and installs deps.
4. Double-click **`RUN.bat`** — wizard asks content type, language, topic, then renders.

**Mac / Linux:**
```bash
git clone https://github.com/chanktb/lingora.git
cd lingora
./SETUP.sh
./RUN.sh
```

First run creates `channels/myvideo/.env` from the template — open it, paste your **Gemini API key** (free: https://aistudio.google.com/apikey), then run again. The MP4 lands under `channels/myvideo/jobs/<auto-…>/output.mp4`.

**Channels = one config per brand.** Set the school/center name, tagline, language pair and voice once; every video for that channel comes out on-brand. Run one channel per page you manage.

### CLI (non-interactive)
```bash
python run.py --layout phrases --lang de --topic "ordering coffee"
python run.py --layout dialogue --lang ja --topic "asking directions"
python run.py --layout quiz --lang ko --topic "food vocab" --voice female
```
Leave `--topic` off in the wizard and Lingora auto-picks a fresh topic that avoids the last 100 you've already made — so you never run dry.

### Requirements
- **Python 3.12+** and **Node 22+** on PATH (renderer shells out to `npx hyperframes`).
- **ffmpeg** on PATH.
- **Free Gemini API key** — https://aistudio.google.com/apikey. Add several keys (comma-separated) and they auto-rotate when one hits its daily quota.
- **Optional:** Cloudflare Workers AI account (`CLOUDFLARE_ACCOUNTS`) for AI scene/character images. Formats that don't need images still render fine without it.

### License
MIT — see [LICENSE](LICENSE). Free to use, modify and share.

### Author
Built by **Khue Tran** · https://khuetran.com
Running a whole content calendar? The engine also supports auto-render + auto-post (FB Reels / Stories / TikTok) on a schedule — intentionally off in this open build. Reach out via the site to set it up for your center.

---

## Tiếng Việt

### Bắt đầu nhanh (không cần biết code)

Không cần lập trình — nhân viên marketing của trung tâm chạy được ngay.

**Windows:**
1. Cài [Python 3.12+](https://www.python.org/), [Node.js 22+](https://nodejs.org/) và [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) (hoặc `winget install ffmpeg`).
2. Tải repo này (nút xanh **Code** → **Download ZIP**) rồi giải nén.
3. Click đôi **`SETUP.bat`** — script kiểm tra môi trường và cài deps.
4. Click đôi **`RUN.bat`** — wizard hỏi loại content, ngôn ngữ, chủ đề, rồi render.

**Mac / Linux:**
```bash
git clone https://github.com/chanktb/lingora.git
cd lingora
./SETUP.sh
./RUN.sh
```

Lần đầu chạy sẽ tạo `channels/myvideo/.env` từ template — mở file ra, dán **Gemini API key** (miễn phí: https://aistudio.google.com/apikey), rồi chạy lại. File MP4 nằm trong `channels/myvideo/jobs/<auto-…>/output.mp4`.

**Kênh (channel) = một cấu hình cho mỗi thương hiệu.** Đặt tên trung tâm, tagline, cặp ngôn ngữ và giọng đọc một lần; mọi video của kênh đó ra đúng nhận diện. Mỗi page bạn quản lý thì tạo một kênh.

### CLI (không tương tác)
```bash
python run.py --layout phrases --lang de --topic "gọi cà phê"
python run.py --layout dialogue --lang ja --topic "hỏi đường"
python run.py --layout quiz --lang ko --topic "đồ ăn" --voice female
```
Bỏ trống `--topic` trong wizard, Lingora tự chọn một chủ đề mới, tránh 100 chủ đề bạn đã làm gần nhất — nên không bao giờ bí ý tưởng.

### Yêu cầu
- **Python 3.12+** và **Node 22+** trong PATH (renderer dùng `npx hyperframes`).
- **ffmpeg** trong PATH.
- **Gemini API key miễn phí** — https://aistudio.google.com/apikey. Thêm nhiều key (cách nhau dấu phẩy) để tự xoay vòng khi 1 key hết quota ngày.
- **Tuỳ chọn:** tài khoản Cloudflare Workers AI (`CLOUDFLARE_ACCOUNTS`) để sinh ảnh AI. Các loại content không cần ảnh vẫn render được nếu thiếu.

### Giấy phép
MIT — xem [LICENSE](LICENSE). Tự do dùng, sửa, chia sẻ.

### Tác giả
**Khue Tran** · https://khuetran.com
Cần chạy cả lịch content? Engine còn hỗ trợ tự render + tự đăng (FB Reels / Stories / TikTok) theo lịch — cố tình tắt trong bản open-source này. Liên hệ qua website nếu muốn setup cho trung tâm của bạn.

---

> 🌐 Web demo (chạy chậm, host VPS chung): https://app.khuetran.com/lingora
