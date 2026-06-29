# Static assets

Đặt vào đây các file mà mọi video đều dùng chung.

## `logo.png`

Logo channel, hiển thị trong vòng tròn avatar góc trên trái video.

- **Định dạng:** PNG (nên có nền trong suốt)
- **Kích thước:** ≥ 256×256 px (vuông), bot tự co về 130×130 trong video
- **Đặt tên chính xác:** `logo.png` (lowercase)

Nếu file này tồn tại → bot dùng làm avatar.
Nếu không → fallback emoji `AVATAR_EMOJI` trong `.env`.

Sau khi thả logo vào đây, bấm đúp `tools/03-update-video-bot.bat` để đồng bộ lên VPS.
