"""Topic + layout picker for auto-post pipeline.

Maintains a list of topics + quiz questions per niche, picks one that hasn't
been used recently. Alternates between phrases and special layouts.

Layout cycle (language niche, 10-state):
  phrases → quiz_forward (VN→target) → phrases → quiz_reverse (target→VN) →
  phrases → whats_this (visual vocab) → phrases → whats_board (9-grid) →
  phrases → dialogue (2-char skit) → repeat

ktb-lingora addition (vs production telegram-video-bot):
  Channels can declare DEFAULT_NATIVE_LANG=en (e.g. Russian Path teaches
  Russian to English speakers) or DEFAULT_NATIVE_LANG=vi (default — e.g.
  Tieng Nhat 5 Phut teaches Japanese to Vietnamese speakers). The format
  helpers below switch the request wording so Gemini parses native_lang
  correctly per channel.
"""
from __future__ import annotations

import random

# Module-level native language for the current pick. auto_post sets this via
# pick_next_request(native_lang=...). Defaults to "vi" for backwards-compat
# with production telegram-video-bot pools.
_NATIVE_LANG: str = "vi"


# ─── LANGUAGE: phrases topics ──────────────────────────────────────────
# Per-language phrase topic pools. Each item = a "request" topic.
# Audience varies by language — see comments per pool.
LANGUAGE_TOPICS = {
    # ─── Đức: giao tiếp hàng ngày A1-B2, chủ đề đời thường ─────
    "de": [
        # Chào hỏi & Giới thiệu
        "tự giới thiệu bản thân bằng tiếng Đức",
        "hỏi thăm sức khỏe và chào hỏi",
        "giới thiệu gia đình với bạn bè người Đức",
        "nói về tuổi tác và ngày sinh",
        # Gia đình & Các mối quan hệ
        "nói về gia đình và người thân",
        "mô tả tính cách của bạn bè",
        "mời bạn bè đến nhà chơi",
        "nói chuyện với hàng xóm",
        # Số đếm / Thời gian / Lịch
        "nói về ngày tháng và lịch hẹn",
        "hỏi và trả lời về giờ giấc",
        "đặt hẹn gặp bạn vào cuối tuần",
        "nói về kế hoạch tuần tới",
        # Thức ăn & Nhà hàng
        "đặt món tại nhà hàng Đức",
        "gọi đồ uống tại quán cafe",
        "hỏi về thực đơn và giá tiền",
        "khen ngợi món ăn ngon",
        "nói về sở thích ăn uống",
        # Mua sắm
        "mua sắm tại siêu thị Đức",
        "hỏi giá và trả giá tại cửa hàng",
        "đổi trả sản phẩm không vừa ý",
        "mua quần áo tại trung tâm mua sắm",
        "hỏi đường đến cửa hàng",
        # Phương tiện & Đi lại
        "mua vé tàu và hỏi giờ chạy",
        "đi tàu điện và hỏi hướng đi",
        "gọi taxi và chỉ địa chỉ",
        "hỏi đường đến địa điểm cần tìm",
        "thuê xe đạp trong thành phố",
        # Du lịch & Khách sạn
        "nhận phòng và trả phòng khách sạn",
        "hỏi thông tin du lịch tại Đức",
        "mua vé tham quan bảo tàng",
        "đặt phòng khách sạn qua điện thoại",
        "hỏi về các điểm tham quan nổi tiếng",
        # Thời tiết & Mùa
        "nói về thời tiết hôm nay",
        "lên kế hoạch theo thời tiết",
        "mô tả các mùa trong năm ở Đức",
        # Nhà cửa & Nội thất
        "mô tả căn phòng và đồ nội thất",
        "thuê căn hộ và ký hợp đồng",
        "nói về công việc nhà hàng ngày",
        "mời bạn về nhà ăn tối",
        # Sức khỏe & Bác sĩ
        "đặt lịch khám bác sĩ",
        "mô tả triệu chứng ốm cho bác sĩ",
        "mua thuốc tại nhà thuốc",
        "hỏi về cách uống thuốc",
        "nói về thói quen sức khỏe",
        # Sở thích & Thời gian rảnh
        "nói về sở thích và hobby",
        "rủ bạn đi xem phim cuối tuần",
        "nói về môn thể thao yêu thích",
        "kể về chuyến đi du lịch cuối tuần",
        "nói về âm nhạc và phim ảnh yêu thích",
        # Công việc & Văn phòng (tổng quát)
        "phỏng vấn xin việc cơ bản",
        "tự giới thiệu trong ngày đầu đi làm",
        "nói chuyện với đồng nghiệp lúc nghỉ",
        "xin nghỉ phép và báo ốm",
        "nói về lịch làm việc hàng tuần",
        # Điện thoại & Công nghệ
        "mua SIM điện thoại mới",
        "hỏi về mạng wifi và internet",
        "nói về mạng xã hội yêu thích",
        "nhờ người chụp ảnh hộ",
        # Cảm xúc & Ý kiến
        "bày tỏ cảm xúc vui buồn",
        "nêu ý kiến về một bộ phim",
        "đồng ý và không đồng ý lịch sự",
        "khen ngợi và cảm ơn người khác",
        # Văn hóa & Lễ hội
        "nói về lễ hội Giáng sinh Đức",
        "hỏi về phong tục đón năm mới Đức",
        "nói về ngày lễ và nghỉ phép",
    ],
    # ─── Trung: giao tiếp hàng ngày A1-B2, chủ đề đời thường ────
    "zh": [
        # Chào hỏi & Giới thiệu
        "tự giới thiệu bản thân bằng tiếng Trung",
        "chào hỏi và hỏi thăm sức khỏe",
        "giới thiệu gia đình và bạn bè",
        "nói về quê quán và nơi sinh sống",
        # Gia đình & Quan hệ
        "nói về các thành viên trong gia đình",
        "mô tả tính cách và ngoại hình",
        "nói chuyện với bạn bè về cuộc sống",
        "mời bạn đến nhà ăn tối",
        # Số đếm / Tiền / Thời gian
        "nói về ngày tháng và giờ giấc",
        "hỏi và trả lời về giá tiền",
        "đặt lịch hẹn gặp bạn",
        "nói về kế hoạch trong tuần",
        # Thức ăn & Nhà hàng
        "gọi món tại nhà hàng Trung Quốc",
        "hỏi về thực đơn và giá cả",
        "khen ngợi món ăn ngon",
        "đặt bàn nhà hàng qua điện thoại",
        "nói về sở thích ăn uống",
        "mua đồ ăn sáng tại quán vỉa hè",
        # Mua sắm
        "mua sắm tại chợ hoặc trung tâm thương mại",
        "hỏi giá và thương lượng tại cửa hàng",
        "đổi trả hàng không vừa ý",
        "mua quần áo và hỏi size",
        "thanh toán bằng ứng dụng di động",
        # Phương tiện & Đi lại
        "mua vé tàu và hỏi giờ khởi hành",
        "đi tàu điện ngầm và hỏi hướng đi",
        "bắt taxi và chỉ địa chỉ",
        "hỏi đường đến địa điểm cần tìm",
        "đặt xe qua ứng dụng điện thoại",
        # Du lịch & Khách sạn
        "đặt phòng và nhận phòng khách sạn",
        "hỏi thông tin điểm du lịch",
        "mua vé tham quan và check-in",
        "nói về chuyến du lịch vừa rồi",
        "hỏi về đặc sản địa phương",
        # Thời tiết & Mùa
        "nói về thời tiết hôm nay",
        "mô tả các mùa và thời tiết Trung Quốc",
        "lên kế hoạch dựa theo thời tiết",
        # Nhà cửa & Nội thất
        "mô tả căn phòng và đồ đạc",
        "thuê nhà và hỏi về điều kiện",
        "nói về việc nhà hàng ngày",
        "mời hàng xóm sang chơi",
        # Sức khỏe & Bác sĩ
        "đặt lịch khám bác sĩ",
        "mô tả triệu chứng khi ốm",
        "mua thuốc tại hiệu thuốc",
        "hỏi về chế độ ăn uống lành mạnh",
        "nói về thói quen tập thể dục",
        # Sở thích & Thời gian rảnh
        "nói về sở thích cá nhân",
        "rủ bạn xem phim hoặc hòa nhạc",
        "nói về môn thể thao yêu thích",
        "kể về chuyến du lịch gần đây",
        "nói về âm nhạc và phim yêu thích",
        # Công việc & Văn phòng (tổng quát)
        "phỏng vấn xin việc cơ bản",
        "tự giới thiệu trong ngày đầu đi làm",
        "nói chuyện với đồng nghiệp lúc nghỉ trưa",
        "xin nghỉ phép và báo ốm",
        "nói về công việc và lịch làm việc",
        # Điện thoại & Công nghệ
        "hỏi về mạng wifi và mật khẩu",
        "nói về ứng dụng điện thoại yêu thích",
        "nhờ chụp ảnh tại điểm du lịch",
        "nói về mạng xã hội Trung Quốc",
        # Cảm xúc & Ý kiến
        "bày tỏ cảm xúc vui buồn",
        "nêu ý kiến về phim hoặc sách",
        "đồng ý và không đồng ý lịch sự",
        "khen ngợi và cảm ơn người khác",
        # Văn hóa & Lễ hội
        "nói về Tết Nguyên Đán Trung Quốc",
        "hỏi về phong tục lễ hội Trung Quốc",
        "nói về ẩm thực vùng miền Trung Quốc",
    ],
    # ─── Hàn: giao tiếp hàng ngày A1-B2, chủ đề đời thường ──────
    "ko": [
        # Chào hỏi & Giới thiệu
        "tự giới thiệu bản thân bằng tiếng Hàn",
        "chào hỏi và hỏi thăm sức khỏe",
        "giới thiệu gia đình và bạn bè",
        "nói về quê quán và nơi sống",
        # Gia đình & Quan hệ
        "nói về các thành viên trong gia đình",
        "mô tả tính cách và ngoại hình bạn bè",
        "rủ bạn đi chơi cuối tuần",
        "nhờ người giúp đỡ lịch sự",
        # Số đếm / Tiền / Thời gian
        "nói về ngày tháng và giờ giấc",
        "hỏi và trả lời về giá tiền mua sắm",
        "đặt lịch hẹn gặp bạn",
        "nói về kế hoạch cuối tuần",
        # Thức ăn & Nhà hàng
        "đặt món tại nhà hàng Hàn Quốc",
        "gọi đồ uống tại quán cafe",
        "hỏi về thực đơn và giá cả",
        "đặt giao đồ ăn qua ứng dụng",
        "nói về món Hàn yêu thích",
        "rủ bạn đi ăn tối cùng",
        # Mua sắm
        "mua sắm tại siêu thị hoặc trung tâm thương mại",
        "hỏi giá và thử đồ tại cửa hàng quần áo",
        "đổi trả hàng không vừa ý",
        "thanh toán và hỏi về khuyến mãi",
        "mua đồ tại cửa hàng tiện lợi",
        # Phương tiện & Đi lại
        "đi tàu điện ngầm và hỏi đường",
        "đi xe buýt và hỏi điểm dừng",
        "bắt taxi và chỉ địa chỉ",
        "hỏi đường đến địa điểm cần tìm",
        "thuê xe đạp hoặc xe máy",
        # Du lịch & Khách sạn
        "đặt phòng và nhận phòng khách sạn",
        "hỏi thông tin điểm du lịch ở Hàn",
        "mua vé tham quan và đặt tour",
        "nói về chuyến du lịch vừa qua",
        "hỏi về đặc sản địa phương",
        # Thời tiết & Mùa
        "nói về thời tiết và dự báo",
        "mô tả các mùa đẹp ở Hàn Quốc",
        "lên kế hoạch dựa theo thời tiết",
        # Nhà cửa & Nội thất
        "mô tả phòng ở và đồ đạc",
        "thuê nhà và hỏi điều kiện",
        "nói về việc nhà hàng ngày",
        "mời bạn bè đến nhà chơi",
        # Sức khỏe & Bác sĩ
        "đặt lịch khám bác sĩ",
        "mô tả triệu chứng ốm cho bác sĩ",
        "mua thuốc tại nhà thuốc",
        "hỏi về thói quen ăn uống lành mạnh",
        "nói về tập thể dục và vận động",
        # Sở thích & Thời gian rảnh
        "nói về sở thích và thời gian rảnh",
        "rủ bạn xem phim hoặc đi concert",
        "nói về môn thể thao yêu thích",
        "kể về chuyến du lịch gần đây",
        "nói về K-pop và K-drama yêu thích",
        # Công việc & Văn phòng (tổng quát)
        "phỏng vấn xin việc cơ bản",
        "tự giới thiệu trong ngày đầu đi làm",
        "nói chuyện với đồng nghiệp lúc nghỉ",
        "xin nghỉ phép và báo ốm",
        "nói về lịch làm việc và giờ tan sở",
        # Điện thoại & Công nghệ
        "hỏi về mạng wifi và mật khẩu",
        "nói về ứng dụng điện thoại yêu thích",
        "nhờ chụp ảnh tại điểm du lịch",
        "nói về mạng xã hội và giải trí online",
        # Cảm xúc & Ý kiến
        "bày tỏ cảm xúc vui buồn",
        "nêu ý kiến về phim hoặc ca nhạc",
        "đồng ý và không đồng ý lịch sự",
        "khen ngợi và cảm ơn người khác",
        # Văn hóa & Lễ hội
        "nói về Tết Chuseok và lễ hội Hàn",
        "hỏi về phong tục và văn hóa Hàn",
        "nói về ẩm thực truyền thống Hàn Quốc",
    ],
    # ─── Russian (native=en, general beginner — Russian Path) ─────────────
    # Channel vision: GENERAL language learning. Basic everyday topics,
    # daily conversation, beginner-friendly. NO niche audience targeting.
    "ru": [
        # Greetings & introductions
        "saying hello and goodbye in Russian",
        "introducing yourself to a new friend",
        "asking how someone is doing",
        "asking where someone is from",
        "telling someone your age",
        "introducing your family in Russian",
        "polite phrases for first meetings",
        # Numbers, time & dates
        "counting from one to ten in Russian",
        "telling the time in Russian",
        "days of the week",
        "months of the year",
        "talking about your birthday",
        "asking what day it is today",
        # Basic daily life
        "ordering coffee at a cafe",
        "ordering food at a restaurant",
        "asking for the bill in Russian",
        "buying bread at a bakery",
        "shopping for groceries",
        "asking 'how much does it cost'",
        "trying on clothes at a shop",
        "paying with cash or by card",
        # Asking for help & directions
        "asking for directions on the street",
        "asking where the toilet is",
        "asking for help when you're lost",
        "asking someone to repeat slowly",
        "saying 'I don't understand'",
        "asking for the wifi password",
        # Travel
        "checking in at a hotel reception",
        "asking for the airport in Russian",
        "taking a taxi to your hotel",
        "buying a metro ticket",
        "asking for a train platform",
        "ordering room service",
        # Food, drink & weather
        "saying you're hungry or thirsty",
        "talking about your favourite food",
        "ordering a beer or tea",
        "talking about the weather today",
        "saying it's cold or hot outside",
        # Health basics
        "saying you don't feel well",
        "asking for a pharmacy nearby",
        "describing a simple headache",
        "calling for emergency help",
        # Hobbies & free time
        "talking about your hobbies",
        "saying what music you like",
        "talking about a movie you watched",
        "inviting a friend to dinner",
        "making weekend plans",
        # Family, friends & feelings
        "saying you miss someone",
        "wishing a friend happy birthday",
        "saying thank you in different ways",
        "apologising politely in Russian",
        "expressing love or affection",
        "complimenting someone's outfit",
        # Polite small talk
        "small talk in the elevator",
        "talking about your day at work",
        "asking if someone speaks English",
        "saying goodbye for the night",
        # Practical phrases learners need first
        "saying 'yes' and 'no' politely",
        "the 20 most useful Russian phrases",
        "common Russian phrases you'll hear daily",
    ],
    # ─── Japanese (native=vi, general beginner — Tiếng Nhật 5 Phút) ─────
    # Tầm nhìn: học tiếng Nhật cơ bản, giao tiếp hàng ngày — KHÔNG niche.
    "ja": [
        # Chào hỏi & giới thiệu
        "chào hỏi cơ bản (xin chào, tạm biệt)",
        "tự giới thiệu bản thân",
        "hỏi tên người khác",
        "hỏi ai đó từ đâu đến",
        "nói tuổi của bạn bằng tiếng Nhật",
        "giới thiệu gia đình",
        "câu lịch sự khi gặp lần đầu",
        # Số đếm, thời gian, ngày tháng
        "đếm số 1-10 bằng tiếng Nhật",
        "đếm số 100-1000",
        "nói giờ bằng tiếng Nhật",
        "các thứ trong tuần",
        "các tháng trong năm",
        "nói về sinh nhật của bạn",
        "hỏi hôm nay là thứ mấy",
        # Đời sống hàng ngày
        "gọi cà phê tại quán",
        "gọi món tại nhà hàng",
        "xin hoá đơn",
        "mua bánh tại tiệm bánh",
        "đi siêu thị mua đồ",
        "hỏi giá tiền",
        "thử quần áo tại cửa hàng",
        "thanh toán bằng tiền mặt hay thẻ",
        # Hỏi đường & xin trợ giúp
        "hỏi đường trên phố",
        "hỏi nhà vệ sinh ở đâu",
        "xin trợ giúp khi bị lạc",
        "nhờ ai đó nói chậm lại",
        "nói 'tôi không hiểu'",
        "xin mật khẩu wifi",
        # Du lịch
        "check-in khách sạn",
        "hỏi đường ra sân bay",
        "gọi taxi về khách sạn",
        "mua vé tàu điện",
        "hỏi sân ga tàu",
        "gọi room service",
        # Ăn uống & thời tiết
        "nói bạn đói hay khát",
        "nói về món ăn yêu thích",
        "gọi bia hoặc trà",
        "nói về thời tiết hôm nay",
        "nói trời lạnh hay nóng",
        # Sức khoẻ cơ bản
        "nói bạn không khoẻ",
        "hỏi tiệm thuốc gần nhất",
        "nói bị đau đầu",
        "gọi cấp cứu trong tình huống khẩn",
        # Sở thích & rảnh rỗi
        "nói về sở thích của bạn",
        "nói loại nhạc bạn thích",
        "nói về phim bạn vừa xem",
        "rủ bạn đi ăn tối",
        "lên kế hoạch cuối tuần",
        # Gia đình, bạn bè, cảm xúc
        "nói bạn nhớ ai đó",
        "chúc sinh nhật bạn bè",
        "nhiều cách nói cảm ơn",
        "xin lỗi lịch sự",
        "thể hiện tình cảm",
        "khen ngợi trang phục của ai đó",
        # Small talk lịch sự
        "small talk trong thang máy",
        "nói về một ngày làm việc",
        "hỏi ai đó có nói tiếng Anh không",
        "chúc ngủ ngon",
        # Câu thực dụng beginner cần đầu tiên
        "nói 'có' và 'không' lịch sự",
        "20 câu tiếng Nhật hữu ích nhất",
        "những câu tiếng Nhật bạn nghe hàng ngày",
        # Văn hoá cơ bản (general, không hardcore XKLĐ)
        "đi onsen lần đầu",
        "đi izakaya cùng bạn",
        "đi xem hoa anh đào",
        "đi siêu thị Aeon",
    ],
}


# ─── LANGUAGE: quiz questions ──────────────────────────────────────────
# Quiz topics xoay quanh: thủ tục, vocab thực dụng cho người Việt tại Đức
LANGUAGE_QUIZ_QUESTIONS = {
    "de": [
        # Chào hỏi & Giao tiếp cơ bản
        "xin chào", "tạm biệt", "cảm ơn", "xin lỗi",
        "không có gì", "tôi không hiểu",
        "nói chậm hơn được không", "có thể nhắc lại không",
        "tôi đến từ Việt Nam", "tôi đang học tiếng Đức",
        "rất vui được gặp bạn", "hẹn gặp lại",
        # Gia đình & Con người
        "bố mẹ", "anh chị em", "bạn bè", "hàng xóm",
        "trẻ em", "người già", "vợ chồng",
        # Số đếm & Thời gian
        "ngày hôm nay", "ngày mai", "tuần này", "tháng trước",
        "mấy giờ rồi", "hôm nay thứ mấy", "sinh nhật",
        # Thức ăn & Đồ uống
        "bánh mì", "thịt heo", "rau củ", "trái cây",
        "cà phê", "trà", "nước lọc", "bia",
        "ngon quá", "no rồi", "còn đói",
        # Mua sắm
        "siêu thị", "cửa hàng", "bao nhiêu tiền",
        "đắt quá", "rẻ hơn được không", "giảm giá",
        "thanh toán", "hóa đơn", "tiền lẻ",
        # Giao thông & Đi lại
        "sân bay", "ga tàu", "tàu điện", "xe buýt",
        "taxi", "đi bộ", "rẽ phải", "rẽ trái", "đi thẳng",
        "bao xa", "bao lâu đến nơi",
        # Nhà cửa
        "phòng ngủ", "phòng bếp", "phòng tắm", "phòng khách",
        "thuê nhà", "tầng mấy", "chìa khóa",
        # Y tế & Sức khỏe
        "bệnh viện", "bác sĩ", "đơn thuốc", "nhà thuốc",
        "bảo hiểm y tế", "đau đầu", "sốt", "ho",
        # Công việc & Học tập (tổng quát)
        "đi làm", "đi học", "công ty", "trường học",
        "đồng nghiệp", "sếp", "giáo viên", "lớp học",
        "nghỉ phép", "bài tập về nhà",
        # Sở thích & Giải trí
        "xem phim", "nghe nhạc", "đọc sách", "chơi thể thao",
        "đi du lịch", "nấu ăn", "vẽ tranh", "chụp ảnh",
        # Thời tiết
        "nắng", "mưa", "lạnh", "nóng", "gió", "tuyết",
        # Tính cách & Cảm xúc
        "vui vẻ", "buồn", "mệt mỏi", "hào hứng",
        "thân thiện", "lịch sự", "kiên nhẫn",
    ],
    "zh": [
        # Chào hỏi & Giao tiếp cơ bản
        "xin chào", "tạm biệt", "cảm ơn", "xin lỗi",
        "không có gì", "tôi không hiểu",
        "nói chậm hơn được không", "có thể nhắc lại không",
        "tôi đến từ Việt Nam", "tôi đang học tiếng Trung",
        "rất vui được gặp bạn", "chúc sức khỏe",
        # Gia đình & Con người
        "bố mẹ", "anh chị em", "bạn bè", "hàng xóm",
        "trẻ em", "ông bà", "vợ chồng",
        # Số đếm & Thời gian
        "ngày hôm nay", "ngày mai", "tuần này", "tháng trước",
        "mấy giờ rồi", "hôm nay thứ mấy", "sinh nhật",
        "một trăm", "một nghìn", "một triệu",
        # Thức ăn & Đồ uống
        "cơm trắng", "mì sợi", "rau xào", "thịt heo",
        "trà", "nước lọc", "trà sữa", "bia",
        "ngon quá", "no rồi", "đói bụng",
        # Mua sắm
        "siêu thị", "chợ", "cửa hàng", "bao nhiêu tiền",
        "đắt quá", "rẻ hơn được không", "giảm giá",
        "thanh toán", "hóa đơn", "tiền lẻ",
        # Giao thông & Đi lại
        "sân bay", "ga tàu", "tàu điện ngầm", "xe buýt",
        "taxi", "đi bộ", "rẽ phải", "rẽ trái", "đi thẳng",
        "bao xa", "bao lâu đến nơi",
        # Nhà cửa
        "phòng ngủ", "phòng bếp", "phòng tắm", "phòng khách",
        "thuê nhà", "tầng mấy", "chìa khóa",
        # Y tế & Sức khỏe
        "bệnh viện", "bác sĩ", "đơn thuốc", "nhà thuốc",
        "bảo hiểm y tế", "đau đầu", "sốt", "ho",
        # Công việc & Học tập (tổng quát)
        "đi làm", "đi học", "công ty", "trường học",
        "đồng nghiệp", "sếp", "giáo viên", "nghỉ phép",
        # Sở thích & Giải trí
        "xem phim", "nghe nhạc", "đọc sách", "chơi thể thao",
        "đi du lịch", "nấu ăn", "chụp ảnh",
        "lễ hội", "múa rồng", "pháo hoa",
        # Thời tiết
        "nắng", "mưa", "lạnh", "nóng", "gió", "tuyết",
        # Tính cách & Cảm xúc
        "vui vẻ", "buồn", "mệt mỏi", "hào hứng",
        "thân thiện", "lịch sự", "kiên nhẫn",
        # Màu sắc & Mô tả
        "màu đỏ", "màu xanh", "màu vàng", "màu trắng",
        "to lớn", "nhỏ bé", "đẹp", "xấu",
    ],
    "ko": [
        # Chào hỏi & Giao tiếp cơ bản
        "xin chào", "tạm biệt", "cảm ơn", "xin lỗi",
        "không có gì", "tôi không hiểu",
        "nói chậm hơn được không", "có thể nhắc lại không",
        "tôi đến từ Việt Nam", "tôi đang học tiếng Hàn",
        "rất vui được gặp bạn", "chúc ngủ ngon",
        # Gia đình & Con người
        "bố mẹ", "anh chị em", "bạn bè", "hàng xóm",
        "trẻ em", "ông bà", "vợ chồng",
        # Số đếm & Thời gian
        "ngày hôm nay", "ngày mai", "tuần này", "tháng trước",
        "mấy giờ rồi", "hôm nay thứ mấy", "sinh nhật",
        # Thức ăn & Đồ uống
        "cơm", "kimchi", "canh rong biển", "thịt nướng",
        "trà", "nước lọc", "cafe", "sữa",
        "ngon quá", "no rồi", "đói bụng",
        # Mua sắm
        "siêu thị", "cửa hàng tiện lợi", "bao nhiêu tiền",
        "đắt quá", "rẻ hơn được không", "giảm giá",
        "thanh toán thẻ", "tiền mặt",
        # Giao thông & Đi lại
        "tàu điện ngầm", "xe buýt", "taxi", "đi bộ",
        "rẽ phải", "rẽ trái", "đi thẳng",
        "bao xa", "bao lâu đến nơi", "thẻ giao thông",
        # Nhà cửa
        "phòng ngủ", "phòng bếp", "phòng tắm", "phòng khách",
        "thuê nhà", "tầng mấy", "chìa khóa",
        # Y tế & Sức khỏe
        "bệnh viện", "bác sĩ", "đơn thuốc", "nhà thuốc",
        "bảo hiểm y tế", "đau đầu", "sốt", "ho",
        # Công việc & Học tập (tổng quát)
        "đi làm", "đi học", "công ty", "trường học",
        "đồng nghiệp", "sếp", "giáo viên", "nghỉ phép",
        # Sở thích & Giải trí
        "xem phim", "nghe nhạc", "đọc sách", "chơi thể thao",
        "đi du lịch", "nấu ăn", "chụp ảnh",
        "K-pop", "K-drama", "hoa anh đào",
        # Thời tiết
        "nắng", "mưa", "lạnh", "nóng", "gió", "tuyết",
        # Tính cách & Cảm xúc
        "vui vẻ", "buồn", "mệt mỏi", "hào hứng",
        "thân thiện", "lịch sự", "chăm chỉ",
    ],
    # ─── Russian quiz (native=en, general beginner) ─────────────────────
    "ru": [
        # Greetings & politeness
        "hello", "good morning", "good evening", "good night", "goodbye",
        "thank you", "thank you very much", "please", "you're welcome",
        "excuse me", "I'm sorry",
        # Yes / no / common reactions
        "yes", "no", "maybe", "of course", "great", "I don't know",
        # Introductions
        "what's your name", "my name is", "nice to meet you",
        "I am from England", "I am from America",
        "I am a student", "I am a teacher",
        # Time & numbers
        "what time is it", "today", "tomorrow", "yesterday",
        "one two three", "ten twenty thirty", "one hundred",
        "morning", "afternoon", "evening", "midnight",
        # Money & shopping
        "how much does it cost", "too expensive", "is there a discount",
        "I'll take it", "cash or card",
        # Food & drink
        "I would like coffee", "I would like tea", "water please",
        "the menu please", "the bill please", "delicious",
        "I am hungry", "I am thirsty", "I am vegetarian",
        # Travel & directions
        "where is the metro", "where is the toilet",
        "I am lost", "left", "right", "straight",
        "a taxi please", "to the airport please",
        # Hotel
        "I have a reservation", "my room key",
        "wifi password", "check-out time",
        # Polite small talk
        "how are you", "I'm fine", "and you",
        "I don't understand", "could you speak more slowly",
        "I am learning Russian", "help me please",
        # Health
        "I don't feel well", "I have a headache",
        "call an ambulance", "where is the pharmacy",
        # Feelings
        "I love you", "I miss you", "I'm tired",
        "I'm happy", "happy birthday",
    ],
    # ─── Japanese quiz (native=vi, general beginner) ────────────────────
    "ja": [
        # Lịch sự / Chào hỏi
        "xin chào", "chào buổi sáng", "chào buổi tối", "tạm biệt", "ngủ ngon",
        "cảm ơn", "cảm ơn rất nhiều", "xin lỗi", "không có gì",
        "rất vui được gặp",
        # Yes / no / common reactions
        "có", "không", "có lẽ", "tất nhiên", "tuyệt", "tôi không biết",
        # Giới thiệu
        "bạn tên là gì", "tôi tên là", "rất vui được gặp",
        "tôi đến từ Việt Nam", "tôi là sinh viên", "tôi là giáo viên",
        # Thời gian & số đếm
        "mấy giờ rồi", "hôm nay", "ngày mai", "hôm qua",
        "một hai ba", "mười hai mươi ba mươi", "một trăm",
        "buổi sáng", "buổi chiều", "buổi tối", "nửa đêm",
        # Tiền & mua sắm
        "bao nhiêu tiền", "đắt quá", "có giảm giá không",
        "tôi lấy cái này", "tiền mặt hay thẻ",
        # Ăn uống
        "cho tôi cà phê", "cho tôi trà", "cho tôi nước",
        "cho xin menu", "tính tiền", "ngon quá",
        "tôi đói", "tôi khát", "tôi ăn chay",
        # Du lịch & hỏi đường
        "tàu điện ngầm ở đâu", "nhà vệ sinh ở đâu",
        "tôi bị lạc", "rẽ trái", "rẽ phải", "đi thẳng",
        "gọi taxi", "đến sân bay",
        # Khách sạn
        "tôi đã đặt phòng", "chìa khoá phòng",
        "mật khẩu wifi", "giờ check-out",
        # Small talk
        "bạn khoẻ không", "tôi khoẻ", "còn bạn",
        "tôi không hiểu", "nói chậm hơn được không",
        "tôi đang học tiếng Nhật", "giúp tôi với",
        # Y tế
        "tôi không khoẻ", "tôi đau đầu",
        "gọi cấp cứu", "tiệm thuốc ở đâu",
        # Cảm xúc
        "tôi yêu bạn", "tôi nhớ bạn", "tôi mệt",
        "tôi vui", "chúc mừng sinh nhật",
    ],
}


# ─── REVERSE QUIZ topic categories (target → VN) ───────────────────────
# Reverse quiz: hiển thị cụm tiếng đích → user đoán nghĩa tiếng Việt.
# Mỗi item = 1 CATEGORY broad — Gemini sẽ tự pick 1 cụm cụ thể từ category đó.
# Khác với forward (VN keyword → Gemini dịch sang target).
LANGUAGE_REVERSE_CATEGORIES = {
    "de": [
        # Chào hỏi & Giao tiếp
        "chào hỏi & giới thiệu bản thân",
        "cảm ơn & xin lỗi",
        "hỏi thăm sức khỏe",
        # Gia đình & Bạn bè
        "gia đình & họ hàng",
        "bạn bè & các mối quan hệ",
        # Thức ăn & Nhà hàng
        "gọi món & thực đơn",
        "đồ uống & quán cafe",
        "khen ngợi món ăn",
        # Mua sắm
        "mua sắm & giá cả",
        "quần áo & thời trang",
        "đổi trả hàng",
        # Giao thông & Đi lại
        "hỏi đường & chỉ đường",
        "tàu xe & phương tiện",
        "sân bay & du lịch",
        # Nhà cửa & Sinh hoạt
        "thuê nhà & hợp đồng",
        "nội thất & phòng ở",
        # Y tế & Sức khỏe
        "bệnh viện & bác sĩ",
        "thuốc & bảo hiểm y tế",
        # Công việc & Học tập
        "phỏng vấn xin việc",
        "công việc & đồng nghiệp",
        "trường học & học tập",
        # Thời tiết & Thiên nhiên
        "thời tiết & mùa trong năm",
        # Sở thích & Giải trí
        "sở thích & hoạt động cuối tuần",
        "phim ảnh & âm nhạc",
        # Số đếm & Thời gian
        "số đếm & tiền bạc",
        "ngày tháng & lịch hẹn",
    ],
    "zh": [
        # Chào hỏi & Giao tiếp
        "chào hỏi & giới thiệu bản thân",
        "cảm ơn & xin lỗi lịch sự",
        "hỏi thăm sức khỏe",
        # Gia đình & Bạn bè
        "gia đình & họ hàng",
        "bạn bè & các mối quan hệ",
        # Thức ăn & Nhà hàng
        "gọi món & thực đơn",
        "đồ uống & quán ca fe",
        "khen ngợi món ăn",
        "đặt bàn nhà hàng",
        # Mua sắm
        "mua sắm & giá cả",
        "quần áo & thời trang",
        "thanh toán & hóa đơn",
        # Giao thông & Đi lại
        "hỏi đường & chỉ đường",
        "tàu xe & phương tiện",
        "sân bay & du lịch",
        # Nhà cửa & Sinh hoạt
        "thuê nhà & sinh hoạt",
        "nội thất & phòng ở",
        # Y tế & Sức khỏe
        "bệnh viện & bác sĩ",
        "thuốc & sức khỏe",
        # Công việc & Học tập
        "phỏng vấn xin việc",
        "công việc & đồng nghiệp",
        "trường học & học tập",
        # Thời tiết & Thiên nhiên
        "thời tiết & mùa trong năm",
        # Sở thích & Giải trí
        "sở thích & hoạt động cuối tuần",
        "phim ảnh & âm nhạc",
        # Số đếm & Tiền
        "số đếm & tiền bạc",
        "ngày tháng & lịch hẹn",
        # Văn hóa
        "lễ hội & phong tục Trung Quốc",
    ],
    "ko": [
        # Chào hỏi & Giao tiếp
        "chào hỏi & giới thiệu bản thân",
        "cảm ơn & xin lỗi lịch sự",
        "hỏi thăm sức khỏe",
        # Gia đình & Bạn bè
        "gia đình & họ hàng",
        "bạn bè & các mối quan hệ",
        # Thức ăn & Nhà hàng
        "gọi món & thực đơn nhà hàng Hàn",
        "đồ uống & quán cafe Hàn",
        "đặt giao đồ ăn",
        # Mua sắm
        "mua sắm & giá cả",
        "quần áo & thời trang",
        "cửa hàng tiện lợi & thanh toán",
        # Giao thông & Đi lại
        "hỏi đường & chỉ đường",
        "tàu điện ngầm & xe buýt",
        "sân bay & du lịch",
        # Nhà cửa & Sinh hoạt
        "thuê nhà & sinh hoạt",
        "nội thất & phòng ở",
        # Y tế & Sức khỏe
        "bệnh viện & bác sĩ",
        "thuốc & bảo hiểm y tế",
        # Công việc & Học tập
        "phỏng vấn xin việc",
        "công việc & đồng nghiệp",
        "trường học & học tập",
        # Thời tiết & Thiên nhiên
        "thời tiết & mùa ở Hàn Quốc",
        # Sở thích & Giải trí
        "sở thích & hoạt động cuối tuần",
        "K-pop & K-drama & văn hóa Hàn",
        # Số đếm & Thời gian
        "số đếm & tiền bạc",
        "ngày tháng & lịch hẹn",
        # Cảm xúc & Ý kiến
        "cảm xúc & bày tỏ ý kiến",
    ],
    # ─── Russian reverse (native=en, general beginner) ────────────────────
    "ru": [
        "greetings & politeness", "introducing yourself",
        "numbers and counting", "telling the time",
        "ordering food and drink", "asking for the bill",
        "shopping for clothes", "shopping at a grocery",
        "asking for directions", "taking a taxi",
        "metro & public transport", "checking in at a hotel",
        "weather and seasons", "talking about hobbies",
        "family and relationships", "feelings and emotions",
        "small talk basics", "health and pharmacy",
        "wishing a happy birthday", "saying thank you and sorry",
    ],
    # ─── Japanese reverse (native=vi, general beginner) ──────────────────
    "ja": [
        "chào hỏi & lịch sự", "giới thiệu bản thân",
        "đếm số & thời gian", "ngày tháng & thứ trong tuần",
        "gọi món ăn uống", "xin hoá đơn nhà hàng",
        "mua quần áo", "đi siêu thị",
        "hỏi đường phố", "gọi taxi",
        "tàu điện ngầm & xe buýt", "check-in khách sạn",
        "thời tiết & mùa", "nói về sở thích",
        "gia đình & bạn bè", "cảm xúc & tâm trạng",
        "small talk cơ bản", "y tế & tiệm thuốc",
        "chúc mừng sinh nhật", "nhiều cách cảm ơn xin lỗi",
    ],
}


# ─── LANGUAGE: FILL_BLANK_QUIZ topics (short photo+sentence quiz) ─────
# 1 sentence with ___ blank, 2-3 option chips, AI photo bg.
# Format inspired by English Canbe / Grammar Goat viral pages.
LANGUAGE_FILL_BLANK_TOPICS = {
    "de": [
        "giới từ chỉ vị trí (in, an, auf, unter)",
        "giới từ chỉ hướng (nach, zu, in)",
        "động từ chia ngôi thứ nhất ich",
        "động từ chia ngôi thứ ba er/sie",
        "mạo từ xác định der/die/das",
        "mạo từ không xác định ein/eine",
        "đại từ phản thân mich/dich/sich",
        "thì hiện tại với động từ thường",
        "thì quá khứ Perfekt với haben/sein",
        "câu hỏi với Wie/Wo/Was/Wann",
        "động từ Modalverben können/müssen/wollen",
        "tính từ vị ngữ với sein",
        "cách chia số nhiều danh từ",
        "câu phủ định với nicht/kein",
        "cách dùng weil và dass",
    ],
    "zh": [
        "lượng từ (个, 只, 本, 张, 杯)",
        "trợ từ kết cấu 的/得/地",
        "giới từ chỉ vị trí 在/到/从",
        "động từ thì quá khứ 了",
        "câu hỏi với 吗/呢/什么/哪里",
        "đại từ sở hữu 我的/你的/他的",
        "số đếm + danh từ (一个, 两本)",
        "động từ tình thái 能/会/想/要",
        "trạng từ 很/真/非常/太",
        "động từ chỉ phương hướng 来/去",
        "câu so sánh với 比",
        "liên từ 因为/所以",
        "câu phủ định với 不/没",
        "cấu trúc 是...的 nhấn mạnh",
        "cách dùng 还是 vs 或者",
    ],
    "ko": [
        "trợ từ chủ ngữ 이/가",
        "trợ từ tân ngữ 을/를",
        "trợ từ chủ đề 은/는",
        "trợ từ địa điểm 에/에서",
        "trợ từ phương hướng 에/으로",
        "đuôi câu kính ngữ 습니다/-ㅂ니다",
        "đuôi câu thân mật 아요/어요",
        "thì quá khứ 았/었",
        "thì tương lai 겠/을 거예요",
        "câu hỏi 까요?/-나요?",
        "động từ phủ định 안/못",
        "động từ tình thái 수 있다/없다",
        "cấu trúc -고 싶다 muốn làm gì",
        "cấu trúc -아/어서 vì/nên",
        "cách dùng 이/가 아니다 phủ định",
    ],
    # ─── Russian fill-blank topics (native=en) ────────────────────────────
    "ru": [
        "nominative vs accusative case",
        "genitive case with quantity",
        "prepositional case after в/на",
        "dative case for indirect object",
        "instrumental case after с",
        "verb conjugation first person я",
        "verb conjugation third person он/она",
        "perfective vs imperfective aspect",
        "past tense agreement (m/f/n)",
        "future tense with буду",
        "imperative form (just one verb)",
        "reflexive verbs ending in -ся",
    ],
    # ─── Japanese fill-blank topics (native=vi) ───────────────────────────
    "ja": [
        "trợ từ chủ ngữ は",
        "trợ từ chủ ngữ が",
        "trợ từ tân ngữ を",
        "trợ từ địa điểm に/で",
        "trợ từ phương hướng へ/に",
        "thể て (te-form)",
        "thể ます lịch sự",
        "thể た quá khứ",
        "đuôi câu です/だ",
        "động từ tình thái できる/られる",
        "câu phủ định ない / ません",
        "thể điều kiện たら / ば",
    ],
}


# ─── LANGUAGE: VOCAB_TABLE_IMAGE topics (static PNG poster) ──────────
# 6-8 vocab items in 3-column table (VN | Target | Pronunciation) + character mascot.
# Static PNG posted to FB /photos endpoint (not /videos). Screenshot-friendly.
LANGUAGE_VOCAB_TABLE_TOPICS = {
    "de": [
        "8 từ vựng trong nhà bếp tiếng Đức (Küche)",
        "8 từ vựng tại sân bay (Flughafen)",
        "8 từ vựng đi siêu thị (Supermarkt)",
        "8 từ vựng phòng tắm (Badezimmer)",
        "8 từ vựng quần áo và thời trang",
        "8 từ vựng phương tiện đi lại",
        "8 từ vựng nội thất phòng ngủ",
        "8 từ vựng thời tiết và mùa",
        "8 từ vựng nơi làm việc và văn phòng",
        "8 từ vựng bệnh viện và sức khỏe",
        "8 từ vựng trường học và học tập",
        "8 từ vựng đồ ăn Đức truyền thống",
        "8 từ vựng hoa quả và rau củ",
        "8 từ vựng thể thao và sở thích",
        "8 từ vựng đồ điện tử hàng ngày",
    ],
    "zh": [
        "8 từ vựng trong nhà bếp tiếng Trung",
        "8 từ vựng tại sân bay",
        "8 từ vựng đi chợ và siêu thị",
        "8 từ vựng quần áo và thời trang",
        "8 từ vựng phương tiện đi lại",
        "8 từ vựng thời tiết và mùa",
        "8 từ vựng món ăn Trung Quốc nổi tiếng",
        "8 từ vựng đồ uống và quán cafe",
        "8 từ vựng tại nhà hàng",
        "8 từ vựng tại khách sạn",
        "8 từ vựng phòng ngủ và nội thất",
        "8 từ vựng bệnh viện và sức khỏe",
        "8 từ vựng hoa quả và rau củ",
        "8 từ vựng thể thao và sở thích",
        "8 từ vựng đồ điện tử hàng ngày",
    ],
    "ko": [
        "8 từ vựng trong nhà bếp tiếng Hàn",
        "8 từ vựng tại sân bay",
        "8 từ vựng đi siêu thị và mua sắm",
        "8 từ vựng quần áo và thời trang",
        "8 từ vựng phương tiện đi lại",
        "8 từ vựng thời tiết và mùa Hàn Quốc",
        "8 từ vựng món ăn Hàn nổi tiếng",
        "8 từ vựng đồ uống và quán cafe Hàn",
        "8 từ vựng tại nhà hàng",
        "8 từ vựng tại cửa hàng tiện lợi",
        "8 từ vựng phòng ngủ và nội thất",
        "8 từ vựng bệnh viện và sức khỏe",
        "8 từ vựng hoa quả và rau củ",
        "8 từ vựng thể thao và sở thích",
        "8 từ vựng văn hóa và lễ hội Hàn",
    ],
    # ─── Russian vocab table (native=en) ──────────────────────────────────
    "ru": [
        "8 Russian kitchen vocabulary words",
        "8 Russian airport vocabulary words",
        "8 Russian supermarket vocabulary words",
        "8 Russian bathroom vocabulary words",
        "8 Russian winter clothing words",
        "8 Russian transport vocabulary",
        "8 Russian renting & housing words",
        "8 Russian weather and seasons words",
        "8 Russian office vocabulary",
        "8 Russian hospital and health words",
        "8 Russian school and study words",
        "8 traditional Russian food words",
    ],
    # ─── Japanese vocab table (native=vi) ─────────────────────────────────
    "ja": [
        "8 từ vựng trong nhà bếp tiếng Nhật",
        "8 từ vựng sân bay Narita",
        "8 từ vựng văn phòng cty Nhật",
        "8 từ vựng quần áo bốn mùa",
        "8 từ vựng phương tiện đi lại Tokyo",
        "8 từ vựng thời tiết Nhật Bản",
        "8 từ vựng món Nhật (sushi, ramen)",
        "8 từ vựng đồ uống (sake, trà)",
        "8 từ vựng tại izakaya",
        "8 từ vựng tại khách sạn Nhật",
        "8 từ vựng đi tàu Shinkansen",
        "8 từ vựng konbini Family Mart",
    ],
}


# ─── LANGUAGE: COMPARE scenarios (2-column basic vs fluent / don't say vs say) ────
# Gemini generates 8 pairs comparing same-meaning phrases (casual vs formal,
# basic vs fluent, common mistake vs correct, etc). Static PNG poster 1080x1350.
# Inspired by viral pages like "Basic vs Fluent English", "Don't say / Say".
LANGUAGE_COMPARE_SCENARIOS = {
    "de": [
        # casual_vs_formal — Du vs Sie variants
        "Du vs Sie — cách xưng hô trang trọng",
        "Cách chào hỏi lịch sự trong tiếng Đức",
        "Cảm ơn và xin lỗi lịch sự tiếng Đức",
        # basic_vs_fluent
        "Cách nói cảm xúc tinh tế (Đức)",
        "Cách mời ai đó đi ăn lịch sự hơn",
        "Cách từ chối khéo bằng tiếng Đức",
        "Cách khen ngợi và động viên người Đức",
        "Cách bày tỏ ý kiến lịch sự tiếng Đức",
        # dont_say_vs_say
        "Đừng nói thế ở Đức — hãy nói thay vào",
        "Câu kém duyên → câu lịch sự tiếng Đức",
        # mistake_vs_correct
        "Lỗi ngữ pháp người mới học hay mắc (Đức)",
        "Sai mạo từ der/die/das — sửa đúng",
        "Lỗi chia động từ Đức phổ biến",
        "Câu cơ bản vs câu người bản ngữ hay dùng",
    ],
    "zh": [
        # casual_vs_formal — 你 vs 您
        "你 vs 您 — xưng hô tiếng Trung",
        "Cách chào hỏi lịch sự trong tiếng Trung",
        "Cảm ơn và xin lỗi lịch sự (Trung)",
        # basic_vs_fluent
        "Cách nói cảm ơn tinh tế hơn (Trung)",
        "Cách mời ai đó đi ăn lịch sự hơn",
        "Cách từ chối khéo bằng tiếng Trung",
        "Cách khen ngợi và động viên người Trung",
        "Cách bày tỏ ý kiến lịch sự tiếng Trung",
        # dont_say_vs_say
        "Câu khó nghe → câu lịch sự (Trung)",
        "Đừng nói thế trong cuộc trò chuyện — nói thay",
        # mistake_vs_correct
        "Lỗi sai phổ biến khi học tiếng Trung",
        "Sai thanh điệu pinyin — sửa đúng",
        "Lỗi dùng 了 / 过 phổ biến",
        "Câu cơ bản vs câu người bản ngữ hay dùng",
    ],
    "ko": [
        # casual_vs_formal — 반말 vs 존댓말
        "반말 vs 존댓말 — thân mật vs kính ngữ",
        "Cách chào hỏi lịch sự trong tiếng Hàn",
        "Cảm ơn và xin lỗi lịch sự (Hàn)",
        # basic_vs_fluent
        "Cách nói cảm xúc tinh tế (Hàn)",
        "Cách mời ai đó đi ăn lịch sự hơn",
        "Cách từ chối khéo bằng tiếng Hàn",
        "Cách khen ngợi và động viên người Hàn",
        "Cách bày tỏ ý kiến lịch sự tiếng Hàn",
        # dont_say_vs_say
        "Câu kém lịch sự → câu trang trọng (Hàn)",
        "Đừng nói thế trong tiếng Hàn — hãy nói thay",
        # mistake_vs_correct
        "Lỗi sai phổ biến khi học tiếng Hàn",
        "Sai 은/는 vs 이/가 — sửa đúng",
        "Lỗi dùng 합니다 / 해요 / 해 phổ biến",
        "Câu cơ bản vs câu người bản ngữ hay dùng",
    ],
    # ─── Russian compare (native=en) ──────────────────────────────────────
    "ru": [
        "ты vs вы — informal vs formal Russian",
        "casual vs formal Russian greetings",
        "polite Russian apology vs casual one",
        "basic vs fluent Russian feelings",
        "common Russian email mistakes vs correct",
        "don't say / say this instead — Russian",
        "rude vs polite phrases in Russian",
        "Russian noun case mistakes — fix them",
        "Russian aspect mistakes (perfective vs imperfective)",
        "Russian word stress mistakes — fix them",
        "Russian тоже vs также — pick the right one",
        "Russian этот vs тот — proximity nuance",
    ],
    # ─── Japanese compare (native=vi) ─────────────────────────────────────
    "ja": [
        "tameguchi vs keigo — thân mật vs lịch sự",
        "cách chào hỏi formal vs casual Nhật",
        "Cảm ơn / xin lỗi với sếp Nhật",
        "Cách diễn đạt cảm xúc tinh tế (Nhật)",
        "Câu giao tiếp văn phòng Nhật nâng cao",
        "Đừng nói thế ở Nhật — nói thay vào",
        "Câu kém duyên → câu lịch sự Nhật",
        "Lỗi sai JLPT N5 N4 hay mắc",
        "Sai は vs が — sửa đúng",
        "Sai に vs で địa điểm — sửa đúng",
        "Lỗi dùng です / だ / である phổ biến",
        "Sai thể kính ngữ — keigo cơ bản",
    ],
}


# ─── LANGUAGE: DIALOGUE scenarios (2-character mini skit) ────────────
# Gemini generates 6-8 turns of dialogue between 2 characters in real scenarios.
# Output rendered as visual-novel style: 2 char portraits + speech bubble + sub.
LANGUAGE_DIALOGUE_SCENARIOS = {
    "de": [
        "đặt món tại nhà hàng Đức",
        "mua sắm tại siêu thị",
        "tìm thuê căn hộ và hỏi chủ nhà",
        "mở tài khoản ngân hàng",
        "hỏi đường đến ga tàu",
        "phỏng vấn xin việc văn phòng",
        "đặt lịch khám bác sĩ",
        "mua vé tàu và hỏi giờ chạy",
        "hỏi đường và chỉ đường trong thành phố",
        "mua SIM điện thoại mới",
        "gặp bạn cùng phòng lần đầu",
        "rủ bạn đi cafe và chọn đồ uống",
        "mua quần áo và thử đồ tại cửa hàng",
        "đặt chỗ tại khách sạn qua điện thoại",
    ],
    "zh": [
        "đặt món tại nhà hàng Trung Quốc",
        "mua sắm tại chợ và trả giá",
        "phỏng vấn xin việc văn phòng",
        "check-in khách sạn",
        "hỏi đường và chỉ đường trong thành phố",
        "đặt phòng khách sạn qua điện thoại",
        "mua đồ và thanh toán tại cửa hàng",
        "gặp bạn mới và tự giới thiệu",
        "rủ bạn đi ăn và chọn nhà hàng",
        "đặt lịch khám bác sĩ",
        "mua vé tàu và hỏi giờ đi",
        "nói chuyện với hàng xóm",
        "mua quần áo và hỏi size",
        "hỏi về thời tiết và lên kế hoạch",
    ],
    "ko": [
        "đặt món tại quán thịt nướng Hàn",
        "mua sắm và hỏi giá tại cửa hàng",
        "phỏng vấn xin việc văn phòng",
        "check-in khách sạn ở Seoul",
        "đi tàu điện ngầm và hỏi đường",
        "mua đồ tại cửa hàng tiện lợi",
        "gặp bạn mới và tự giới thiệu",
        "rủ bạn đi cafe và chọn đồ uống",
        "đặt lịch khám bác sĩ",
        "mua vé và hỏi giờ tàu",
        "nói chuyện với hàng xóm",
        "mua quần áo và hỏi size",
        "rủ bạn đi concert và mua vé",
        "hỏi về thời tiết và lên kế hoạch cuối tuần",
    ],
    # ─── Russian dialogue (native=en, general beginner) ─────────────────
    "ru": [
        "meeting a new friend at a cafe",
        "ordering food at a restaurant",
        "asking a stranger for directions",
        "checking in at a hotel",
        "shopping for clothes at a store",
        "buying tickets at the cinema",
        "calling a taxi to the airport",
        "introducing your family to a friend",
        "small talk at a birthday party",
        "asking the pharmacist for medicine",
        "ordering tea at a tea shop",
        "making plans for the weekend",
    ],
    # ─── Japanese dialogue (native=vi, general beginner) ───────────────
    "ja": [
        "gặp bạn mới tại quán cafe",
        "gọi món tại nhà hàng",
        "hỏi đường người lạ",
        "check-in khách sạn",
        "mua quần áo tại cửa hàng",
        "mua vé xem phim",
        "gọi taxi ra sân bay",
        "giới thiệu gia đình cho bạn",
        "small talk tại tiệc sinh nhật",
        "mua thuốc tại tiệm thuốc",
        "gọi trà tại quán trà",
        "lên kế hoạch cuối tuần",
    ],
}


# ─── LANGUAGE: WHATS_BOARD themes (9-grid cheat-sheet vocab) ─────────
# 9 items per video, all visible at once. Highlight reads each item 2x in target lang.
# Themes must be CONCRETE and IMAGEABLE (animals, food, objects, transport...).
# Avoid: emotions, body parts (CF safety filter triggers).
LANGUAGE_BOARD_THEMES = {
    "de": [
        "9 con vật phổ biến (Tiere)",
        "9 loại trái cây (Obst)",
        "9 món ăn Đức (Speisen)",
        "9 loại rau củ (Gemüse)",
        "9 đồ uống (Getränke)",
        "9 phương tiện đi lại (Verkehrsmittel)",
        "9 quần áo cơ bản (Kleidung)",
        "9 đồ nội thất (Möbel)",
        "9 đồ nhà bếp (Küchengeräte)",
        "9 dụng cụ học tập (Schulsachen)",
        "9 đồ điện tử (Elektronik)",
        "9 thể thao (Sportarten)",
        "9 loài hoa và cây cảnh",
        "9 đồ dùng phòng tắm",
        "9 nhạc cụ phổ biến",
    ],
    "zh": [
        "9 con vật phổ biến (动物)",
        "9 loại trái cây (水果)",
        "9 món ăn Trung Quốc (食物)",
        "9 loại rau củ (蔬菜)",
        "9 đồ uống (饮料)",
        "9 phương tiện đi lại (交通工具)",
        "9 quần áo cơ bản (衣服)",
        "9 đồ nội thất (家具)",
        "9 đồ nhà bếp (厨房用品)",
        "9 dụng cụ học tập (文具)",
        "9 đồ điện tử (电子产品)",
        "9 đồ thể thao (运动用品)",
        "9 loài hoa và cây cảnh",
        "9 đồ dùng phòng tắm",
        "9 nhạc cụ phổ biến",
    ],
    "ko": [
        "9 con vật phổ biến (동물)",
        "9 loại trái cây (과일)",
        "9 món ăn Hàn Quốc (음식)",
        "9 loại rau củ (채소)",
        "9 đồ uống (음료)",
        "9 phương tiện đi lại (교통수단)",
        "9 quần áo cơ bản (옷)",
        "9 đồ nội thất (가구)",
        "9 đồ nhà bếp (주방용품)",
        "9 dụng cụ học tập (학용품)",
        "9 đồ điện tử (전자제품)",
        "9 đồ thể thao (운동용품)",
        "9 loài hoa và cây cảnh",
        "9 đồ dùng phòng tắm",
        "9 nhạc cụ phổ biến",
    ],
    # ─── Russian board themes (native=en) ────────────────────────────────
    "ru": [
        "9 common animals (животные)",
        "9 fruits (фрукты)",
        "9 traditional Russian dishes (блюда)",
        "9 vegetables (овощи)",
        "9 drinks (напитки)",
        "9 modes of transport (транспорт)",
        "9 basic clothing items (одежда)",
        "9 pieces of furniture (мебель)",
        "9 kitchen tools (кухонная утварь)",
        "9 school supplies (школьные принадлежности)",
        "9 electronics (электроника)",
        "9 sports (спорт)",
    ],
    # ─── Japanese board themes (native=vi) ───────────────────────────────
    "ja": [
        "9 con vật phổ biến (動物)",
        "9 loại trái cây (果物)",
        "9 món ăn Nhật (料理)",
        "9 loại rau củ (野菜)",
        "9 đồ uống (飲み物)",
        "9 phương tiện đi lại (乗り物)",
        "9 quần áo cơ bản (服)",
        "9 đồ nội thất (家具)",
        "9 đồ nhà bếp (台所用品)",
        "9 dụng cụ học tập (文房具)",
        "9 đồ điện tử (電化製品)",
        "9 thể thao (スポーツ)",
    ],
}


# ─── LANGUAGE: WHATS_THIS themes (visual vocab "Đây là gì?") ──────────
# 10 items per video. Each theme = broad category Gemini drills down into.
# Per item: AI image (Cloudflare FLUX) + target word + voice + phonetic.
LANGUAGE_WHATS_THIS_THEMES = {
    "de": [
        "10 hành động hàng ngày (động từ)",
        "10 món ăn & đồ uống Đức",
        "10 phương tiện đi lại ở Đức",
        "10 đồ vật trong nhà (Möbel)",
        "10 nghề nghiệp phổ biến",
        "10 thời tiết & mùa trong năm",
        "10 đồ vật trong nhà bếp",
        "10 hoạt động cuối tuần ở Đức",
        "10 loài động vật quen thuộc",
        "10 loại quần áo và phụ kiện",
        "10 đồ vật trong phòng học",
        "10 loại nhạc cụ phổ biến",
    ],
    "zh": [
        "10 hành động hàng ngày (động từ)",
        "10 món ăn Trung Quốc nổi tiếng",
        "10 phương tiện đi lại ở Trung Quốc",
        "10 đồ vật trong nhà",
        "10 nghề nghiệp phổ biến",
        "10 thời tiết & mùa",
        "10 đồ dùng nhà bếp",
        "10 lễ hội và biểu tượng Trung Quốc",
        "10 loài động vật quen thuộc",
        "10 loại quần áo và phụ kiện",
        "10 đồ vật trong phòng học",
        "10 loại nhạc cụ phổ biến",
    ],
    "ko": [
        "10 hành động hàng ngày (동작)",
        "10 món ăn Hàn Quốc nổi tiếng",
        "10 phương tiện đi lại ở Hàn",
        "10 đồ vật trong nhà",
        "10 nghề nghiệp phổ biến",
        "10 thời tiết & mùa Hàn Quốc",
        "10 đồ dùng nhà bếp",
        "10 hoạt động K-culture và giải trí",
        "10 loài động vật quen thuộc",
        "10 loại quần áo và phụ kiện",
        "10 đồ vật trong phòng học",
        "10 loại nhạc cụ phổ biến",
    ],
    # ─── Russian whats-this (native=en, general beginner) ───────────────
    "ru": [
        "10 daily verbs (eat, sleep, drink, walk...)",
        "10 common foods and drinks",
        "10 modes of transport",
        "10 objects inside a home",
        "10 common jobs and professions",
        "10 emotions and feelings",
        "10 weather words and seasons",
        "10 body parts",
        "10 fruits",
        "10 vegetables",
        "10 clothes and accessories",
        "10 animals you see every day",
    ],
    # ─── Japanese whats-this (native=vi, general beginner) ──────────────
    "ja": [
        "10 hành động hàng ngày (ăn, ngủ, uống, đi...)",
        "10 món ăn & đồ uống phổ biến",
        "10 phương tiện đi lại",
        "10 đồ vật trong nhà",
        "10 nghề nghiệp phổ biến",
        "10 cảm xúc & trạng thái",
        "10 thời tiết & mùa",
        "10 bộ phận cơ thể người",
        "10 loại trái cây",
        "10 loại rau củ",
        "10 quần áo & phụ kiện",
        "10 con vật quen thuộc hàng ngày",
    ],
}


# ─── HEALTH topics (Đọc Vị Cơ Thể) ─────────────────────────────────────
HEALTH_TOPICS = [
    "dấu hiệu bệnh tiểu đường", "dấu hiệu đột quỵ",
    "dấu hiệu thiếu vitamin D", "dấu hiệu thiếu sắt",
    "dấu hiệu cao huyết áp", "dấu hiệu bệnh gan",
    "dấu hiệu bệnh tim mạch", "dấu hiệu mất nước",
    "dấu hiệu bệnh thận", "dấu hiệu bệnh dạ dày",
    "nguyên nhân mất ngủ", "nguyên nhân đau lưng",
    "nguyên nhân tăng cân", "nguyên nhân rụng tóc",
    "cách phòng cảm cúm", "cách phòng đột quỵ",
    "cách phòng tiểu đường", "cách phòng đau lưng",
    "cách chữa mất ngủ", "cách chữa căng thẳng",
    "cách chữa đau đầu", "cách giảm stress",
    "mẹo tăng cường miễn dịch", "mẹo giảm cân hiệu quả",
    "thực phẩm tốt cho tim", "thực phẩm tốt cho não",
    "thực phẩm tốt cho da", "thực phẩm tốt cho mắt",
    "lợi ích của trà xanh", "lợi ích của uống đủ nước",
    "lợi ích của đi bộ 30 phút", "lợi ích của ngủ đủ giấc",
]


# ── META instruction injected into every en-native request ───────────────
# The 8 layout system prompts show Vietnamese examples for native-language
# fields. Gemini tends to mimic those examples even when the user message is
# English — so we override with a per-request instruction that wins over any
# in-prompt Vietnamese examples AND demands IDIOMATIC native English phrasing
# (not "translated from Vietnamese" English like "you must know" / "In X, Y
# is what?" — which sound like robotic transliterations).
_EN_META = (
    "[NATIVE_LANG=en] CRITICAL: native_lang is English. ALL native-language "
    "fields in your JSON output (question_native, intro_native, outro_native, "
    "topic_label, short_title, explanation, native, native_answer, "
    "native_translation, left_native / right_native, caption text) MUST be "
    "IDIOMATIC NATIVE ENGLISH — phrased the way a real English-speaking "
    "TikTok/Reels creator would write them. DO NOT translate Vietnamese "
    "templates literally. Avoid stiff calques: NEVER \"X phrases ... you "
    "must know\" (use \"Top 10 X phrases for Y\" / \"essential X phrases\" "
    "/ \"must-know X phrases for Y\"); NEVER \"In X, 'Y' is what?\" (use "
    "\"How do you say 'Y' in X?\" or \"What's 'Y' in X?\"); NEVER \"Save "
    "and follow for daily practice! Good luck!\" (use \"Hit save and follow "
    "for more daily X!\" or \"Like, save, share — see you tomorrow!\"). "
    "Caption line 1 = \"<flag> {target_lang_name} for everyday "
    "conversation\". Caption uses \"🎧 Listen & learn:\" not \"🎧 Nghe & "
    "học cùng nhau:\". The prompt examples below show Vietnamese; ignore "
    "them and write natively in English.\n\n"
)


def _format_phrases_request(topic: str, target_lang_name: str = "Đức") -> str:
    """Format a natural-language request for phrases layout."""
    if _NATIVE_LANG == "en":
        return _EN_META + f"10 {target_lang_name} phrases about {topic}"
    return f"10 câu tiếng {target_lang_name} về {topic}"


def _format_quiz_request(question: str, target_lang_name: str = "Đức") -> str:
    """Format a forward quiz request: native keyword → Gemini picks target translation."""
    if _NATIVE_LANG == "en":
        return (
            _EN_META
            + f"Quiz: how do you say '{question}' in {target_lang_name}? "
            f"question_native must be \"In {target_lang_name}, how do you say "
            f"'{question}'?\" (English), NOT \"Trong tiếng {target_lang_name}…\"."
        )
    return f"Quiz: '{question}' trong tiếng {target_lang_name} nói thế nào?"


def _format_quiz_reverse_request(category: str, target_lang_name: str = "Đức") -> str:
    """Format a reverse quiz request: target phrase → user guesses native meaning."""
    if _NATIVE_LANG == "en":
        return (
            _EN_META
            + f"Reverse quiz on the topic '{category}': "
            f"pick ONE common phrase in {target_lang_name} and ask the user "
            f"to guess the English meaning (4 options, exactly 1 correct)."
        )
    return (
        f"Reverse quiz về chủ đề '{category}': "
        f"chọn 1 cụm từ tiếng {target_lang_name} phổ biến rồi yêu cầu user "
        f"đoán nghĩa tiếng Việt."
    )


def _format_whats_this_request(theme: str, target_lang_name: str = "Đức") -> str:
    """Format a visual-vocab request: pick 10 concrete items from theme."""
    if _NATIVE_LANG == "en":
        return (
            _EN_META
            + f"Whats-this visual vocab — theme '{theme}' in {target_lang_name}. "
            f"Generate 10 concrete items (single noun or single verb). Each item "
            f"needs an image_prompt suitable for AI illustration. Each item's "
            f"native_answer is the English translation of the target word."
        )
    return (
        f"Whats-this visual vocab — theme '{theme}' bằng tiếng {target_lang_name}. "
        f"Sinh 10 items concrete (noun đơn hoặc verb đơn), MỖI item có image_prompt "
        f"mô tả cụ thể để gen AI illustration."
    )


def _format_whats_board_request(theme: str, target_lang_name: str = "Đức") -> str:
    """Format a 9-board cheat-sheet vocab request."""
    if _NATIVE_LANG == "en":
        return (
            _EN_META
            + f"Whats-board 9-grid cheat sheet — theme '{theme}' in {target_lang_name}. "
            f"Generate EXACTLY 9 concrete imageable NOUNS (object / animal / food). "
            f"Each item needs an image_prompt the AI can draw clearly."
        )
    return (
        f"Whats-board 9-grid cheat sheet — theme '{theme}' bằng tiếng {target_lang_name}. "
        f"Sinh ĐÚNG 9 items concrete NOUN imageable (an object/animal/food the AI can draw clearly). "
        f"MỖI item có image_prompt mô tả tượng hình."
    )


def _format_guess_word_request(theme: str, target_lang_name: str = "Đức") -> str:
    """Format a guess-the-word (10-word) request. Reuses board vocab themes."""
    if _NATIVE_LANG == "en":
        return (
            _EN_META
            + f"Guess-the-word game — 10 common {target_lang_name} words on the "
            f"theme '{theme}'. For EACH word give: target word, native (English) "
            f"translation, first-letter hint, part of speech, IPA, and ONE short "
            f"example sentence in {target_lang_name}."
        )
    return (
        f"Trò đoán từ — 10 từ {target_lang_name} thông dụng theo chủ đề '{theme}'. "
        f"MỖI từ cần: từ {target_lang_name}, nghĩa tiếng Việt, gợi ý chữ cái đầu, "
        f"loại từ, IPA, và 1 câu ví dụ ngắn bằng {target_lang_name}."
    )


def _format_vocab_card_request(theme: str, target_lang_name: str = "Đức") -> str:
    """Format a vocab_card request — ONE word, illustration, multi-language grid.

    CEO 2026-06-30: single concrete word in target lang, with:
      • a photorealistic illustration,
      • a pronunciation guide,
      • an example sentence (target lang) with the word highlighted,
      • translations into the channel's native lang + 6 popular world langs.
    """
    if _NATIVE_LANG == "en":
        return (
            _EN_META
            + f"Vocab card — pick ONE useful {target_lang_name} word on the theme "
            f"'{theme}'. Include pronunciation, an example sentence that uses "
            f"the word (mark the inflected form), a photo-realistic illustration "
            f"prompt, and translations into the channel's native language plus "
            f"6 popular world languages (exclude the target language itself)."
        )
    return (
        f"Vocab card — 1 từ {target_lang_name} hữu ích theo chủ đề '{theme}'. "
        f"Kèm phiên âm, 1 câu ví dụ {target_lang_name} chứa từ (đánh dấu dạng biến "
        f"đổi nếu có), prompt ảnh photorealistic, và bản dịch sang tiếng mẹ đẻ "
        f"+ 6 ngôn ngữ thông dụng (loại bỏ chính {target_lang_name})."
    )


def _format_dialogue_request(scenario: str, target_lang_name: str = "Đức") -> str:
    """Format a 2-character dialogue (mini skit) request."""
    if _NATIVE_LANG == "en":
        return (
            _EN_META
            + f"Dialogue mini-skit — scenario '{scenario}' in {target_lang_name}. "
            f"Generate 2 characters (A and B) + a scene + 6-8 turns. "
            f"Each turn has target text + pronunciation + English translation."
        )
    return (
        f"Dialogue mini-skit — scenario '{scenario}' bằng tiếng {target_lang_name}. "
        f"Sinh 2 nhân vật (A và B) + cảnh nền + 6-8 lượt thoại. "
        f"Mỗi lượt có target text + phiên âm + dịch tiếng Việt."
    )


def _format_fill_blank_request(topic: str, target_lang_name: str = "Đức") -> str:
    """Format a fill-in-blank short quiz request."""
    if _NATIVE_LANG == "en":
        return (
            _EN_META
            + f"Fill-blank short quiz — topic '{topic}' in {target_lang_name}. "
            f"Generate ONE sentence with a single blank ___ + 3 options (1 right, 2 close-but-wrong) + "
            f"a photo-realistic image_prompt of a person doing the action. "
            f"native_translation is in English."
        )
    return (
        f"Fill-blank short quiz — chủ đề '{topic}' bằng tiếng {target_lang_name}. "
        f"Sinh 1 câu duy nhất có 1 từ blank ___ + 3 options (1 đúng, 2 sai gần đúng) + "
        f"image_prompt photo realistic người đang làm hành động liên quan câu."
    )


def _format_vocab_table_request(topic: str, target_lang_name: str = "Đức") -> str:
    """Format a static vocab table image request."""
    if _NATIVE_LANG == "en":
        return (
            _EN_META
            + f"Vocab table static poster — topic '{topic}' in {target_lang_name}. "
            f"Generate 8 concrete items + a character mascot fitting the theme. "
            f"Native column header = \"English\", target column header = {target_lang_name!r}."
        )
    return (
        f"Vocab table static poster — chủ đề '{topic}' bằng tiếng {target_lang_name}. "
        f"Sinh 8 items concrete + character mascot fitting the theme (cooking → chef, "
        f"office → businessman, garden → grandmother). Output là PNG static post FB."
    )


def _format_compare_request(topic: str, target_lang_name: str = "Đức") -> str:
    """Format a 2-column compare static image request."""
    if _NATIVE_LANG == "en":
        return (
            _EN_META
            + f"Compare 2-column static poster — topic '{topic}' in {target_lang_name}. "
            f"Generate 8 comparison pairs (left=basic/casual/wrong, right=fluent/formal/correct). "
            f"Same meaning, different expression. left_native + right_native fields in English."
        )
    return (
        f"Compare 2-column static poster — chủ đề '{topic}' bằng tiếng {target_lang_name}. "
        f"Sinh 8 cặp so sánh (left=basic/casual/wrong, right=fluent/formal/correct). "
        f"Mỗi cặp cùng nghĩa nhưng diễn đạt khác (cách lịch sự hơn, hoặc câu đúng). "
        f"Output là PNG static post FB 4:5. Có emoji + phiên âm cả 2 phía."
    )


# DEPRECATED — kept for backward compat only. New code uses _ALL_LAYOUTS below.
# Old 16-state cycle had phrases as anchor (every other slot = 50% phrases).
_SPECIAL_LAYOUTS = [
    "forward", "reverse", "visual", "board", "dialogue", "fill_blank", "vocab_table", "compare",
]

# 9-state EVEN cycle. Every layout (incl. phrases) appears once per cycle.
# At 1h/post → full cycle = 9h → each layout 2-3 posts/day per channel.
# state["last_layout"] is the source of truth — it's set by auto_post.py after
# each successful upload to the layout_type string returned by these pickers.
_ALL_LAYOUTS = [
    "phrases", "quiz", "quiz_reverse", "whats_this", "whats_board",
    "dialogue", "fill_blank", "vocab_table", "compare", "guess_word",
    "vocab_card",
]


# ─────────────────────────────────────────────────────────────────────────
# Anti-dup sliding window sizes per layout.
# Larger window → less topic repetition. Should be ~70-85% of pool size:
#   leave 1-2 topics each cycle so `available` list is never empty.
# Pool sizes (current): phrases 44-59, quizzes 65-75, specials 10-12.
# 1h/bài × 14-state cycle → phrases ~12 posts/day, specials ~1-2 posts/day.
# Window 35 for phrases → no repeat within ~3 days of phrases content.
# ─────────────────────────────────────────────────────────────────────────
_DUP_WINDOW_PHRASES = 35       # was 12 — pool 44-59
_DUP_WINDOW_QUIZ = 35          # was 12 — pool 65-75
_DUP_WINDOW_REVERSE_QUIZ = 12  # was 8 — pool variable
_DUP_WINDOW_WHATS_THIS = 8     # was 6 — pool 10
_DUP_WINDOW_WHATS_BOARD = 10   # was 8 — pool 12
_DUP_WINDOW_DIALOGUE = 10      # was 8 — pool 12
_DUP_WINDOW_FILL_BLANK = 10    # was 8 — pool 12
_DUP_WINDOW_VOCAB_TABLE = 10   # was 8 — pool 12
_DUP_WINDOW_COMPARE = 10       # new: pool 12 per lang

# Cross-layout dedup for VISUAL vocab pickers (CEO 2026-07-13). The 5 pickers
# whats_this, whats_board, guess_word, vocab_card, vocab_table all draw
# from overlapping visual taxonomies (kitchen, clothes, fruit, stationery...).
# Each keeps its own per-layout list, but the same "kitchen" theme was landing
# on the feed 4-5 times via different layouts. This shared window is the
# cross-cutting dedup: normalize each theme to a core noun and skip any theme
# whose normalized key was used by ANY visual layout in the last N picks.
_DUP_WINDOW_VISUAL_GLOBAL = 18

_VISUAL_STRIP_LEADING_TOKENS = (
    "từ vựng", "loại", "dụng cụ", "đồ dùng", "đồ",
    "con", "loài", "nghề nghiệp", "nghề", "phương tiện",
    "nhạc cụ", "lễ hội", "món ăn", "thời tiết", "quần áo",
    "hoa quả và", "hoa quả",
)


def _normalize_visual_theme(topic: str) -> str:
    """Extract core noun from a visual theme label for cross-layout dedup.

    Examples:
      "9 loại rau củ (蔬菜)"          => "rau củ"
      "8 từ vựng trong nhà bếp"      => "trong nhà bếp"
      "10 đồ dùng nhà bếp"           => "nhà bếp"
      "9 đồ nhà bếp (厨房用品)"       => "nhà bếp"
      "9 dụng cụ học tập (文具)"      => "học tập"
    """
    import re as _re
    t = topic or ""
    t = _re.sub(r"\s*[\(（][^)）]*[\)）]", "", t)
    t = _re.sub(r"^\s*\d+\s+", "", t)
    lowered = t.lower()
    for tok in sorted(_VISUAL_STRIP_LEADING_TOKENS, key=len, reverse=True):
        if lowered.startswith(tok + " "):
            t = t[len(tok) + 1 :]
            break
    return _re.sub(r"\s+", " ", t).strip().lower()


def _visual_pool_filter(pool, own_used, own_window, state):
    """Filter pool by own recent + shared visual recent (cross-layout dedup)."""
    global_used = state.setdefault("used_visual_themes", [])
    own_ban = set(own_used[-own_window:])
    global_ban = {_normalize_visual_theme(t) for t in global_used[-_DUP_WINDOW_VISUAL_GLOBAL:]}
    return [
        t for t in pool
        if t not in own_ban and _normalize_visual_theme(t) not in global_ban
    ]


def _visual_track(state, theme):
    """Append theme to the shared visual-history window."""
    state.setdefault("used_visual_themes", []).append(theme)


def _avoid_recent_hint(used: list[str], current_topic: str, n_show: int = 15) -> str:
    """Build a Vietnamese exclusion hint string injected into the Gemini request.

    Tells the LLM: "you ARE generating content for `current_topic`, but here are
    other topics we recently posted — pick angles/phrases that don't overlap with
    those neighbors". Gemini then nudges toward fresh wording, even if the topic
    label is the same as some past one (which the sliding window may eventually
    allow back in).

    Returns an empty string when there's no history (first few posts).
    """
    # Take last n_show used items, dedupe in order, drop the current_topic itself
    recent = []
    seen = set()
    for t in reversed(used):
        if t == current_topic or t in seen:
            continue
        recent.append(t)
        seen.add(t)
        if len(recent) >= n_show:
            break
    if not recent:
        return ""
    bullet_list = "\n".join(f"  • {t}" for t in recent)
    if _NATIVE_LANG == "en":
        return (
            f"\n\nIMPORTANT (anti-dup): the channel recently posted these "
            f"topics — pick angles / phrases / vocabulary that DO NOT overlap "
            f"with them:\n{bullet_list}"
        )
    return (
        f"\n\nQUAN TRỌNG (chống trùng): Các chủ đề đã đăng gần đây trên kênh — "
        f"hãy chọn góc nhìn / câu / vocabulary KHÁC để tránh lặp với chúng:\n"
        f"{bullet_list}"
    )


def pick_next_request(
    state: dict,
    niche: str = "language",
    target_lang: str = "de",
    target_lang_name: str = "Đức",
    native_lang: str = "vi",
) -> tuple[str, str]:
    """Pick the next auto-post request text + layout.

    Returns (request_text, layout_type).

    9-state EVEN cycle — every layout appears once per cycle (~11% each):
      phrases → quiz → quiz_reverse → whats_this → whats_board →
      dialogue → fill_blank → vocab_table → compare → repeat

    Picks the next layout based on state["last_layout"] (set by auto_post.py
    after each successful upload). Falls back gracefully if the value is
    missing or unknown.

    Mutates `state` in place to track usage history.
    """
    # Stash the channel's native_lang so the per-layout format helpers can
    # emit Vietnamese or English request strings as appropriate. Reset at the
    # bottom would be cleaner, but each pick_next_request call covers exactly
    # one auto-post cycle, so leaving it set until the next call is fine.
    global _NATIVE_LANG
    _NATIVE_LANG = (native_lang or "vi").lower()

    if niche == "health":
        # Health: phrases only (quiz not yet wired for health)
        used = state.setdefault("used_health_topics", [])
        available = [t for t in HEALTH_TOPICS if t not in used[-10:]]
        if not available:
            available = HEALTH_TOPICS
            used.clear()
        topic = random.choice(available)
        used.append(topic)
        return topic, "phrases"   # Gemini auto-detects query type via NICHE prompt

    # ─── Language: 9-state even cycle ─────────────────────────────
    last_layout = state.get("last_layout", "compare")
    try:
        idx = _ALL_LAYOUTS.index(last_layout)
    except ValueError:
        # Unknown / legacy value → restart from start of cycle
        idx = len(_ALL_LAYOUTS) - 1
    next_layout = _ALL_LAYOUTS[(idx + 1) % len(_ALL_LAYOUTS)]

    # Dispatch to the picker for `next_layout`. Each picker returns its own
    # (request_text, layout_type) tuple and updates its own dup-window history.
    PICKERS = {
        "phrases":      _pick_phrases,
        "quiz":         _pick_quiz_forward,
        "quiz_reverse": _pick_quiz_reverse,
        "whats_this":   _pick_whats_this,
        "whats_board":  _pick_whats_board,
        "dialogue":     _pick_dialogue,
        "fill_blank":   _pick_fill_blank,
        "vocab_table":  _pick_vocab_table,
        "compare":      _pick_compare,
        "guess_word":   _pick_guess_word,
        "vocab_card":   _pick_vocab_card,
    }
    picker = PICKERS.get(next_layout, _pick_phrases)
    return picker(state, target_lang, target_lang_name)


def _pick_phrases(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    used = state.setdefault("used_topics", [])
    pool = LANGUAGE_TOPICS.get(target_lang) or LANGUAGE_TOPICS["de"]
    available = [t for t in pool if t not in used[-_DUP_WINDOW_PHRASES:]]
    if not available:
        available = pool
        used.clear()
    topic = random.choice(available)
    used.append(topic)
    base_req = _format_phrases_request(topic, target_lang_name)
    return base_req + _avoid_recent_hint(used, topic, n_show=15), "phrases"


def _pick_quiz_forward(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    used = state.setdefault("used_quizzes", [])
    pool = LANGUAGE_QUIZ_QUESTIONS.get(target_lang) or LANGUAGE_QUIZ_QUESTIONS["de"]
    available = [q for q in pool if q not in used[-_DUP_WINDOW_QUIZ:]]
    if not available:
        available = pool
        used.clear()
    question = random.choice(available)
    used.append(question)
    state["last_quiz_direction"] = "forward"
    state["last_special_layout"] = "forward"
    base_req = _format_quiz_request(question, target_lang_name)
    return base_req + _avoid_recent_hint(used, question, n_show=15), "quiz"


# Forward decl handled — _pick_whats_board defined below



def _pick_quiz_reverse(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    used = state.setdefault("used_reverse_quizzes", [])
    pool = LANGUAGE_REVERSE_CATEGORIES.get(target_lang) or LANGUAGE_REVERSE_CATEGORIES.get("de", [])
    if not pool:
        # Fallback to forward if reverse pool empty for this lang
        return _pick_quiz_forward(state, target_lang, target_lang_name)
    available = [c for c in pool if c not in used[-_DUP_WINDOW_REVERSE_QUIZ:]]
    if not available:
        available = pool
        used.clear()
    category = random.choice(available)
    used.append(category)
    state["last_quiz_direction"] = "reverse"
    state["last_special_layout"] = "reverse"
    base_req = _format_quiz_reverse_request(category, target_lang_name)
    return base_req + _avoid_recent_hint(used, category, n_show=12), "quiz_reverse"


def _pick_whats_this(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    """Pick a visual-vocab theme and return (request_text, 'whats_this')."""
    used = state.setdefault("used_whats_this_themes", [])
    pool = LANGUAGE_WHATS_THIS_THEMES.get(target_lang) or LANGUAGE_WHATS_THIS_THEMES.get("de", [])
    if not pool:
        return _pick_quiz_forward(state, target_lang, target_lang_name)
    available = _visual_pool_filter(pool, used, _DUP_WINDOW_WHATS_THIS, state)
    if not available:
        available = [t for t in pool if t not in used[-_DUP_WINDOW_WHATS_THIS:]]
    if not available:
        available = pool
        used.clear()
    theme = random.choice(available)
    used.append(theme)
    _visual_track(state, theme)
    state["last_special_layout"] = "visual"
    base_req = _format_whats_this_request(theme, target_lang_name)
    return base_req + _avoid_recent_hint(used, theme, n_show=8), "whats_this"


def _pick_whats_board(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    """Pick a 9-grid board theme and return (request_text, 'whats_board')."""
    used = state.setdefault("used_whats_board_themes", [])
    pool = LANGUAGE_BOARD_THEMES.get(target_lang) or LANGUAGE_BOARD_THEMES.get("de", [])
    if not pool:
        return _pick_whats_this(state, target_lang, target_lang_name)
    available = _visual_pool_filter(pool, used, _DUP_WINDOW_WHATS_BOARD, state)
    if not available:
        available = [t for t in pool if t not in used[-_DUP_WINDOW_WHATS_BOARD:]]
    if not available:
        available = pool
        used.clear()
    theme = random.choice(available)
    used.append(theme)
    _visual_track(state, theme)
    state["last_special_layout"] = "board"
    base_req = _format_whats_board_request(theme, target_lang_name)
    return base_req + _avoid_recent_hint(used, theme, n_show=10), "whats_board"


def _pick_guess_word(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    """Pick a vocab theme for the guess-the-word layout (reuses board themes)."""
    used = state.setdefault("used_guess_word_themes", [])
    pool = LANGUAGE_BOARD_THEMES.get(target_lang) or LANGUAGE_BOARD_THEMES.get("de", [])
    if not pool:
        return _pick_quiz_forward(state, target_lang, target_lang_name)
    available = _visual_pool_filter(pool, used, _DUP_WINDOW_WHATS_BOARD, state)
    if not available:
        available = [t for t in pool if t not in used[-_DUP_WINDOW_WHATS_BOARD:]]
    if not available:
        available = pool
        used.clear()
    theme = random.choice(available)
    used.append(theme)
    _visual_track(state, theme)
    state["last_special_layout"] = "guess_word"
    base_req = _format_guess_word_request(theme, target_lang_name)
    return base_req + _avoid_recent_hint(used, theme, n_show=10), "guess_word"


def _pick_vocab_card(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    """Pick a single-word vocab theme for the vocab_card layout (CEO 2026-06-30).

    Reuses the BOARD theme pool (kitchen / nature / office / ...) — Gemini
    then picks ONE concrete word inside that theme. Anti-dup window
    shares ``used_vocab_card_themes`` so the same theme doesn't repeat
    too often per channel.
    """
    used = state.setdefault("used_vocab_card_themes", [])
    pool = LANGUAGE_BOARD_THEMES.get(target_lang) or LANGUAGE_BOARD_THEMES.get("de", [])
    if not pool:
        return _pick_quiz_forward(state, target_lang, target_lang_name)
    available = _visual_pool_filter(pool, used, _DUP_WINDOW_WHATS_BOARD, state)
    if not available:
        available = [t for t in pool if t not in used[-_DUP_WINDOW_WHATS_BOARD:]]
    if not available:
        available = pool
        used.clear()
    theme = random.choice(available)
    used.append(theme)
    _visual_track(state, theme)
    state["last_special_layout"] = "vocab_card"
    base_req = _format_vocab_card_request(theme, target_lang_name)
    return base_req + _avoid_recent_hint(used, theme, n_show=10), "vocab_card"


def _pick_dialogue(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    """Pick a dialogue scenario and return (request_text, 'dialogue')."""
    used = state.setdefault("used_dialogue_scenarios", [])
    pool = LANGUAGE_DIALOGUE_SCENARIOS.get(target_lang) or LANGUAGE_DIALOGUE_SCENARIOS.get("de", [])
    if not pool:
        return _pick_whats_board(state, target_lang, target_lang_name)
    available = [s for s in pool if s not in used[-_DUP_WINDOW_DIALOGUE:]]
    if not available:
        available = pool
        used.clear()
    scenario = random.choice(available)
    used.append(scenario)
    state["last_special_layout"] = "dialogue"
    base_req = _format_dialogue_request(scenario, target_lang_name)
    return base_req + _avoid_recent_hint(used, scenario, n_show=10), "dialogue"


def _pick_fill_blank(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    """Pick a fill-blank quiz topic and return (request_text, 'fill_blank')."""
    used = state.setdefault("used_fill_blank_topics", [])
    pool = LANGUAGE_FILL_BLANK_TOPICS.get(target_lang) or LANGUAGE_FILL_BLANK_TOPICS.get("de", [])
    if not pool:
        return _pick_dialogue(state, target_lang, target_lang_name)
    available = [t for t in pool if t not in used[-_DUP_WINDOW_FILL_BLANK:]]
    if not available:
        available = pool
        used.clear()
    topic = random.choice(available)
    used.append(topic)
    state["last_special_layout"] = "fill_blank"
    base_req = _format_fill_blank_request(topic, target_lang_name)
    return base_req + _avoid_recent_hint(used, topic, n_show=10), "fill_blank"


def _pick_vocab_table(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    """Pick a vocab table topic and return (request_text, 'vocab_table')."""
    used = state.setdefault("used_vocab_table_topics", [])
    pool = LANGUAGE_VOCAB_TABLE_TOPICS.get(target_lang) or LANGUAGE_VOCAB_TABLE_TOPICS.get("de", [])
    if not pool:
        return _pick_fill_blank(state, target_lang, target_lang_name)
    available = _visual_pool_filter(pool, used, _DUP_WINDOW_VOCAB_TABLE, state)
    if not available:
        available = [t for t in pool if t not in used[-_DUP_WINDOW_VOCAB_TABLE:]]
    if not available:
        available = pool
        used.clear()
    topic = random.choice(available)
    used.append(topic)
    _visual_track(state, topic)
    state["last_special_layout"] = "vocab_table"
    base_req = _format_vocab_table_request(topic, target_lang_name)
    return base_req + _avoid_recent_hint(used, topic, n_show=10), "vocab_table"


def _pick_compare(state: dict, target_lang: str, target_lang_name: str) -> tuple[str, str]:
    """Pick a 2-column compare scenario and return (request_text, 'compare')."""
    used = state.setdefault("used_compare_topics", [])
    pool = LANGUAGE_COMPARE_SCENARIOS.get(target_lang) or LANGUAGE_COMPARE_SCENARIOS.get("de", [])
    if not pool:
        return _pick_vocab_table(state, target_lang, target_lang_name)
    available = [t for t in pool if t not in used[-_DUP_WINDOW_COMPARE:]]
    if not available:
        available = pool
        used.clear()
    topic = random.choice(available)
    used.append(topic)
    state["last_special_layout"] = "compare"
    base_req = _format_compare_request(topic, target_lang_name)
    return base_req + _avoid_recent_hint(used, topic, n_show=10), "compare"
