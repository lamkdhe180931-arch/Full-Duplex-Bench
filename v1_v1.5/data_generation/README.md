# Vietnamese Data Generation for Full-Duplex-Bench v1.5

Thư mục này chứa các kịch bản (scripts) và công cụ phục vụ việc sinh tự động dữ liệu đánh giá Tiếng Việt (Vietnamese Benchmark Dataset) cho mô hình Full-Duplex Spoken Dialogue theo tiêu chuẩn của **Full-Duplex-Bench v1.5**.

## Cấu trúc đề xuất (Proposed Structure)

Để tái sử dụng tối đa mã nguồn (vì cả hai phiên bản đều dùng chung công cụ sinh giọng nói TTS và xử lý âm thanh), chúng ta nên cấu trúc thư mục dạng dùng chung lõi (core) nhưng chia kịch bản riêng cho v1_0 và v1_5 để khớp chuẩn cấu trúc tập dữ liệu đầu ra:

```
data_generation/
├── README.md               # Tài liệu này
├── requirements.txt        # Thư viện phục vụ sinh dữ liệu (TTS, pydub, soundfile, ...)
│
├── core/                   # Các module xử lý lõi dùng chung
│   ├── tts_generator.py    # Sinh giọng nói từ văn bản (Vietnamese TTS)
│   └── audio_mixer.py      # Trộn đè audio, chèn khoảng lặng (silence injection)
│
├── v1_0/                   # Dành riêng cho v1.0 (Turn-Taking)
│   ├── templates/          # Text kịch bản (synthetic_pause_handling, candor_turn_taking, synthetic_user_interruption)
│   └── generate_v1_0.py    # Script chạy tạo dữ liệu cho v1.0 (chèn khoảng lặng/cắt ghép)
│
└── v1_5/                   # Dành riêng cho v1.5 (Overlap)
    ├── templates/          # Text kịch bản (user_interruption, user_backchannel, talking_to_other, background_speech)
    └── generate_v1_5.py    # Script chạy tạo dữ liệu cho v1.5 (trộn đè âm thanh theo mốc thời gian)
```


## Các bước triển khai tiếp theo (Next Steps)

1. **Chuẩn bị file `requirements.txt`**: Cài đặt các thư viện cần thiết như `pydub`, `soundfile`, `edge-tts` để xử lý âm thanh và chuyển văn bản thành giọng nói.
2. **Thiết kế kịch bản Tiếng Việt**: Xây dựng danh sách các câu hội thoại tiếng Việt mẫu khớp chuẩn đầu vào trong các file JSON ở thư mục `v1_0/templates/` và `v1_5/templates/`.
3. **Hiện thực hóa `tts_generator.py`**: Viết mã nguồn gọi công cụ Text-to-Speech (TTS) để tạo ra các đoạn âm thanh giọng Việt tự nhiên.
4. **Hiện thực hóa `audio_mixer.py`**: Viết mã nguồn ghép đè các tệp âm thanh ở các mốc thời gian chỉ định (`timestamps`), xuất ra tệp `input.wav` (noisy) và `clean_input.wav` kèm theo tệp chú thích chuẩn (`turn_taking.json`, `interrupt.json`, hoặc `metadata.json`).
5. **Chạy các script generate**: Chạy `generate_v1_0.py` và `generate_v1_5.py` để quét các template và sinh ra thư mục dữ liệu hoàn chỉnh tương thích với cấu trúc của Full-Duplex-Bench.

