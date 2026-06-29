# Hướng dẫn tạo dữ liệu chuẩn cho Full-Duplex-Bench v1.0

Tài liệu này định nghĩa các ràng buộc (Constraints) cực kỳ nghiêm ngặt bắt buộc phải tuân thủ khi tạo thêm dữ liệu (thêm mẫu vào các file JSON templates) cho bộ Benchmark V1. Việc vi phạm các ràng buộc này sẽ làm mất đi tính công bằng và chính xác khi đánh giá các mô hình Voice AI.

Dưới đây là các quy tắc cho 3 task (bài test) chính:

---

## 1. Bài test Xử lý khoảng lặng (Pause Handling)
*File template tương ứng: `synthetic_pause_handling.json`*

Bài test này lừa AI bằng những khoảng ngập ngừng. Để tạo ra cái "bẫy" này đúng chuẩn, data phải thỏa mãn:

*   **Ràng buộc về Ngữ nghĩa (Semantic Incompleteness):** 
    *   Vị trí cắt câu (giữa `part_1` và `part_2`) **tuyệt đối không được** nằm ở cuối một ý trọn vẹn (không được nằm ở TRP - Transition Relevance Place).
    *   Nó bắt buộc phải nằm ở lưng chừng câu, ví dụ: giữa chủ ngữ - vị ngữ, trước liên từ (và, hoặc, nhưng...), trước giới từ (ở, tại, để...).
    *   *✅ Ví dụ ĐÚNG:* "Tôi muốn đặt một vé... [lặng 0.5s]... đi Hà Nội." (Câu chưa xong, AI phải đợi).
    *   *❌ Ví dụ SAI:* "Tôi muốn đi Hà Nội... [lặng 0.5s]... vào ngày mai." (Vế đầu đã đủ nghĩa, AI nhảy vào trả lời là hợp lý, đánh trượt AI là sai).
*   **Ràng buộc về Âm điệu (Prosodic Constraint):** Nửa câu đầu (`part_1`) khi đọc bằng TTS **không được có âm điệu chùng xuống (falling intonation)**. Giọng đọc nên được cấu hình lơ lửng, kéo dài hoặc giữ ngang tone để báo hiệu rõ ràng là câu chưa kết thúc.
*   **Ràng buộc về Độ dài (Duration Constraint - Mô phỏng thực tế):** 
    *   Để tạo ra tập dữ liệu chân thực nhất, khoảng ngập ngừng chèn vào (qua biến `pause_duration_sec` trong JSON) **bắt buộc phải nằm trong khoảng từ 0.35 giây đến 0.65 giây (350ms - 650ms)**.
    *   **Tuyệt đối không** sử dụng các khoảng lặng dài phi thực tế như 1.5 hay 2.0 giây. Đây là bài test gắt gao nhất yêu cầu AI phải phản xạ cực nhanh nhưng vẫn hiểu đúng ngữ nghĩa lưng chừng của câu.

---

## 2. Bài test Luân phiên mượt mà (Smooth Turn-Taking)
*File template tương ứng: `candor_turn_taking.json`*

Bài test này đo độ trễ (latency) của AI ở các lượt hội thoại hỏi - đáp bình thường.

*   **Ràng buộc về Tính trọn vẹn (Completeness):** Câu đầu vào bắt buộc phải hoàn chỉnh 100% về mặt ngữ pháp, ngữ nghĩa và ngữ điệu (ví dụ: lên giọng ở cuối câu hỏi, xuống giọng kết thúc ở câu trần thuật). Nó phải phát ra tín hiệu "nhường lời" (yielding the floor) cực kỳ rõ ràng tới AI.
*   **Ràng buộc về Độ dài câu hỏi:** Không nên đặt câu hỏi quá ngắn (dưới 1 giây) vì dễ bị trùng lẫn với tiếng backchannel, cũng không nên quá dài dòng lê thê làm mất thời gian tổng hợp âm thanh. Lý tưởng là câu thoại từ 2 đến 5 giây.

---

## 3. Bài test Người dùng ngắt lời (User Interruption)
*File template tương ứng: `synthetic_user_interruption.json`*

Bài test này đo lường khả năng AI xử lý khi bị chen ngang (Barge-in).

*   **Ràng buộc Đảo hướng Ngữ nghĩa (Semantic Shift / Pivot Constraint):** 
    *   Câu ngắt lời (`interrupt`) **bắt buộc phải mâu thuẫn, thay đổi thông số, hoặc chuyển hẳn sang một chủ đề mới** so với câu gốc (`context`). 
    *   *Tại sao?* Vì nếu câu ngắt lời chỉ là câu tán thành ("Ừ, đúng rồi", "Làm đi") thì AI chỉ việc nói tiếp, bài test sẽ chấm sai. AI phải bị ép vào thế: "Dừng câu cũ lại, vứt bỏ luồng suy nghĩ cũ, và trả lời theo ý mới".
*   **Ràng buộc về Mốc thời gian (Timing Constraint):** 
    *   Biến `interrupt_delay_sec` (thời gian trễ trước khi chen ngang) không được quá nhỏ.
    *   Phải cài đặt độ trễ thường từ **2.0 đến 3.0 giây**. Chừa khoảng trống vài giây đầu tiên để AI "cắn câu" (tức là AI đã kịp tạo ra một vài từ của câu trả lời cho ý cũ). Ngắt lời khi AI ĐANG nói mới là ngắt lời thực sự (Barge-in).
*   **Ràng buộc về Xung đột Hội thoại (Dialogue Conflict):** 
    *   Nội dung của câu `context` gốc phải đủ phức tạp hoặc gợi mở để **chắc chắn AI sẽ phải đưa ra một câu trả lời dài** (ví dụ: yêu cầu AI giải thích, liệt kê, hoặc hướng dẫn 1 quy trình). 
    *   Nếu AI trả lời quá ngắn (nói mất có 1 giây là xong câu trả lời), người dùng chưa kịp ngắt lời ở giây thứ 2.5 thì AI đã nói xong mất rồi, dẫn đến tình trạng "Ngắt lời trượt" (Missed Interruption).
