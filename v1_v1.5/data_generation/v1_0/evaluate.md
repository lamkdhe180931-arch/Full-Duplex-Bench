# Báo cáo đánh giá benchmark v1.0

Tài liệu này mô tả cách tạo dữ liệu mô phỏng v1.0, cách chạy Gemini để sinh phản hồi, cách chuyển phản hồi audio thành transcript có timestamp, và cách đọc các chỉ số đánh giá.

input.wav  = user stem từ 0s
output.wav = agent stem từ 0s, có silence trước lúc Gemini bắt đầu nói
combined.wav = input.wav + output.wav overlay tại 0s
output.json = ASR timestamp theo timeline chung, không cộng offset nữa


Đọc input.wav.

Resample input.wav về 16kHz mono.

Chia input.wav thành các chunk audio.

Tạo recorder để sẵn sàng ghi Gemini response.

Kết nối Gemini Live API.

Sender chuẩn bị gửi chunk đầu tiên của input.wav.

Ngay trước khi gửi chunk đầu tiên, set mốc timeline:
## 1. Mục tiêu benchmark

Benchmark v1.0 dùng để kiểm tra năng lực hội thoại full-duplex của agent âm thanh trong ba tình huống cơ bản:

1. **Pause Handling**: người dùng ngập ngừng giữa câu.
2. **Smooth Turn-Taking**: người dùng nói xong và nhường lượt cho AI.
3. **User Interruption**: người dùng ngắt lời hoặc đổi ý khi AI đang phản hồi.

Mục tiêu không chỉ là kiểm tra AI có trả lời đúng nội dung hay không, mà còn kiểm tra **AI nói vào thời điểm nào**. Một agent full-duplex tốt phải biết lúc nào nên im lặng, lúc nào nên phản hồi, và lúc nào phải chuyển hướng theo input mới.

## 2. Pipeline hiện tại

Pipeline v1.0 hiện được tách thành bốn bước:

```text
template JSON
  -> TTS tạo input.wav
  -> Gemini Live tạo output.wav
  -> PhoWhisper ASR tạo output.json
  -> evaluator tính TOR / latency / rating
```

### 2.1. Template JSON

Các kịch bản đầu vào nằm tại:

```text
v1_v1.5/data_generation/v1_0/templates/
```

Ba file chính:

```text
synthetic_pause_handling.json
candor_turn_taking.json
synthetic_user_interruption.json
```

Mỗi item trong template tương ứng với một sample benchmark. Hiện mỗi template đang để một item để chạy thử nhanh, dễ trace và dễ nghe kết quả.

### 2.2. Tạo `input.wav`

Data generation chỉ tạo **âm thanh đầu vào của người dùng**. Không nhét cứng 15 giây im lặng ở cuối `input.wav` nữa.

Quy ước hiện tại:

- `pause_handling`: `input.wav = part_1 + pause + part_2`
- `turn_taking`: `input.wav = câu hỏi/câu lệnh của user`
- `user_interruption`: `input.wav = context + khoảng trống vừa đủ + interrupt`

Các file annotation sinh kèm:

```text
pause.json
turn_taking.json
interrupt.json
```

Các file này giữ timestamp sự kiện trên timeline của `input.wav`.

### 2.3. Gemini tạo `output.wav`

Gemini nhận stream `input.wav` theo thời gian thực. Khi input kết thúc, script gửi `audio_stream_end=True`, nhưng recorder vẫn tiếp tục chờ phản hồi Gemini.

Cơ chế hiện tại là **dynamic response window**:

- `output.wav` chỉ chứa phần Gemini thật sự nói.
- Nếu Gemini trả lời xong sớm, file kết thúc ngay.
- Nếu Gemini nói dài, hệ thống ghi đủ tới khi nhận `turn_complete`.
- Nếu Gemini bị kẹt quá lâu, hệ thống cắt theo giới hạn `MAX_RESPONSE_SEC`, mặc định là `30s`.

Script tạo thêm:

```text
inference_timing.json
```

File này dùng để map `output.wav` trở lại timeline gốc của input:

```json
{
  "input_duration_sec": 5.0,
  "response_start_sec": 5.8,
  "response_latency_sec": 0.8,
  "response_duration_sec": 3.2,
  "response_end_sec": 9.0,
  "max_response_sec": 30
}
```

Nếu cần nghe toàn cảnh hội thoại, script tạo thêm:

```text
combined.wav
```

Trong đó `input.wav` và `output.wav` được ghép theo `response_start_sec`.

### 2.4. PhoWhisper tạo `output.json`

PhoWhisper transcribe `output.wav` thành:

```text
output.json
```

Định dạng:

```json
{
  "text": "nội dung Gemini đã nói",
  "chunks": [
    {
      "text": "từ",
      "timestamp": [5.9, 6.1]
    }
  ]
}
```

Timestamp trong `output.json` đã được cộng offset từ `inference_timing.json`, nên evaluator đọc được thời điểm Gemini bắt đầu nói trên timeline gốc.

## 3. Notebook dùng để chạy và kiểm tra

Notebook hiện tại:

```text
v1_v1.5/data_generation/benchmark_v1_workflow.ipynb
```

Notebook có các phần chính:

1. Clone/pull repo và cài dependency.
2. Setup `GEMINI_API_KEY` và `HF_TOKEN` nếu có.
3. Tạo input/audio trung gian từng bước.
4. Test tải PhoWhisper model riêng.
5. Chạy Gemini để sinh `output.wav`.
6. Nghe trực tiếp `output.wav` và `combined.wav`.
7. Chạy PhoWhisper ASR để tạo `output.json`.
8. Hiển thị bảng evaluation trace.

Link import raw cho Kaggle:

```text
https://raw.githubusercontent.com/lamkdhe180931-arch/Full-Duplex-Bench/LamKD/v1_v1.5/data_generation/benchmark_v1_workflow.ipynb
```

## 4. Các tiêu chí đánh giá

### 4.1. Average Take Turn (TOR)

TOR đo xem AI có phát ra câu trả lời đủ dài hay không.

Quy ước:

- `TOR = 1`: AI có nói.
- `TOR = 0`: AI giữ im lặng hoặc chỉ phát ra âm thanh quá ngắn.

Ngưỡng hiện tại:

```text
duration >= 1s hoặc số từ > 3
```

Ý nghĩa theo từng task:

- `pause_handling`: lý tưởng là `TOR = 0`, vì AI không nên cướp lời khi user chỉ đang ngập ngừng.
- `turn_taking`: lý tưởng là `TOR = 1`, vì user đã nhường lượt.
- `user_interruption`: lý tưởng là `TOR = 1`, nhưng nội dung phải bám theo câu interrupt mới.

### 4.2. Average Latency

Latency đo thời gian phản xạ của AI:

```text
latency = thời điểm AI bắt đầu nói - thời điểm user dứt lời / kết thúc interrupt
```

Latency càng thấp càng tốt, nhưng không nên thấp tới mức cướp lời. Với full-duplex voice agent, latency tốt giúp hội thoại tự nhiên và tránh khoảng chết.

### 4.3. Average Rating

Rating dùng cho `user_interruption`.

Mục tiêu: đánh giá phản hồi của AI có bám theo câu interrupt mới không.

Hiện tại phần semantic rating có thể dùng Gemini với `GEMINI_API_KEY`. Thang điểm:

```text
0 = hoàn toàn không liên quan
5 = phản hồi rất đúng với ý interrupt
```

### 4.4. Logic evaluator gốc của repo fork

Repo gốc `DanielLin94144/Full-Duplex-Bench` đánh giá v1 bằng các script trong:

```text
v1_v1.5/evaluation/
```

Các chỉ số gốc được thiết kế khá gọn, chủ yếu dựa vào `output.json` do ASR tạo ra:

| Bài toán | Chỉ số gốc | Cách tính chính | Ý nghĩa |
| --- | --- | --- | --- |
| Pause Handling | `Average take turn` | Nếu `output.json` rỗng hoặc output rất ngắn thì `TOR=0`; nếu output đủ dài thì `TOR=1` | Đo xem model có cướp lượt trong tình huống pause hay không |
| Smooth Turn-Taking | `Average take turn`, `Average latency` | `latency = first_output_timestamp - turn_end` | Đo model có phản hồi sau khi user kết thúc lượt không và phản hồi nhanh hay chậm |
| User Interruption | `Average rating`, `Average take turn`, `Average latency` | LLM chấm response có liên quan interrupt không; latency lấy từ first output trừ `interrupt_end` | Đo model có trả lời sau interrupt và câu trả lời có bám theo câu interrupt không |
| Backchannel | `JSD`, `TOR`, `Frequency` | Dùng Silero VAD tìm các đoạn speech ngắn rồi so phân phối với ground-truth | Đo tần suất và phân phối backchannel có giống dữ liệu tham chiếu không |

Điểm mạnh của evaluator gốc:

- Đơn giản, dễ chạy, dễ tổng hợp số liệu.
- Có thể dùng chung cho nhiều model vì chỉ cần `output.wav` và `output.json`.
- `User Interruption` có thêm LLM rating nên đánh giá được phần liên quan nội dung.
- `Backchannel` đã dùng VAD, phù hợp hơn với việc phát hiện đoạn nói ngắn.

Điểm yếu của evaluator gốc:

- Phần lớn metric phụ thuộc vào timestamp trong `output.json`, trong khi timestamp ASR có thể lệch khi audio có silence đầu, hallucination, hoặc bị trim im lặng.
- `Pause Handling` không kiểm tra model nói ở đúng giai đoạn nào của input. Nó chỉ biết output dài hay ngắn, nên chưa phân biệt được model nói trong pause, nói khi user đang nói tiếp, hay nói sau khi user kết thúc.
- `Smooth Turn-Taking` đo latency bằng first ASR timestamp, nên có thể báo âm hoặc lệch so với cảm nhận khi nghe `combined.wav`.
- `User Interruption` chưa đo việc model có ngừng nói nhanh khi user interrupt hay không. Nó chủ yếu đo response sau interrupt, chưa đo stop behavior.
- Một số lỗi cướp lời có thể bị che mất nếu ASR/timestamp bị lọc hoặc latency âm bị ép về 0.

### 4.5. Hướng cải tiến evaluator hiện tại

Phương án cải tiến là tách rõ ba nguồn dữ liệu:

```text
VAD / inference_timing.json  -> đo hành vi thời gian: nói lúc nào, dừng lúc nào
output.json                  -> lấy nội dung agent đã nói
LLM judge                    -> chấm semantic: có đúng ý user không
```

Mỗi bài toán vẫn cần các chỉ số chung:

| Chỉ số chung | Cách hiểu | Nguồn đo phù hợp |
| --- | --- | --- |
| `response_rate` | Agent có phản hồi khi cần không | VAD trên `output.wav` hoặc `inference_timing.json` |
| `valid_response_latency` | Agent phản hồi sau mốc user hoàn tất bao lâu | VAD / `response_start_sec`, không ưu tiên ASR timestamp |
| `semantic_score` | Nội dung phản hồi có đúng ý user không | `output.json` + LLM judge |

Ngoài chỉ số chung, mỗi bài toán cần chỉ số riêng đúng bản chất:

| Bài toán | Hành vi cần đo | Chỉ số riêng đề xuất | Vì sao phù hợp hơn |
| --- | --- | --- | --- |
| Pause Handling | Agent phải tiếp tục lắng nghe khi user ngập ngừng | `pause_barge_in_rate`, `continuation_barge_in_rate`, `listen_through_success`, `final_response_latency` | Đo đúng việc agent có im lặng trong pause và chỉ trả lời sau khi user nói xong hay không |
| Smooth Turn-Taking | Agent phải đợi user kết thúc rồi trả lời nhanh | `barge_in_rate`, `smooth_turn_success`, `turn_latency` | Phân biệt rõ trả lời đúng lượt với cướp lời; latency lấy theo timeline audio thật |
| User Interruption | Agent phải dừng nhanh, nghe interrupt mới, và đổi nội dung trả lời | `stop_latency`, `interrupt_overlap_duration`, `recovery_latency`, `new_intent_score`, `old_context_contamination` | Đo được cả ba bước: dừng nói, lắng nghe, phản hồi theo ý mới |
| Backchannel | Agent chỉ nên phản hồi ngắn, không chiếm lượt | `backchannel_brevity`, `takeover_rate`, giữ thêm `JSD` và `Frequency` | Giữ ưu điểm của repo gốc nhưng bổ sung kiểm tra không biến backchannel thành một lượt trả lời dài |

Logic mới phù hợp hơn với benchmark full-duplex vì nó không chỉ hỏi "agent có nói không", mà còn hỏi:

- Agent nói có đúng thời điểm không?
- Agent có biết tiếp tục lắng nghe khi user chưa nói xong không?
- Agent có dừng nhanh khi bị interrupt không?
- Agent có chuyển sang ý mới hay vẫn bám context cũ?
- Nội dung trả lời có đúng với câu hỏi mới nhất không?

Tóm lại, evaluator gốc là một baseline tốt để chạy nhanh, nhưng evaluator cải tiến cần dùng audio timeline làm nguồn chính cho các metric thời gian. `output.json` nên dùng chủ yếu cho text và semantic rating, không nên là nguồn duy nhất để kết luận latency hoặc cướp lời.

## 5. Ý nghĩa từng bài test

### 5.1. Pause Handling

Vai trò: kiểm tra sự kiên nhẫn và khả năng hiểu câu chưa hoàn tất.

Trong thực tế, người dùng hay ngập ngừng, dừng giữa câu hoặc vừa nói vừa suy nghĩ. Bài test này kiểm tra AI có cướp lời khi thấy một khoảng im lặng ngắn hay không.

Kết quả tốt:

- AI không nói trong đoạn pause.
- `TOR` thấp.
- Không tạo `output.wav` dài hoặc transcript ảo giác.

### 5.2. Smooth Turn-Taking

Vai trò: kiểm tra tốc độ phản xạ và sự trôi chảy.

Khi user nói xong, AI cần nhận biết đúng thời điểm được nói và phản hồi nhanh vừa đủ.

Kết quả tốt:

- AI có phản hồi.
- Latency thấp.
- Nội dung phản hồi liên quan tới input.

### 5.3. User Interruption

Vai trò: kiểm tra khả năng nhún nhường, dừng ý cũ và bẻ lái theo thông tin mới.

Một agent tốt phải biết ưu tiên câu interrupt mới, thay vì tiếp tục trả lời ý cũ.

Kết quả tốt:

- AI phản hồi sau interrupt.
- Latency hợp lý.
- Rating semantic cao.

## 6. Đánh giá benchmark hiện tại

Phần này dùng để ghi nhận kết quả sau mỗi lần chạy benchmark.

### 6.1. Kết quả chạy hiện tại

| Task | Sample | TOR | Latency | Rating | Nhận xét nhanh |
| --- | --- | ---: | ---: | ---: | --- |
| Pause Handling |  |  |  |  |  |
| Smooth Turn-Taking |  |  |  |  |  |
| User Interruption |  |  |  |  |  |

### 6.2. Chất lượng audio

Ghi nhận khi nghe `input.wav`, `output.wav`, và `combined.wav`:

- `input.wav` có tự nhiên không:
- Gemini có bắt đầu nói đúng lúc không:
- `output.wav` có bị cắt cụt không:
- `output.wav` có bị dư im lặng cuối không:
- `combined.wav` có dễ nghe và đúng timeline không:

### 6.3. Chất lượng transcript ASR

Ghi nhận chất lượng `output.json`:

- PhoWhisper có nhận đúng tiếng Việt không:
- Timestamp word-level có hợp lý không:
- Có hallucination do im lặng dài không:
- Có cần đổi model ASR hoặc hậu xử lý transcript không:

### 6.4. Những điểm cần chỉnh sửa

Gợi ý các hướng chỉnh:

- Rút ngắn hoặc tăng `MAX_RESPONSE_SEC`.
- Điều chỉnh VAD của Gemini Live nếu AI nói quá sớm hoặc quá muộn.
- Tăng/giảm độ dài pause trong `synthetic_pause_handling.json`.
- Thay đổi nội dung template để tình huống tự nhiên hơn.
- Thêm nhiều speaker TTS để tránh benchmark quá đơn điệu.
- Thêm noise/RIR/off-axis cho các phiên bản v1.5.
- Kiểm tra lại evaluator nếu latency bị lệch so với cảm nhận khi nghe `combined.wav`.

### 6.5. Kết luận tạm thời

Ghi kết luận ngắn sau khi chạy:

```text
Benchmark hiện tại đã/ chưa mô phỏng tốt tình huống full-duplex vì ...
Điểm mạnh hiện tại là ...
Điểm yếu hiện tại là ...
Ưu tiên chỉnh tiếp theo là ...
```
