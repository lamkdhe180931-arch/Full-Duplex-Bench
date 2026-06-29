import json
import re
import os
import argparse
from tqdm import tqdm
import time

turn_duration_threshold = 1
turn_num_words_threshold = 3


def parse_output(data):
    # Regular expression to match each example

    example_pattern = re.compile(
        r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)", re.DOTALL
    )

    example = {}
    # Parse the examples
    for match in example_pattern.finditer(data):
        analysis = match.group(1).strip()
        rating = match.group(2).strip()

        # Append the parsed example
        example = {"analysis": analysis, "rating": int(rating)}

    return example


def generate_gemini_rating(client, system_msg, user_msg):
    model_name = os.getenv("GEMINI_RATING_MODEL", "gemini-2.5-flash")
    prompt = f"{system_msg.strip()}\n\n{user_msg.strip()}"
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text or ""


def eval_user_interruption(root_dir, client):

    if client is None:
        raise ValueError("Gemini client is required for user_interruption rating.")

    system_msg = """
   Tình huống là người dùng và AI đang trò chuyện bằng giọng nói.
   Người dùng nói trước, sau đó AI phản hồi. Nhưng khi AI đang nói, người dùng bất ngờ ngắt lời AI.
   Nhiệm vụ của bạn là đánh giá chất lượng câu phản hồi của AI sau khi bị người dùng ngắt lời.

   Dưới đây là thang điểm đánh giá (từ 0 đến 5, 0 là tệ nhất và 5 là tốt nhất):
   - 0: Phản hồi của AI hoàn toàn không liên quan đến câu ngắt lời của người dùng.
   - 1: Phản hồi của AI không liên quan đến câu ngắt lời của người dùng.
   - 2: Phản hồi của AI hơi có chút liên quan đến câu ngắt lời.
   - 3: Phản hồi của AI có liên quan đến câu ngắt lời.
   - 4: Phản hồi của AI rất liên quan và bám sát câu ngắt lời.
   - 5: Phản hồi của AI hoàn hảo và xử lý xuất sắc câu ngắt lời.

   Đầu tiên, hãy phân tích ngắn gọn câu ngắt lời của người dùng và phản hồi của AI.
   Sau đó, bạn BẮT BUỘC phải trả về kết quả theo đúng định dạng sau:
   Analysis: [Bài phân tích của bạn bằng tiếng Việt].
   I would rate the AI's response as [Điểm số].
   """

    file_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            file_dirs.append(os.path.join(root, dir))

    score_list = []
    take_turn_list = []
    latency_list = []

    for file_dir in tqdm(sorted(file_dirs)):
        # read the json file
        for attempt in range(3):
            print(f"Processing {file_dir} ...")

            out_after_interrupt_path = os.path.join(file_dir, "output.json")
            # check must have output.json, if not, raise error
            if not os.path.exists(out_after_interrupt_path):
                raise FileNotFoundError("Required file 'output.json' not found.")

            with open(out_after_interrupt_path, "r") as f:
                out_after_interrupt = json.load(f)

            metadata_path = os.path.join(file_dir, "interrupt.json")
            is_v15 = False
            if not os.path.exists(metadata_path):
                metadata_path = os.path.join(file_dir, "metadata.json")
                if not os.path.exists(metadata_path):
                    raise FileNotFoundError("Required file 'interrupt.json' or 'metadata.json' not found.")
                is_v15 = True

            # read the json file
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if is_v15:
                in_interrupt_text = metadata["current_turn_text"]
                in_before_interrupt_text = metadata["context_text"]
                input_start_time = metadata["timestamps"][0]
                input_end_time = metadata["timestamps"][1]
            else:
                in_interrupt_text = metadata[0]["interrupt"]
                in_before_interrupt_text = metadata[0]["context"]
                input_start_time = metadata[0]["timestamp"][0]
                input_end_time = metadata[0]["timestamp"][1]
                
            # Lấy toàn bộ chunks từ kết quả ASR (Không lọc, không cắt xén)
            segments_cw = out_after_interrupt.get("chunks", [])

            # Tạo text có kèm timestamp cho AI's response để LLM dễ đánh giá
            # Ví dụ: "[0.5-0.8] một [0.8-1.2] hà [1.2-1.5] nội"
            ai_timestamped_text = " ".join([
                f"[{c['timestamp'][0]:.2f}-{c['timestamp'][1]:.2f}] {c['text']}" 
                for c in segments_cw if c.get("timestamp") and c["timestamp"][0] is not None
            ])

            # TOR and latency
            TOR = None
            latency = None

            # Tính toán trực tiếp toàn bộ theo thực tế
            if len(segments_cw) == 0:
                TOR = 0
            else:
                output_start_time = segments_cw[0]["timestamp"][0]
                duration = (
                    segments_cw[-1]["timestamp"][-1] - segments_cw[0]["timestamp"][0]
                )
                if duration < turn_duration_threshold:
                    if len(segments_cw) <= turn_num_words_threshold:
                        TOR = 0
                    else:
                        TOR = 1
                        latency = output_start_time - input_end_time
                else:
                    TOR = 1
                    latency = output_start_time - input_end_time

            take_turn_list.append(TOR)
            if TOR == 1:
                user_msg = f"""
                - Câu mồi ban đầu của user: {in_before_interrupt_text}
                - Câu ngắt lời của user (xảy ra ở đoạn [{input_start_time:.2f}-{input_end_time:.2f}] giây): {in_interrupt_text}
                - Toàn bộ phản hồi của AI (kèm mốc thời gian): {ai_timestamped_text}
                
                Hãy nhìn vào các mốc thời gian để xác định chính xác những gì AI đã nói SAU KHI người dùng ngắt lời xong ở mốc {input_end_time:.2f}s, và đánh giá chất lượng của riêng đoạn phản hồi đó.
                """

                prediction = generate_gemini_rating(client, system_msg, user_msg)
                
                # Tránh lỗi Rate Limit (429) của gói API Free Tier (giới hạn 5 req/phút)
                time.sleep(15)

                print(prediction)
                parsed_output = parse_output(prediction + "\n")

                # In chi tiết thay vì in dict thô
                print("\n--- Nhận xét của AI ---")
                print(parsed_output.get("analysis", "Không có phân tích."))
                print(f"Rating: {parsed_output.get('rating', 'N/A')}")
                print("-----------------------")

                if "rating" not in parsed_output:
                    if attempt == 2:
                        print(f"Could not parse rating for {file_dir}; skipping rating.")
                        break
                    continue
                score = parsed_output["rating"]
                score_list.append(score)

                # save the parsed_output to a json file
                with open(os.path.join(file_dir, "rating.json"), "w") as f:
                    json.dump(parsed_output, f)

                # Lấy kết quả thực tế (chấp nhận cả số âm nếu AI nói trước khi user nói xong)
                if latency is not None:
                    latency_list.append(latency)

            break

    avg_rating = sum(score_list) / len(score_list) if len(score_list) > 0 else 0.0
    avg_tor = sum(take_turn_list) / len(take_turn_list) if len(take_turn_list) > 0 else 0.0
    avg_latency = sum(latency_list) / len(latency_list) if len(latency_list) > 0 else 0.0

    print("---------------------------------------------------")
    print("[Result]")
    status_rating = "tốt" if avg_rating >= 4.0 else ("khá" if avg_rating >= 3.0 else "xấu")
    status_tt = "tốt" if avg_tor > 0.7 else "xấu"
    if avg_latency < 0:
        status_lat = "xấu (lảm nhảm ý cũ)"
    elif avg_latency <= 2.0:
        status_lat = "tốt"
    else:
        status_lat = "kém (chậm)"

    print(f"Average rating (điểm chất lượng xử lý): {avg_rating}: {status_rating} (0-5 \"càng lớn càng tốt\")")
    print(f"Average take turn (tỉ lệ phản hồi): {avg_tor}: {status_tt} (0-1 \"càng lớn càng tốt\")")
    print(f"Average latency (độ trễ ngắt lời): {avg_latency:.3f}s: {status_lat} (dương \"càng nhỏ càng tốt, âm là lảm nhảm ý cũ\")")
    print("---------------------------------------------------")
    
    good_tests = sum(1 for t, l, r in zip(take_turn_list, latency_list, score_list) if t == 1 and l >= 0 and r >= 3.0)
    return {
        "Average rating": avg_rating,
        "Average take turn": avg_tor,
        "Average latency": avg_latency,
        "Total tests": len(score_list),
        "Good tests (TOR=1 & lat>=0 & rating>=3)": good_tests
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()

    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment.")
    eval_user_interruption(args.root_dir, genai.Client(api_key=gemini_api_key))
