import json
import re
import os
import argparse
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
import time

from v1_timeline_metrics import (
    average,
    evaluate_user_interruption_sample,
    iter_sample_dirs,
)

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

    rows = []
    for file_dir in tqdm(list(iter_sample_dirs(root_dir))):
        print(f"Processing {file_dir} ...")

        out_after_interrupt_path = os.path.join(file_dir, "output.json")
        if not os.path.exists(out_after_interrupt_path):
            raise FileNotFoundError("Required file 'output.json' not found.")

        result = evaluate_user_interruption_sample(file_dir)
        result["semantic_score"] = None

        if result["post_interrupt_response_rate"]:
            user_msg = f"""
            - Câu mồi ban đầu của user: {result['context']}
            - Câu ngắt lời của user (xảy ra ở đoạn [{result['interrupt_start']:.2f}-{result['interrupt_end']:.2f}] giây): {result['interrupt']}
            - Đoạn AI nói SAU KHI user ngắt lời xong: {result['post_interrupt_text']}
            - Toàn bộ transcript AI để đối chiếu nhiễm context cũ: {result['output_text']}

            Hãy chỉ đánh giá đoạn AI nói sau mốc {result['interrupt_end']:.2f}s:
            - Có theo đúng ý hỏi mới không?
            - Có còn tiếp tục bám ý cũ trước khi bị ngắt không?
            """

            for attempt in range(3):
                prediction = generate_gemini_rating(client, system_msg, user_msg)
                time.sleep(15)

                print(prediction)
                parsed_output = parse_output(prediction + "\n")
                print("\n--- Nhận xét của AI ---")
                print(parsed_output.get("analysis", "Không có phân tích."))
                print(f"Rating: {parsed_output.get('rating', 'N/A')}")
                print("-----------------------")

                if "rating" not in parsed_output:
                    if attempt == 2:
                        print(f"Could not parse rating for {file_dir}; skipping rating.")
                    continue

                result["semantic_score"] = parsed_output["rating"]
                with open(os.path.join(file_dir, "rating.json"), "w", encoding="utf-8") as f:
                    json.dump(parsed_output, f, ensure_ascii=False, indent=2)
                break

        rows.append(result)

    avg_rating = average(row["semantic_score"] for row in rows)
    avg_tor = average(row["post_interrupt_response_rate"] for row in rows)
    recovery_latencies = [row["recovery_latency"] for row in rows if row["recovery_latency"] is not None]
    avg_recovery_latency = average(recovery_latencies) if recovery_latencies else None
    avg_stop_latency = average(row["stop_latency"] for row in rows)
    avg_overlap_duration = average(row["interrupt_overlap_duration"] for row in rows)
    stop_success_rate = average(row["stop_success"] for row in rows)
    listening_success_rate = average(row["listening_success"] for row in rows)
    overlap_rate = average(row["interrupt_overlap"] for row in rows)

    print("---------------------------------------------------")
    print("[Result: User Interruption (Xử lý khi bị ngắt lời)]")
    status_tt = "tốt" if avg_tor > 0.7 else "kém"
    print(f"1. Post-interrupt response rate (Tỉ lệ phản hồi sau ngắt lời): {avg_tor:.1%} ({status_tt}) - Càng cao càng tốt")

    status_rating = "tốt" if avg_rating >= 4.0 else ("khá" if avg_rating >= 3.0 else "kém")
    print(f"2. New intent understanding (Điểm hiểu ý mới): {avg_rating:.2f}/5.0 ({status_rating}) - Càng cao càng tốt")

    status_stop = "tốt" if stop_success_rate > 0.7 else "kém"
    print(f"3. Stop success rate (Tỉ lệ ngừng nói nhanh khi bị ngắt): {stop_success_rate:.1%} ({status_stop}) - Càng cao càng tốt")
    print(f"   - Avg stop latency: {avg_stop_latency:.3f}s; avg interrupt overlap: {avg_overlap_duration:.3f}s")

    status_listen = "tốt" if listening_success_rate > 0.7 else "kém"
    print(f"4. Listening success rate (Tỉ lệ im lặng để nghe câu ngắt lời): {listening_success_rate:.1%} ({status_listen})")

    status_lat = "không có phản hồi sau ngắt" if avg_recovery_latency is None else ("tốt" if 0 <= avg_recovery_latency <= 2.0 else "chậm")
    latency_text = "N/A" if avg_recovery_latency is None else f"{avg_recovery_latency:.3f}s"
    print(f"5. Avg recovery latency (Độ trễ phản hồi sau khi ngắt): {latency_text} ({status_lat}) - Càng sát 0 càng tốt")
    
    perfect_tests = sum(
        1
        for row in rows
        if row["post_interrupt_response_rate"]
        and row["stop_success"]
        and row["listening_success"]
        and row["recovery_latency"] is not None
        and 0 <= row["recovery_latency"] <= 2.0
        and (row["semantic_score"] or 0) >= 4.0
    )
    perfect_rate = perfect_tests / len(rows) if rows else 0.0
    status_perfect = "xuất sắc" if perfect_rate > 0.7 else "cần cải thiện"
    print(f"6. Perfect handling rate (Tỉ lệ xử lý ngắt lời hoàn hảo): {perfect_rate:.1%} ({status_perfect})")
    print("---------------------------------------------------")
    
    return {
        "Response rate": avg_tor,
        "Context rating": avg_rating,
        "New intent rating": avg_rating,
        "Barge-in rate": overlap_rate,
        "Interruption overlap rate": overlap_rate,
        "Avg interrupt overlap duration": avg_overlap_duration,
        "Stop success rate": stop_success_rate,
        "Avg stop latency": avg_stop_latency,
        "Listening success rate": listening_success_rate,
        "Avg valid latency": avg_recovery_latency,
        "Avg recovery latency": avg_recovery_latency,
        "Perfect handling rate": perfect_rate,
        "Total tests": len(rows),
        "Perfect tests (TOR=1 & lat>=0 & rating>=4)": perfect_tests,
        "Per-sample": rows,
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
