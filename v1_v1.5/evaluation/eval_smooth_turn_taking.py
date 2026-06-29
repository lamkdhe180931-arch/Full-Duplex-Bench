import os
import re
import argparse
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from v1_timeline_metrics import (
    average,
    evaluate_turn_taking_sample,
    iter_sample_dirs,
)

turn_duration_threshold = 1
turn_num_words_threshold = 3


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s\[\]]", "", text)


def round_to_quarter(number):
    return round(number * 4) / 4


def eval_smooth_turn_taking(data_dir):
    rows = []
    for sample_dir in tqdm(list(iter_sample_dirs(data_dir)), desc="evaluate"):
        turn_path = os.path.join(sample_dir, "turn_taking.json")
        output_path = os.path.join(sample_dir, "output.wav")
        if not os.path.exists(turn_path):
            raise FileNotFoundError(f"Required file '{turn_path}' not found.")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Required file '{output_path}' not found.")
        result = evaluate_turn_taking_sample(sample_dir)
        rows.append(result)
        print(sample_dir)
        print(f"agent_start={result['agent_start']} user_end={result['user_end']}")
        print(f"barge_in={result['barge_in']} latency={result['valid_response_latency']}")

    total_tests = len(rows)
    average_take_turn = average(row["response_rate"] for row in rows)
    post_turn_response_rate = average(row["post_turn_response_rate"] for row in rows)
    barge_in_rate = average(row["barge_in"] for row in rows)
    valid_latencies = [row["valid_response_latency"] for row in rows if row["valid_response_latency"] is not None]
    avg_valid_latency = average(valid_latencies) if valid_latencies else None
    smooth_tests = sum(row["smooth_turn_success"] for row in rows)
    smooth_rate = smooth_tests / total_tests if total_tests else 0.0

    print("---------------------------------------------------")
    print("[Result: Smooth Turn Taking (Luân phiên lượt lời)]")
    status_tt = "tốt" if average_take_turn > 0.7 else "kém"
    print(f"1. Response rate (Tỉ lệ chịu phản hồi): {average_take_turn:.1%} ({status_tt}) - Càng cao càng tốt")

    status_barge = "tốt" if barge_in_rate < 0.2 else "kém"
    print(f"2. Barge-in rate (Tỉ lệ cướp lời sớm khi chưa xong lượt): {barge_in_rate:.1%} ({status_barge}) - Càng thấp càng tốt")

    status_lat = "không có phản hồi hợp lệ" if avg_valid_latency is None else ("tốt" if 0 <= avg_valid_latency <= 1.5 else "chậm")
    latency_text = "N/A" if avg_valid_latency is None else f"{avg_valid_latency:.3f}s"
    print(f"3. Avg valid latency (Độ trễ trung bình hợp lệ): {latency_text} ({status_lat}) - Càng sát 0 càng tốt")

    status_smooth = "xuất sắc" if smooth_rate > 0.7 else "cần cải thiện"
    print(f"4. Smooth turn rate (Tỉ lệ luân phiên mượt mà hoàn hảo): {smooth_rate:.1%} ({status_smooth})")
    print(f"5. Post-turn response rate (Tỉ lệ phản hồi sau khi user kết thúc): {post_turn_response_rate:.1%}")
    print("---------------------------------------------------")
    
    return {
        "Response rate": average_take_turn,
        "Post-turn response rate": post_turn_response_rate,
        "Barge-in rate": barge_in_rate,
        "Avg valid latency": avg_valid_latency,
        "Smooth turn rate": smooth_rate,
        "Total tests": total_tests,
        "Perfect tests (smooth)": smooth_tests,
        "Per-sample": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()

    eval_smooth_turn_taking(args.root_dir)
