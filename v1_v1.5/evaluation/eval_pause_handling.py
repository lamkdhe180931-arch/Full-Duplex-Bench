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
    evaluate_pause_sample,
    iter_sample_dirs,
)

turn_duration_threshold = 1
turn_num_words_threshold = 3


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s\[\]]", "", text)


def eval_pause_handling(data_dir):
    rows = []
    for sample_dir in tqdm(list(iter_sample_dirs(data_dir)), desc="evaluate"):
        pause_path = os.path.join(sample_dir, "pause.json")
        output_path = os.path.join(sample_dir, "output.wav")
        if not os.path.exists(pause_path):
            raise FileNotFoundError(f"Required file '{pause_path}' not found.")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Required file '{output_path}' not found.")
        result = evaluate_pause_sample(sample_dir)
        rows.append(result)
        print(sample_dir)
        print(
            "pause_barge_in={pause_barge_in} continuation_barge_in={continuation_barge_in} "
            "listen_success={listen_through_success} latency={valid_response_latency}".format(**result)
        )

    total_tests = len(rows)
    listen_rate = average(row["listen_through_success"] for row in rows)
    barge_in_rate = average(row["barge_in"] for row in rows)
    pause_barge_in_rate = average(row["pause_barge_in"] for row in rows)
    continuation_barge_in_rate = average(row["continuation_barge_in"] for row in rows)
    response_rate = average(row["response_rate"] for row in rows)
    valid_latencies = [row["valid_response_latency"] for row in rows if row["valid_response_latency"] is not None]
    avg_latency = average(valid_latencies) if valid_latencies else None
    perfect_tests = sum(row["final_response_success"] for row in rows)
    final_success_rate = perfect_tests / total_tests if total_tests else 0.0

    print("---------------------------------------------------")
    print("[Result: Pause Handling (Xử lý khoảng lặng)]")
    status_silence = "tốt" if listen_rate > 0.7 else "kém"
    print(f"1. Listen-through rate (Tỉ lệ tiếp tục lắng nghe đến hết input): {listen_rate:.1%} ({status_silence}) - Càng cao càng tốt")
    
    status_tor = "tốt" if barge_in_rate < 0.3 else "kém"
    print(f"2. Barge-in rate (Tỉ lệ cướp lời sai trước khi user nói xong): {barge_in_rate:.1%} ({status_tor}) - Càng thấp càng tốt")
    print(f"   - Trong pause: {pause_barge_in_rate:.1%}; trong phần user nói tiếp: {continuation_barge_in_rate:.1%}")
    
    status_lat = "không có phản hồi hợp lệ" if avg_latency is None else ("tốt" if 0 <= avg_latency <= 1.5 else "chậm")
    latency_text = "N/A" if avg_latency is None else f"{avg_latency:.3f}s"
    print(f"3. Avg valid latency (Độ trễ trả lời hợp lệ ở cuối câu): {latency_text} ({status_lat}) - Càng sát 0 càng tốt")
    print(f"4. Final response success rate (Nghe hết rồi trả lời nhanh): {final_success_rate:.1%}")
    print("---------------------------------------------------")
    
    return {
        "Silence rate": listen_rate,
        "Listen-through rate": listen_rate,
        "Barge-in rate": barge_in_rate,
        "Pause barge-in rate": pause_barge_in_rate,
        "Continuation barge-in rate": continuation_barge_in_rate,
        "Response rate": response_rate,
        "Avg valid latency": avg_latency,
        "Final response success rate": final_success_rate,
        "Total tests": total_tests,
        "Perfect tests (TOR=0)": sum(row["listen_through_success"] for row in rows),
        "Perfect tests (listen+respond)": perfect_tests,
        "Per-sample": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()

    eval_pause_handling(args.root_dir)
