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
    read_json,
)
from semantic_rating import rate_answer_relevance

turn_duration_threshold = 1
turn_num_words_threshold = 3


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s\[\]]", "", text)


def round_to_quarter(number):
    return round(number * 4) / 4


def _turn_user_request(sample_dir):
    item = (read_json(os.path.join(sample_dir, "turn_taking.json"), [{}]) or [{}])[0]
    text = item.get("user_text") or item.get("text") or ""
    if text and not text.startswith("["):
        return text
    input_data = read_json(os.path.join(sample_dir, "input.json"), {})
    return input_data.get("text", "")


def _semantic_success(row):
    rating = row.get("answer_relevance_rating")
    return int(rating is not None and rating >= 4)


def eval_smooth_turn_taking(data_dir, client=None):
    rows = []
    for sample_dir in tqdm(list(iter_sample_dirs(data_dir)), desc="evaluate"):
        turn_path = os.path.join(sample_dir, "turn_taking.json")
        output_path = os.path.join(sample_dir, "output.wav")
        if not os.path.exists(turn_path):
            raise FileNotFoundError(f"Required file '{turn_path}' not found.")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Required file '{output_path}' not found.")
        result = evaluate_turn_taking_sample(sample_dir)
        result["answer_relevance_rating"] = None
        result["answer_relevance_analysis"] = ""
        user_request = _turn_user_request(sample_dir)
        if client is not None and user_request and result.get("output_text"):
            semantic = rate_answer_relevance(
                client,
                task_name="Smooth Turn Taking",
                user_request=user_request,
                answer_text=result["output_text"],
                extra_context="Chỉ chấm câu trả lời của AI sau khi user kết thúc lượt nói.",
            )
            result.update(semantic)
            with open(os.path.join(sample_dir, "answer_relevance.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(semantic, f, ensure_ascii=False, indent=2)
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
    relevance_scores = [row["answer_relevance_rating"] for row in rows if row["answer_relevance_rating"] is not None]
    avg_relevance = average(relevance_scores) if relevance_scores else None
    semantic_enabled = bool(relevance_scores)
    on_topic_rate = average(_semantic_success(row) for row in rows) if semantic_enabled else None
    semantic_success_tests = sum(_semantic_success(row) for row in rows) if semantic_enabled else 0
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

    relevance_text = "N/A" if avg_relevance is None else f"{avg_relevance:.2f}/5.0"
    print(f"4. Answer relevance score (Đúng trọng tâm): {relevance_text}")
    on_topic_text = "N/A" if on_topic_rate is None else f"{on_topic_rate:.1%}"
    print(f"5. On-topic rate (Tỉ lệ trả lời đúng trọng tâm): {on_topic_text}")
    print(f"6. Post-turn response rate (Tỉ lệ phản hồi sau khi user kết thúc): {post_turn_response_rate:.1%}")
    print("---------------------------------------------------")
    
    return {
        "Response rate": average_take_turn,
        "Post-turn response rate": post_turn_response_rate,
        "Barge-in rate": barge_in_rate,
        "Avg valid latency": avg_valid_latency,
        "Answer relevance score": avg_relevance,
        "On-topic rate": on_topic_rate,
        "Semantic success tests": semantic_success_tests,
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
