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
    read_json,
)
from semantic_rating import (
    load_semantic_cache,
    safe_rate_answer_relevance,
    save_semantic_cache,
)

turn_duration_threshold = 1
turn_num_words_threshold = 3


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s\[\]]", "", text)


def _pause_user_request(sample_dir):
    item = (read_json(os.path.join(sample_dir, "pause.json"), [{}]) or [{}])[0]
    if item.get("user_text"):
        return item["user_text"]
    if item.get("part_1") or item.get("part_2"):
        return " ".join(text for text in [item.get("part_1"), item.get("part_2")] if text).strip()
    input_data = read_json(os.path.join(sample_dir, "input.json"), {})
    return input_data.get("text", "")


def _semantic_success(row):
    rating = row.get("answer_relevance_rating")
    return int(rating is not None and rating >= 4)


def eval_pause_handling(data_dir, client=None):
    rows = []
    for sample_dir in tqdm(list(iter_sample_dirs(data_dir)), desc="evaluate"):
        pause_path = os.path.join(sample_dir, "pause.json")
        output_path = os.path.join(sample_dir, "output.wav")
        if not os.path.exists(pause_path):
            raise FileNotFoundError(f"Required file '{pause_path}' not found.")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Required file '{output_path}' not found.")
        result = evaluate_pause_sample(sample_dir)
        result["answer_relevance_rating"] = None
        result["answer_relevance_analysis"] = ""
        user_request = _pause_user_request(sample_dir)
        if client is not None and user_request and result.get("output_text"):
            cache_path = os.path.join(sample_dir, "answer_relevance.json")
            semantic = load_semantic_cache(cache_path)
            if semantic is None:
                semantic = safe_rate_answer_relevance(
                    client,
                    task_name="Pause Handling",
                    user_request=user_request,
                    answer_text=result["output_text"],
                    extra_context="User có một khoảng ngập ngừng giữa câu; chỉ chấm câu trả lời sau khi user nói xong toàn bộ yêu cầu.",
                )
                if not semantic.get("semantic_error"):
                    save_semantic_cache(cache_path, semantic)
                else:
                    print(f"[WARN] Semantic rating skipped for {sample_dir}: {semantic['semantic_error']}")
            result.update(semantic)
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
    relevance_scores = [row["answer_relevance_rating"] for row in rows if row["answer_relevance_rating"] is not None]
    avg_relevance = average(relevance_scores) if relevance_scores else None
    semantic_enabled = bool(relevance_scores)
    on_topic_rate = average(_semantic_success(row) for row in rows) if semantic_enabled else None
    perfect_tests = sum(
        row["final_response_success"] and (_semantic_success(row) if semantic_enabled else 1)
        for row in rows
    )
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
    relevance_text = "N/A" if avg_relevance is None else f"{avg_relevance:.2f}/5.0"
    print(f"4. Answer relevance score (Đúng trọng tâm): {relevance_text}")
    on_topic_text = "N/A" if on_topic_rate is None else f"{on_topic_rate:.1%}"
    print(f"5. On-topic rate (Tỉ lệ đúng trọng tâm): {on_topic_text}")
    print(f"6. Final response success rate (Nghe hết rồi trả lời nhanh và đúng trọng tâm): {final_success_rate:.1%}")
    print("---------------------------------------------------")
    
    return {
        "Silence rate": listen_rate,
        "Listen-through rate": listen_rate,
        "Barge-in rate": barge_in_rate,
        "Pause barge-in rate": pause_barge_in_rate,
        "Continuation barge-in rate": continuation_barge_in_rate,
        "Response rate": response_rate,
        "Avg valid latency": avg_latency,
        "Answer relevance score": avg_relevance,
        "On-topic rate": on_topic_rate,
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
