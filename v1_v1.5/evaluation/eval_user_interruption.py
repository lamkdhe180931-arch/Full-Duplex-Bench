import json
import os
import argparse
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from v1_timeline_metrics import (
    average,
    evaluate_user_interruption_sample,
    iter_sample_dirs,
)
from semantic_rating import (
    load_semantic_cache,
    safe_rate_interruption_semantics,
    save_semantic_cache,
)

turn_duration_threshold = 1
turn_num_words_threshold = 3


def semantic_success(row):
    previous_rating = row.get("previous_answer_rating")
    new_rating = row.get("new_intent_rating")
    leakage = row.get("old_context_leakage_score")
    return int(
        previous_rating is not None
        and new_rating is not None
        and leakage is not None
        and previous_rating >= 4
        and new_rating >= 4
        and leakage <= 1
    )


def eval_user_interruption(root_dir, client):

    if client is None:
        raise ValueError("Gemini client is required for user_interruption rating.")

    rows = []
    for file_dir in tqdm(list(iter_sample_dirs(root_dir))):
        print(f"Processing {file_dir} ...")

        out_after_interrupt_path = os.path.join(file_dir, "output.json")
        if not os.path.exists(out_after_interrupt_path):
            raise FileNotFoundError("Required file 'output.json' not found.")

        result = evaluate_user_interruption_sample(file_dir)
        result["semantic_score"] = None
        result["previous_answer_rating"] = None
        result["previous_answer_analysis"] = ""
        result["new_intent_rating"] = None
        result["new_intent_analysis"] = ""
        result["old_context_leakage_score"] = None
        result["old_context_leakage_analysis"] = ""
        result["overall_semantic_rating"] = None

        if result.get("output_text"):
            cache_path = os.path.join(file_dir, "semantic_rating.json")
            cached = load_semantic_cache(cache_path)
            if cached is not None:
                result.update(cached)
                result["semantic_score"] = cached.get("new_intent_rating")
                rows.append(result)
                continue

            for attempt in range(3):
                parsed_output = safe_rate_interruption_semantics(
                    client,
                    context_request=result["context"],
                    interrupt_request=result["interrupt"],
                    pre_interrupt_text=result["pre_interrupt_text"],
                    post_interrupt_text=result["post_interrupt_text"],
                    full_output_text=result["output_text"],
                )
                if parsed_output.get("semantic_error"):
                    if attempt == 2:
                        print(f"[WARN] Semantic rating skipped for {file_dir}: {parsed_output['semantic_error']}")
                    continue

                result.update(parsed_output)
                result["semantic_score"] = parsed_output["new_intent_rating"]
                legacy_rating = {
                    "analysis": parsed_output.get("new_intent_analysis", ""),
                    "rating": parsed_output.get("new_intent_rating"),
                    **parsed_output,
                }
                save_semantic_cache(cache_path, parsed_output)
                with open(os.path.join(file_dir, "rating.json"), "w", encoding="utf-8") as f:
                    json.dump(legacy_rating, f, ensure_ascii=False, indent=2)
                print("\n--- Semantic rating ---")
                print(f"Previous answer: {result['previous_answer_rating']}/5")
                print(f"New intent: {result['new_intent_rating']}/5")
                print(f"Old-context leakage: {result['old_context_leakage_score']}/5")
                print("-----------------------")
                break

        rows.append(result)

    avg_previous_rating = average(row["previous_answer_rating"] for row in rows)
    avg_rating = average(row["new_intent_rating"] for row in rows)
    avg_leakage = average(row["old_context_leakage_score"] for row in rows)
    avg_overall_semantic = average(row["overall_semantic_rating"] for row in rows)
    semantic_success_rate = average(semantic_success(row) for row in rows)
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
    status_previous = "tốt" if avg_previous_rating >= 4.0 else ("khá" if avg_previous_rating >= 3.0 else "kém")
    print(f"2. Previous answer quality (Trả lời câu hỏi trước): {avg_previous_rating:.2f}/5.0 ({status_previous}) - Càng cao càng tốt")

    print(f"3. New intent understanding (Điểm hiểu ý mới): {avg_rating:.2f}/5.0 ({status_rating}) - Càng cao càng tốt")

    status_leakage = "tốt" if avg_leakage <= 1.0 else ("cần chú ý" if avg_leakage <= 2.0 else "kém")
    print(f"4. Old context leakage (Dính ngữ cảnh cũ sau interrupt): {avg_leakage:.2f}/5.0 ({status_leakage}) - Càng thấp càng tốt")

    status_stop = "tốt" if stop_success_rate > 0.7 else "kém"
    print(f"5. Stop success rate (Tỉ lệ ngừng nói nhanh khi bị ngắt): {stop_success_rate:.1%} ({status_stop}) - Càng cao càng tốt")
    print(f"   - Avg stop latency: {avg_stop_latency:.3f}s; avg interrupt overlap: {avg_overlap_duration:.3f}s")

    status_listen = "tốt" if listening_success_rate > 0.7 else "kém"
    print(f"6. Listening success rate (Tỉ lệ im lặng để nghe câu ngắt lời): {listening_success_rate:.1%} ({status_listen})")

    status_lat = "không có phản hồi sau ngắt" if avg_recovery_latency is None else ("tốt" if 0 <= avg_recovery_latency <= 2.0 else "chậm")
    latency_text = "N/A" if avg_recovery_latency is None else f"{avg_recovery_latency:.3f}s"
    print(f"7. Avg recovery latency (Độ trễ phản hồi sau khi ngắt): {latency_text} ({status_lat}) - Càng sát 0 càng tốt")
    
    perfect_tests = sum(
        1
        for row in rows
        if row["post_interrupt_response_rate"]
        and row["stop_success"]
        and row["listening_success"]
        and row["recovery_latency"] is not None
        and 0 <= row["recovery_latency"] <= 2.0
        and semantic_success(row)
    )
    perfect_rate = perfect_tests / len(rows) if rows else 0.0
    status_perfect = "xuất sắc" if perfect_rate > 0.7 else "cần cải thiện"
    print(f"8. Semantic success rate (Đúng cả câu cũ, câu mới, ít dính context cũ): {semantic_success_rate:.1%}")
    print(f"9. Perfect handling rate (Tỉ lệ xử lý ngắt lời hoàn hảo): {perfect_rate:.1%} ({status_perfect})")
    print("---------------------------------------------------")
    
    return {
        "Response rate": avg_tor,
        "Context rating": avg_previous_rating,
        "Previous answer rating": avg_previous_rating,
        "New intent rating": avg_rating,
        "Old context leakage score": avg_leakage,
        "Overall semantic rating": avg_overall_semantic,
        "Semantic success rate": semantic_success_rate,
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
