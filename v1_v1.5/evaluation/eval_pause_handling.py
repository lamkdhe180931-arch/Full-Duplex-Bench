import os
import json
import re
import argparse
from tqdm import tqdm

turn_duration_threshold = 1
turn_num_words_threshold = 3


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s\[\]]", "", text)


def eval_pause_handling(data_dir):
    audio_output_files = []

    for folder in os.listdir(data_dir):
        if folder.endswith(".DS_Store"):
            continue
        if folder.endswith(".md"):
            continue

        for file_o in os.listdir(os.path.join(data_dir, folder)):
            if file_o.endswith("output.json"):
                audio_output_files.append(os.path.join(data_dir, folder, file_o))

    take_turn_list = []
    latency_list = []

    for audio_output_file in tqdm(audio_output_files, desc="evaluate"):

        TOR = None
        latency = None
        duration = 0.0

        # if audio_output_file is not found, raise an error
        if not os.path.exists(audio_output_file):
            raise FileNotFoundError(f"Required file '{audio_output_file}' not found.")

        with open(audio_output_file, "r") as f:
            output_data = json.load(f)
            segments_cw = output_data["chunks"]

        # Load pause.json to find when the pause ends
        pause_file = os.path.join(os.path.dirname(audio_output_file), "pause.json")
        pause_end_time = float("inf")
        if os.path.exists(pause_file):
            with open(pause_file, "r") as f:
                pause_data = json.load(f)
            pause_end_time = pause_data[0]["timestamp"][1]

        # if no transcription from CrisperWhisper， means model does not take turn
        if len(segments_cw) == 0:
            TOR = 0
        else:
            output_start_time = segments_cw[0]["timestamp"][0]
            if output_start_time < pause_end_time:
                # Phản hồi trước khi khoảng lặng kết thúc (Cướp lời sai)
                if segments_cw[-1]["timestamp"][-1] == None:
                    duration = (
                        segments_cw[-1]["timestamp"][0] - segments_cw[0]["timestamp"][0]
                    )
                else:
                    duration = (
                        segments_cw[-1]["timestamp"][-1] - segments_cw[0]["timestamp"][0]
                    )
                if duration < turn_duration_threshold:
                    if len(segments_cw) <= turn_num_words_threshold:
                        TOR = 0
                    else:
                        TOR = 1
                else:
                    TOR = 1
            else:
                # Phản hồi ngoan ngoãn sau khi khoảng lặng đã qua (hoặc ở cuối bài)
                TOR = 0
                
                # Tính độ trễ (latency) cho lần trả lời đúng luật này
                timing_file = os.path.join(os.path.dirname(audio_output_file), "inference_timing.json")
                if os.path.exists(timing_file):
                    with open(timing_file, "r") as f:
                        timing_data = json.load(f)
                    input_duration_sec = timing_data.get("input_duration_sec")
                    if input_duration_sec is not None:
                        latency = output_start_time - input_duration_sec

        take_turn_list.append(TOR)
        if TOR == 0 and latency is not None:
            latency_list.append(latency)

    average_take_turn = sum(take_turn_list) / len(take_turn_list) if take_turn_list else 0.0
    silence_rate = 1.0 - average_take_turn

    avg_latency = sum(latency_list) / len(latency_list) if latency_list else 0.0

    print("---------------------------------------------------")
    print("[Result: Pause Handling (Xử lý khoảng lặng)]")
    status_silence = "tốt" if silence_rate > 0.7 else "kém"
    print(f"1. Silence rate (Tỉ lệ giữ im lặng thành công): {silence_rate:.1%} ({status_silence}) - Càng cao càng tốt")
    
    status_tor = "tốt" if average_take_turn < 0.3 else "kém"
    print(f"2. Barge-in rate (Tỉ lệ cướp lời sai): {average_take_turn:.1%} ({status_tor}) - Càng thấp càng tốt")
    
    status_lat = "tốt" if 0 <= avg_latency <= 1.5 else "chậm"
    print(f"3. Avg valid latency (Độ trễ trả lời hợp lệ ở cuối câu): {avg_latency:.3f}s ({status_lat}) - Càng sát 0 càng tốt")
    print("---------------------------------------------------")
    
    return {
        "Silence rate": silence_rate,
        "Barge-in rate": average_take_turn,
        "Avg valid latency": avg_latency,
        "Total tests": len(take_turn_list),
        "Perfect tests (TOR=0)": take_turn_list.count(0)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()

    eval_pause_handling(args.root_dir)
