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
    barge_in_durations = []

    for audio_output_file in tqdm(audio_output_files, desc="evaluate"):

        TOR = None
        duration = 0.0

        # if audio_output_file is not found, raise an error
        if not os.path.exists(audio_output_file):
            raise FileNotFoundError(f"Required file '{audio_output_file}' not found.")

        with open(audio_output_file, "r") as f:
            output_data = json.load(f)
            segments_cw = output_data["chunks"]

        # if no transcription from CrisperWhisper， means model does not take turn
        if len(segments_cw) == 0:
            TOR = 0
        else:
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

        take_turn_list.append(TOR)
        if TOR == 1:
            barge_in_durations.append(duration)

    average_take_turn = sum(take_turn_list) / len(take_turn_list) if take_turn_list else 0.0
    silence_rate = 1.0 - average_take_turn
    avg_barge_in_dur = sum(barge_in_durations) / len(barge_in_durations) if barge_in_durations else 0.0

    print("---------------------------------------------------")
    print("[Result: Pause Handling (Xử lý khoảng lặng)]")
    status_silence = "tốt" if silence_rate > 0.7 else "kém"
    print(f"1. Silence rate (Tỉ lệ giữ im lặng thành công): {silence_rate:.1%} ({status_silence}) - Càng cao càng tốt")
    
    status_tor = "tốt" if average_take_turn < 0.3 else "kém"
    print(f"2. Barge-in rate (Tỉ lệ cướp lời sai): {average_take_turn:.1%} ({status_tor}) - Càng thấp càng tốt")
    
    if barge_in_durations:
        print(f"3. Avg barge-in duration (Độ dài lảm nhảm trung bình khi sai): {avg_barge_in_dur:.2f}s - Càng ngắn càng tốt")
    else:
        print("3. Avg barge-in duration: Không có lỗi cướp lời! (Tuyệt vời)")
    print("---------------------------------------------------")
    
    return {
        "Silence rate": silence_rate,
        "Barge-in rate": average_take_turn,
        "Avg barge-in duration": avg_barge_in_dur,
        "Total tests": len(take_turn_list),
        "Perfect tests (TOR=0)": take_turn_list.count(0)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()

    eval_pause_handling(args.root_dir)
