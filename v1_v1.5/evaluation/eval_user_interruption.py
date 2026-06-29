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
   The scenario is that the user and AI are talking in the spoken conversation.
   The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.
   Your task is to rate the quality of AI's response after the user interrupt the turn.


   Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
   - 0: The AI's response is totally unrelated to the user's interrupting turn.
   - 1: The AI's response is not related to the user's interrupting turn.
   - 2: The AI's response is slightly related to the user's interrupting turn.
   - 3: The AI's response is related to the user's interrupting turn.
   - 4: The AI's response is highly related to the user's interrupting turn.
   - 5: The AI's response is perfectly related to the user's interrupting turn.


   Firstly, briefly analyze the user's interrupting turn and the AI's response
   Then, you must return the overall output as the following format:
   Analysis: [Your analysis].
   I would rate the AI's response as [Rating].
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
                - Contextual user turn: {in_before_interrupt_text}
                - User interrupting turn (occurred at [{input_start_time:.2f}-{input_end_time:.2f}]): {in_interrupt_text}
                - AI's full response (with timestamps): {ai_timestamped_text}
                
                Please look at the timestamps to determine what the AI said AFTER the user's interruption finished at {input_end_time:.2f}s, and evaluate the quality of that specific response.
                """

                prediction = generate_gemini_rating(client, system_msg, user_msg)
                
                # Tránh lỗi Rate Limit (429) của gói API Free Tier (giới hạn 5 req/phút)
                time.sleep(15)

                print(prediction)
                parsed_output = parse_output(prediction + "\n")

                print(parsed_output)
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

                if latency < 0:
                    latency_list.append(0)
                elif latency >= 0:
                    latency_list.append(latency)

            break

    print("---------------------------------------------------")
    print("[Result]")
    print("Average rating: ", sum(score_list) / len(score_list) if len(score_list) > 0 else 0.0)
    print("Average take turn: ", sum(take_turn_list) / len(take_turn_list) if len(take_turn_list) > 0 else 0.0)
    print("Average latency: ", sum(latency_list) / len(latency_list) if len(latency_list) > 0 else 0.0)
    print("---------------------------------------------------")


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
