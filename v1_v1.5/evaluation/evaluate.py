import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation tasks.")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "backchannel",
            "pause_handling",
            "smooth_turn_taking",
            "user_interruption",
            "behavior",
            "general_before_after",
            "v1_all_with_summary",
        ],
        help="Evaluation task to perform.",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing data for evaluation.",
    )

    args = parser.parse_args()

    if args.task == "backchannel":
        from eval_backchannel import eval_backchannel

        eval_backchannel(args.root_dir)
    elif args.task == "pause_handling":
        from eval_pause_handling import eval_pause_handling

        eval_pause_handling(args.root_dir)
    elif args.task == "smooth_turn_taking":
        from eval_smooth_turn_taking import eval_smooth_turn_taking

        eval_smooth_turn_taking(args.root_dir)
    elif args.task == "user_interruption":
        from eval_user_interruption import eval_user_interruption
        from google import genai

        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        client = genai.Client(api_key=gemini_api_key)
        eval_user_interruption(args.root_dir, client)

    elif args.task == "v1_all_with_summary":
        from eval_pause_handling import eval_pause_handling
        from eval_smooth_turn_taking import eval_smooth_turn_taking
        from eval_user_interruption import eval_user_interruption
        from google import genai
        import json

        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        client = genai.Client(api_key=gemini_api_key)

        print("\n=== ĐANG CHẠY TOÀN BỘ BENCHMARK V1 ===")
        print("1. Đang chấm điểm Pause Handling (Xử lý khoảng lặng)...")
        ph_res = eval_pause_handling(os.path.join(args.root_dir, "synthetic_pause_handling"))
        
        print("\n2. Đang chấm điểm Candor Turn Taking (Luân phiên lượt lời)...")
        stt_res = eval_smooth_turn_taking(os.path.join(args.root_dir, "candor_turn_taking"))
        
        print("\n3. Đang chấm điểm User Interruption (Xử lý khi bị ngắt lời)...")
        ui_res = eval_user_interruption(os.path.join(args.root_dir, "synthetic_user_interruption"), client)

        print("\n=== ĐANG TỔNG HỢP KẾT QUẢ ===")
        summary_prompt = f"""
Dưới đây là điểm benchmark âm thanh song công (Full-Duplex V1) của agent.
1. Pause Handling (Xử lý khoảng lặng):
{json.dumps(ph_res, indent=2)}

2. Smooth Turn Taking (Luân phiên lượt lời mượt mà):
{json.dumps(stt_res, indent=2)}

3. User Interruption (Xử lý khi bị ngắt lời):
{json.dumps(ui_res, indent=2)}

Dựa vào các chỉ số kỹ thuật trên (như độ trễ latency âm hay dương, take turn rate cao hay thấp, rating chất lượng xử lý ngắt lời ra sao), hãy ĐÁNH GIÁ CHẤT LƯỢNG của Agent này bằng Tiếng Việt dưới dạng VÀI GẠCH ĐẦU DÒNG NGẮN GỌN.
Chỉ tập trung vào kết luận nhanh: 
- Phản xạ nhanh/chậm ra sao? 
- Có bị lỗi cướp lời không? 
- Xử lý ngắt lời có mượt và đúng ngữ cảnh không?
Tuyệt đối KHÔNG cần in lại các con số, chỉ đưa ra KẾT LUẬN NGẮN GỌN.
"""
        print("Đang gửi số liệu cho Gemini để viết tóm tắt đánh giá...")
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=summary_prompt
            )
            print("\n===================================================")
            print("[ BẢN TÓM TẮT ĐÁNH GIÁ (Bởi Gemini) ]")
            print("---------------------------------------------------")
            print(response.text.strip())
            print("===================================================\n")
        except Exception as e:
            print(f"[ERROR] Không thể lấy tóm tắt từ Gemini: {e}")


    elif args.task == "general_before_after":
        from eval_general_before_after import eval_general_all_split

        config = {
            "trim_silence": True,
            "squim": False,
            "utmosv2": True,
            "speaking_rate": True,
            "trim_mode": "silero",
            "agg": {
                "mode": "trim",
                "trim_prop": 0.05,
            },
            "pitch": True,
            "intensity": True,
        }

        aggregate = True
        output = eval_general_all_split(config, args.root_dir, aggregate=aggregate)
        print("General Evaluation Output:", output)
        if aggregate:
            # save to log file
            dir_name = args.root_dir.split("/")[-1]
            log_path = f"{dir_name}_general.log"
            with open(log_path, "w", encoding="utf-8") as f:
                for key, value in output.items():
                    line = f"{key}: {value}\n"
                    print(line, end="")
                    f.write(line)

    elif args.task == "behavior":
        from openai import OpenAI
        from eval_behavior import eval_behavior_all

        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            # organization=organization,
            api_key=api_key,
        )
        output = eval_behavior_all(
            args.root_dir, client, task=args.task, aggregate=True
        )

        dir_name = args.root_dir.split("/")[-1]
        log_path = f"{dir_name}_{args.task}.log"

        with open(log_path, "w", encoding="utf-8") as f:
            for ax in ["C"]:
                line = f"Ratios ({ax}-axis): {output[ax]}\n"
                print(line, end="")
                f.write(line)


if __name__ == "__main__":
    main()
