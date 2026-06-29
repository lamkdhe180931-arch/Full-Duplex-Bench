import argparse
import os
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return None

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
        ph_res = eval_pause_handling(os.path.join(args.root_dir, "synthetic_pause_handling"), client)
        
        print("\n2. Đang chấm điểm Candor Turn Taking (Luân phiên lượt lời)...")
        stt_res = eval_smooth_turn_taking(os.path.join(args.root_dir, "candor_turn_taking"), client)
        
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

Dựa vào các chỉ số kỹ thuật và semantic trên (latency, barge-in, answer relevance, previous answer rating, new intent rating, old context leakage), hãy ĐÁNH GIÁ CHẤT LƯỢNG của Agent này bằng Tiếng Việt dưới dạng VÀI GẠCH ĐẦU DÒNG NGẮN GỌN.
Chỉ tập trung vào kết luận nhanh: 
- Phản xạ nhanh/chậm ra sao? 
- Có bị lỗi cướp lời không? 
- Câu trả lời có đúng trọng tâm không?
- Xử lý ngắt lời có trả lời tốt câu cũ, chuyển đúng câu mới, và tránh dính context cũ không?
Tuyệt đối KHÔNG cần in lại các con số, chỉ đưa ra KẾT LUẬN NGẮN GỌN.
"""
        print("Đang gửi số liệu cho Gemini để viết tóm tắt đánh giá...")
        gemini_summary = ""
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=summary_prompt
            )
            gemini_summary = response.text.strip()
            print("\n===================================================")
            print("[ BẢN TÓM TẮT ĐÁNH GIÁ (Bởi Gemini) ]")
            print("---------------------------------------------------")
            print(gemini_summary)
            print("===================================================\n")
        except Exception as e:
            gemini_summary = f"Lỗi không thể gọi Gemini: {e}"
            print(f"[ERROR] {gemini_summary}")

        # === GENERATE HTML REPORT ===
        import shutil
        from datetime import datetime
        # Fix path to ensure it appears in Kaggle/Colab output window
        if os.path.exists("/kaggle/working"):
            report_dir = "/kaggle/working/reports"
        elif os.path.exists("/content"):
            report_dir = "/content/reports"
        else:
            report_dir = os.path.join(os.getcwd(), "reports")
            
        audio_dir = os.path.join(report_dir, "audio")
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        
        def collect_audio_html(task_name, base_path):
            html_parts = []
            if not os.path.exists(base_path): return ""
            for folder in sorted(os.listdir(base_path)):
                if folder.startswith("."): continue
                combined_wav = os.path.join(base_path, folder, "combined.wav")
                if os.path.exists(combined_wav):
                    out_name = f"{task_name}_{folder}_combined.wav"
                    shutil.copy2(combined_wav, os.path.join(audio_dir, out_name))
                    html_parts.append(f"""
                    <div style="margin-top:10px;">
                        <strong>{folder}</strong><br>
                        <audio controls src="audio/{out_name}"></audio>
                    </div>
                    """)
            return "".join(html_parts)

        ph_audio_html = collect_audio_html("ph", os.path.join(args.root_dir, "synthetic_pause_handling"))
        stt_audio_html = collect_audio_html("stt", os.path.join(args.root_dir, "candor_turn_taking"))
        ui_audio_html = collect_audio_html("ui", os.path.join(args.root_dir, "synthetic_user_interruption"))

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"benchmark_v1_report_{timestamp_str}.html")

        def fmt_sec(value):
            return "N/A" if value is None else f"{value:.3f}s"

        def fmt_score(value):
            return "N/A" if value is None else f"{value:.2f} / 5.0"

        def fmt_percent(value):
            return "N/A" if value is None else f"{value:.1%}"
        
        total_tests = ph_res.get("Total tests", 0) + stt_res.get("Total tests", 0) + ui_res.get("Total tests", 0)
        good_tests = ph_res.get("Perfect tests (TOR=0)", 0) + stt_res.get("Perfect tests (smooth)", 0) + ui_res.get("Perfect tests (TOR=1 & lat>=0 & rating>=4)", 0)
        
        html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Benchmark Full-Duplex V1</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 20px; background-color: #f5f7fa; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .summary-stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .stat-box {{ flex: 1; background: #3498db; color: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-box.success {{ background: #2ecc71; }}
        .stat-box.fail {{ background: #e74c3c; }}
        .stat-number {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
        .gemini-eval {{ background: #e8f4fd; border-left: 5px solid #3498db; padding: 15px; border-radius: 4px; font-size: 1.1em; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Báo Cáo Đánh Giá Full-Duplex Agent (V1)</h1>
    <p>Thời gian tạo báo cáo: <strong>{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</strong></p>

    <div class="summary-stats">
        <div class="stat-box">
            <div>Tổng số bài test</div>
            <div class="stat-number">{total_tests}</div>
        </div>
        <div class="stat-box success">
            <div>Chạy Tốt (Đạt chuẩn)</div>
            <div class="stat-number">{good_tests}</div>
        </div>
        <div class="stat-box fail">
            <div>Chưa Tốt (Cần cải thiện)</div>
            <div class="stat-number">{total_tests - good_tests}</div>
        </div>
    </div>

    <div class="card">
        <h2>1. Đánh giá chuyên môn (Bởi Gemini)</h2>
        <div class="gemini-eval">
            {gemini_summary.replace(chr(10), '<br>')}
        </div>
    </div>

    <div class="card">
        <h2>2. Chi tiết kết quả theo từng hạng mục</h2>
        
        <h3>2.1. Pause Handling (Xử lý khoảng lặng)</h3>
        <p><strong>Số lượng test:</strong> {ph_res.get("Total tests", 0)} (Hoàn hảo: {ph_res.get("Perfect tests (listen+respond)", ph_res.get("Perfect tests (TOR=0)", 0))})</p>
        <table>
            <tr><th>Chỉ số</th><th>Giá trị</th></tr>
            <tr><td>Listen-through rate (Tiếp tục lắng nghe đến hết input)</td><td>{ph_res.get("Listen-through rate", ph_res.get("Silence rate", 0)):.1%} (Mục tiêu: Càng cao càng tốt)</td></tr>
            <tr><td>Pause barge-in rate (Nói chen trong khoảng ngập ngừng)</td><td>{ph_res.get("Pause barge-in rate", 0):.1%} (Mục tiêu: Càng thấp càng tốt)</td></tr>
            <tr><td>Continuation barge-in rate (Nói chen khi user nói tiếp)</td><td>{ph_res.get("Continuation barge-in rate", 0):.1%} (Mục tiêu: Càng thấp càng tốt)</td></tr>
            <tr><td>Answer relevance score (Trả lời đúng trọng tâm)</td><td>{fmt_score(ph_res.get("Answer relevance score"))} (Mục tiêu: >= 4.0)</td></tr>
            <tr><td>On-topic rate (Tỉ lệ đúng trọng tâm)</td><td>{fmt_percent(ph_res.get("On-topic rate"))} (Mục tiêu: Càng cao càng tốt)</td></tr>
            <tr><td>Final response success rate (Nghe hết rồi trả lời nhanh)</td><td>{ph_res.get("Final response success rate", 0):.1%} (Mục tiêu: Càng cao càng tốt)</td></tr>
            <tr><td>Avg valid latency (Độ trễ sau khi user nói xong)</td><td>{fmt_sec(ph_res.get("Avg valid latency"))} (Mục tiêu: 0-1.5s)</td></tr>
        </table>
        <h4>Bản ghi âm (Input + Output gộp)</h4>
        {ph_audio_html}

        <h3>2.2. Smooth Turn Taking (Luân phiên lượt lời)</h3>
        <p><strong>Số lượng test:</strong> {stt_res.get("Total tests", 0)} (Hoàn hảo: {stt_res.get("Perfect tests (smooth)", 0)})</p>
        <table>
            <tr><th>Chỉ số</th><th>Giá trị</th></tr>
            <tr><td>Response rate (Tỉ lệ phản hồi)</td><td>{stt_res.get("Response rate", 0):.1%} (Mục tiêu: Càng cao càng tốt)</td></tr>
            <tr><td>Post-turn response rate (Phản hồi sau khi user kết thúc)</td><td>{stt_res.get("Post-turn response rate", 0):.1%} (Mục tiêu: Càng cao càng tốt)</td></tr>
            <tr><td>Barge-in rate (Tỉ lệ cướp lời sớm)</td><td>{stt_res.get("Barge-in rate", 0):.1%} (Mục tiêu: Càng thấp càng tốt)</td></tr>
            <tr><td>Avg valid latency (Độ trễ hợp lệ)</td><td>{fmt_sec(stt_res.get("Avg valid latency"))} (Mục tiêu: 0-1.5s)</td></tr>
            <tr><td>Answer relevance score (Trả lời đúng trọng tâm)</td><td>{fmt_score(stt_res.get("Answer relevance score"))} (Mục tiêu: >= 4.0)</td></tr>
            <tr><td>On-topic rate (Tỉ lệ đúng trọng tâm)</td><td>{fmt_percent(stt_res.get("On-topic rate"))} (Mục tiêu: Càng cao càng tốt)</td></tr>
        </table>
        <h4>Bản ghi âm (Input + Output gộp)</h4>
        {stt_audio_html}

        <h3>2.3. User Interruption (Xử lý khi bị ngắt lời)</h3>
        <p><strong>Số lượng test:</strong> {ui_res.get("Total tests", 0)} (Hoàn hảo: {ui_res.get("Perfect tests (TOR=1 & lat>=0 & rating>=4)", 0)})</p>
        <table>
            <tr><th>Chỉ số</th><th>Giá trị</th></tr>
            <tr><td>Previous answer rating (Trả lời câu hỏi trước)</td><td>{ui_res.get("Previous answer rating", ui_res.get("Context rating", 0)):.2f} / 5.0 (Mục tiêu: >= 4.0)</td></tr>
            <tr><td>New intent rating (Điểm hiểu ý mới)</td><td>{ui_res.get("New intent rating", 0):.2f} / 5.0 (Mục tiêu: >= 4.0)</td></tr>
            <tr><td>Old context leakage score (Dính context cũ sau interrupt)</td><td>{ui_res.get("Old context leakage score", 0):.2f} / 5.0 (Mục tiêu: <= 1.0)</td></tr>
            <tr><td>Overall semantic rating (Tổng semantic)</td><td>{ui_res.get("Overall semantic rating", 0):.2f} / 5.0 (Mục tiêu: >= 4.0)</td></tr>
            <tr><td>Semantic success rate (Đúng cả câu cũ/câu mới)</td><td>{ui_res.get("Semantic success rate", 0):.1%} (Mục tiêu: Càng cao càng tốt)</td></tr>
            <tr><td>Stop success rate (Ngừng nói nhanh khi bị ngắt)</td><td>{ui_res.get("Stop success rate", 0):.1%} (Mục tiêu: Càng cao càng tốt)</td></tr>
            <tr><td>Avg stop latency (Độ trễ ngừng nói)</td><td>{fmt_sec(ui_res.get("Avg stop latency"))} (Mục tiêu: <= 0.7s)</td></tr>
            <tr><td>Listening success rate (Im lặng để nghe câu ngắt lời)</td><td>{ui_res.get("Listening success rate", 0):.1%} (Mục tiêu: Càng cao càng tốt)</td></tr>
            <tr><td>Post-interrupt response rate (Tỉ lệ phản hồi sau interrupt)</td><td>{ui_res.get("Response rate", 0):.1%} (Mục tiêu: Càng cao càng tốt)</td></tr>
            <tr><td>Avg recovery latency (Độ trễ phản hồi sau interrupt)</td><td>{fmt_sec(ui_res.get("Avg recovery latency", ui_res.get("Avg valid latency")))} (Mục tiêu: 0-2.0s)</td></tr>
            <tr><td>Interrupt overlap rate (Tỉ lệ còn nói đè lên câu ngắt)</td><td>{ui_res.get("Interruption overlap rate", ui_res.get("Barge-in rate", 0)):.1%} (Mục tiêu: Càng thấp càng tốt)</td></tr>
            <tr><td>Perfect handling rate (Tỉ lệ xử lý hoàn hảo)</td><td>{ui_res.get("Perfect handling rate", 0):.1%} (Mục tiêu: Càng cao càng tốt)</td></tr>
        </table>
        <h4>Bản ghi âm (Input + Output gộp)</h4>
        {ui_audio_html}
    </div>

</body>
</html>"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"✅ Đã xuất báo cáo HTML thành công tại: {report_path}")


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
