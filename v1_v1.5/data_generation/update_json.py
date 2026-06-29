import json
import os

path = "/Users/lam/Library/CloudStorage/GoogleDrive-he180931@gmail.com/Drive của tôi/AI thực chiến/tts/Full-Duplex-Bench/v1_v1.5/data_generation/benchmark_v1_workflow.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changed = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        if not source_lines:
            continue
        
        # Combine to single string for easy replacement
        source = "".join(source_lines)
        
        if 'def show_gemini_outputs_now' in source:
            if 'combined_wav = sample_dir' not in source:
                source = source.replace(
                    'output_wav = sample_dir / "output.wav"', 
                    'output_wav = sample_dir / "output.wav"\n            combined_wav = sample_dir / "combined.wav"'
                )
                source = source.replace(
                    'rows.append({"task": task_name, "sample_id": sample_dir.name, "output_wav": output_wav})',
                    'rows.append({"task": task_name, "sample_id": sample_dir.name, "output_wav": output_wav, "combined_wav": combined_wav if combined_wav.exists() else None})'
                )
                source = source.replace(
                    'display(Audio(str(row["output_wav"])))', 
                    'display(Audio(str(row["output_wav"])))\n            if row.get("combined_wav"):\n                display(Markdown(f"**{row[\'task\']}/{row[\'sample_id\']}/combined.wav** (Full-Duplex)"))\n                display(Audio(str(row["combined_wav"])))'
                )
                changed = True

            # status_df
            if '"combined.wav"' not in source:
                source = source.replace(
                    '"output.wav": (sample_dir / "output.wav").exists(),', 
                    '"output.wav": (sample_dir / "output.wav").exists(),\n                "combined.wav": (sample_dir / "combined.wav").exists(),'
                )
                changed = True

            # ready_outputs
            if 'combined_wav.exists()' not in source:
                source = source.replace(
                    'display(Audio(str(sample_dir / "output.wav")))', 
                    'display(Audio(str(sample_dir / "output.wav")))\n        combined_wav = sample_dir / "combined.wav"\n        if combined_wav.exists():\n            display(Markdown(f"**{getattr(row, \'task\')}/{getattr(row, \'sample_id\')}/combined.wav** (Full-Duplex)"))\n            display(Audio(str(combined_wav)))'
                )
                changed = True
            
        if changed:
            # Reconstruct list of strings
            new_lines = []
            parts = source.split('\n')
            for i, p in enumerate(parts):
                if i < len(parts) - 1:
                    new_lines.append(p + '\n')
                else:
                    if p:
                        new_lines.append(p)
            cell["source"] = new_lines
            changed = False

tmp_path = path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

os.replace(tmp_path, path)
print("Updated notebook successfully!")
