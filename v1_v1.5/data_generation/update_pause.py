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
        
        source = "".join(source_lines)
        
        # Check if this is the pause handling cell
        if 'pause_duration = 1.5' in source and 'synthetic_pause_handling' in source:
            # Remove the global pause_duration = 1.5
            source = source.replace("pause_rows = []\npause_duration = 1.5\nfor item in load_template", "pause_rows = []\nfor item in load_template")
            
            # Read from JSON
            source = source.replace(
                '    sample_id = item["id"]', 
                '    pause_duration = item.get("pause_duration_sec", 1.5)\n    sample_id = item["id"]'
            )
            
            # Update the markdown
            source = source.replace(
                'display(Markdown("**Step 2 - pause**: 1.5s silence, xem mốc trong bảng timeline."))',
                'display(Markdown(f"**Step 2 - pause**: {pause_duration}s silence, xem mốc trong bảng timeline."))'
            )
            changed = True
            
        if changed:
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

tmp_path = path + ".tmp2"
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

os.replace(tmp_path, path)
print("Fixed pause_duration in notebook!")
