import nbformat

path = "/Users/lam/Library/CloudStorage/GoogleDrive-he180931@gmail.com/Drive của tôi/AI thực chiến/tts/Full-Duplex-Bench/v1_v1.5/data_generation/benchmark_v1_workflow.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        if 'def show_gemini_outputs_now' in cell.source:
            # show_gemini_outputs_now
            if 'combined_wav = sample_dir' not in cell.source:
                cell.source = cell.source.replace(
                    'output_wav = sample_dir / "output.wav"', 
                    'output_wav = sample_dir / "output.wav"\n            combined_wav = sample_dir / "combined.wav"'
                )
                cell.source = cell.source.replace(
                    'rows.append({"task": task_name, "sample_id": sample_dir.name, "output_wav": output_wav})',
                    'rows.append({"task": task_name, "sample_id": sample_dir.name, "output_wav": output_wav, "combined_wav": combined_wav if combined_wav.exists() else None})'
                )
                cell.source = cell.source.replace(
                    'display(Audio(str(row["output_wav"])))', 
                    'display(Audio(str(row["output_wav"])))\n            if row.get("combined_wav"):\n                display(Markdown(f"**{row[\'task\']}/{row[\'sample_id\']}/combined.wav** (Full-Duplex)"))\n                display(Audio(str(row["combined_wav"])))'
                )

            # status_df
            if '"combined.wav"' not in cell.source:
                cell.source = cell.source.replace(
                    '"output.wav": (sample_dir / "output.wav").exists(),', 
                    '"output.wav": (sample_dir / "output.wav").exists(),\n                "combined.wav": (sample_dir / "combined.wav").exists(),'
                )

            # ready_outputs
            if 'combined_wav.exists()' not in cell.source:
                cell.source = cell.source.replace(
                    'display(Audio(str(sample_dir / "output.wav")))', 
                    'display(Audio(str(sample_dir / "output.wav")))\n        combined_wav = sample_dir / "combined.wav"\n        if combined_wav.exists():\n            display(Markdown(f"**{getattr(row, \'task\')}/{getattr(row, \'sample_id\')}/combined.wav** (Full-Duplex)"))\n            display(Audio(str(combined_wav)))'
                )

with open(path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Updated notebook to display combined.wav")
