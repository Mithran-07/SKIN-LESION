"""
Dashboard and Results Analyzer.

Reads benchmark.csv (from LOQ) and generates:
- Model rankings
- Markdown tables
- JSON/CSV exports
- HTML Dashboard
"""

import os
import json
import csv
from pathlib import Path

def analyze_benchmarks(csv_path: str = "benchmark.csv", output_dir: str = "results/analysis"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"Waiting for {csv_path} from Lenovo LOQ...")
        return
        
    print(f"Analyzing {csv_path}...")
    
    # Placeholder logic since no benchmark.csv exists yet.
    # In a real scenario, pandas would read this.
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        data = list(reader)
        
    # Generate Markdown Table
    md_table = "# Benchmark Comparison\\n\\n| Model | AUC | F1 | Params (M) | FLOPs (G) |\\n|---|---|---|---|---|\\n"
    for row in data:
        md_table += f"| {row.get('model', 'Unknown')} | {row.get('auc', '0')} | {row.get('f1', '0')} | {row.get('params', '0')} | {row.get('flops', '0')} |\\n"
        
    with open(f"{output_dir}/comparison.md", "w") as f:
        f.write(md_table)
        
    # Generate HTML Dashboard
    html = f"""
    <html>
    <head><title>Experiment Dashboard</title></head>
    <body>
    <h1>Dual-Branch CNN Results</h1>
    <pre>{md_table}</pre>
    </body>
    </html>
    """
    with open(f"{output_dir}/dashboard.html", "w") as f:
        f.write(html)
        
    # Generate JSON
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"Analysis saved to {output_dir}")

if __name__ == "__main__":
    analyze_benchmarks()
