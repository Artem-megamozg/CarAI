"""Create a concise Markdown experiment report from the latest evaluation."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/"reports"/"evaluation"/"summary_by_mode.json"
if not source.exists(): raise SystemExit("Run scripts/evaluate.py first.")
data=json.loads(source.read_text(encoding="utf-8"))
rows=[]
for mode,m in data.get("per_mode",{}).items():
    rows.append(f"| {mode} | {m['episodes']} | {m['success_rate']:.1%} | {m['collision_rate']:.1%} | {m['outside_bounds_rate']:.1%} | {m['timeout_rate']:.1%} | {m['mean_steps']:.1f} | {m['mean_final_distance_m']:.3f} | {m['mean_final_angle_error_deg']:.2f} |")
report=f"""# AutoPark RL v0.4 — evaluation report

**Model:** `{data.get('model','unknown')}`  
**Total episodes:** {data.get('episodes',0)}  
**Difficulty:** {data.get('difficulty','n/a')}

## Overall

- Success rate: **{data.get('success_rate',0):.1%}**
- Collision rate: **{data.get('collision_rate',0):.1%}**
- Outside-bounds rate: **{data.get('outside_bounds_rate',0):.1%}**
- Timeout rate: **{data.get('timeout_rate',0):.1%}**
- Mean steps: **{data.get('mean_steps',0):.1f}**

## Results by mode

| Mode | Episodes | Success | Collision | Outside | Timeout | Mean steps | Position error (m) | Angle error (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Reproducibility

Evaluation CSV: `reports/evaluation/episodes.csv`  
Summary JSON: `reports/evaluation/summary_by_mode.json`  
Chart: `reports/figures/evaluation_v04.png`

The virtual camera is synthetic and uses semantic color segmentation to estimate occupied bays. These results validate the simulator and policy in-domain; they do not establish real-world vehicle readiness.
"""
out=ROOT/"reports"/"evaluation"/"REPORT.md"; out.write_text(report,encoding="utf-8")
print(f"Report saved: {out}")
