"""Run the whole pipeline in order.  Steps 1-3 hit the Wayback Machine (about 2 hours,
resumable); steps 4-13 are offline and take about an hour.

    py run_all.py            # everything
    py run_all.py 4          # from step 4 onward (re-analysis without re-downloading)
"""
import subprocess
import sys
import os

STEPS = ["01_harvest_index", "02_build_sample", "03_download", "04_extract_clean", "05_features",
         "06_markers", "07_style_divergence", "08_distinctive_words", "09_data_quality",
         "10_who_moved", "11_sources", "12_topic_control", "13_summary"]

start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(here, "logs"), exist_ok=True)
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
for i, step in enumerate(STEPS, 1):
    if i < start:
        continue
    print(f"=== step {i}: {step}", flush=True)
    with open(os.path.join(here, "logs", f"{step}.log"), "w", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, "-W", "ignore", os.path.join(here, "src", f"{step}.py")],
                           stdout=log, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        sys.exit(f"step {step} failed — see logs/{step}.log")
print("done")
