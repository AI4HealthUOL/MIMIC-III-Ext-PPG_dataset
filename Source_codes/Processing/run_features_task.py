import os
import gc
import pandas as pd

from utils import round_all_numeric
from main import process_chunk_of_30s_segments  

# ----------------------------
# Settings
# ----------------------------
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
fs = 125
plot_flag = False

base_root = "/path/to/your/base/directory"  # CHANGE THIS to your base directory containing the WFDB files and where you want to save results

wfdb_root = os.path.join(
    base_root,
    "WFDB_files_with_utils_5_15min_30s_segmented_no_Flat_PPG_new_4"
)

# Per-task metadata produced in Phase 1
meta_data_path = os.path.join(
    wfdb_root,
    f"WFDB_files_segmented_task_{task_id}.pkl"
)

# Per-task WFDB folder produced in Phase 1
wfdb_folder = os.path.join(wfdb_root, f"task_{task_id}")

# Output folder for Phase 2 results
out_root = os.path.join(base_root, "NPY_files_with_utils_5_15min_30s_segmented_no_Flat_PPG_new_4")
os.makedirs(out_root, exist_ok=True)

print("===================================")
print(f"🚀 Feature extraction | Task {task_id}")
print(f"Metadata: {meta_data_path}")
print(f"WFDB folder: {wfdb_folder}")
print("===================================")

if not os.path.exists(meta_data_path):
    print(f"❌ Missing metadata file: {meta_data_path}")
    raise SystemExit(0)

if not os.path.isdir(wfdb_folder):
    print(f"❌ Missing WFDB folder: {wfdb_folder}")
    raise SystemExit(0)

# ----------------------------
# Run
# ----------------------------
npy_output_folder = os.path.join(out_root, f"PPG_NPY_task_{task_id}")


results_df = process_chunk_of_30s_segments(
    meta_data_path=meta_data_path,
    fs=fs,
    start_index=0,
    end_index=None,
    plot_flag=plot_flag,
    wfdb_folder=wfdb_folder,
    save_ppg_segments=True,  #  TURN ON for savign the .npy files
    ppg_output_folder=npy_output_folder
)

# ----------------------------
# Save
# ----------------------------
if results_df.empty:
    print("⚠️ results_df is empty (no valid segments).")
else:
    results_df = round_all_numeric(results_df, decimals=2)
    out_path = os.path.join(out_root, f"features_task_{task_id}.pkl")
    results_df.to_pickle(out_path)
    print(f"✅ Saved: {out_path}")
    print("Rows:", len(results_df))

del results_df
gc.collect()
print(f"✅ Task {task_id} finished.")
