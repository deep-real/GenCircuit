import glob
import os
import pandas as pd

# Load DataFrame
paths = [
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/metashift_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/metashift-control_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/PACS_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/PACS-photo_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/PACS-cartoon_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/PACS-art_painting_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/PACS-set2_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/camelyon17_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/terra-incognita-38_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/terra-incognita-43_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/terra-incognita-46_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/terra-incognita-100_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/iwildcam_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/cifar10_sweep_results.csv",
    "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/fmow-set2_sweep_results.csv",

]
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

keep_dir_csvs = [
    # "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/metashift_sweep_dynamic_results.csv",
    # "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/PACS_sweep_dynamic_results.csv",
    # "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/terra-incognita-38_sweep_dynamic_results.csv",

]

# Set of valid checkpoint filenames
valid_checkpoints = set(df["checkpoint"].tolist())
for csv_path in keep_dir_csvs:
    df = pd.read_csv(csv_path)
    if "checkpoint" not in df.columns:
        raise ValueError(f"'checkpoint' column not found in {csv_path}")

    ckpts = df["checkpoint"].dropna().tolist()
    for ckpt in ckpts:
        dirname = os.path.dirname(ckpt)
        # Add *all* files under the same directory
        for f in glob.glob(os.path.join(dirname, "*")):
            valid_checkpoints.add(f)

# Root directory containing checkpoints in subdirectories
output_root = "output"

ab_path = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/'

# File extensions to check (modify if needed)
valid_extensions = [".bin"]

# Walk through all subdirectories
for dirpath, _, filenames in os.walk(output_root):
    for fname in filenames:
        if any(fname.endswith(ext) for ext in valid_extensions):
            this_path = os.path.join(os.path.relpath(dirpath, ab_path), fname)
            if this_path not in valid_checkpoints:
                # if 'fmow' in this_path:
                full_path = os.path.join(ab_path, this_path)
                os.remove(full_path)
                print(f"Deleted: {full_path}")
