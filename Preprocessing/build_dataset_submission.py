import os
from pathlib import Path
import pandas as pd


def build_dataset_submission(root_dir):
    root = Path(root_dir)

    test_data = []

    for hospital_dir in root.iterdir():
        if not hospital_dir.is_dir():
            continue

        # ✅ ONLY RSH
        if hospital_dir.name != "RSH":
            continue

        print(f"\nProcessing TEST hospital: {hospital_dir.name}")

        split_path = hospital_dir / "metadata_unilateral" / "split.csv"
        df_split = pd.read_csv(split_path)

        for _, row in df_split.iterrows():
            uid = row["UID"]

            data_root = hospital_dir / "data_unilateral"
            breast_folders = [
                f for f in data_root.iterdir()
                if f.name.startswith(uid)
            ]

            for bf in breast_folders:
                name = bf.name.lower()

                if "left" in name:
                    breast = "left"
                elif "right" in name:
                    breast = "right"
                else:
                    continue

                nii_files_all = sorted(list(bf.glob("*.nii.gz")))

                selected_files = [
                    f for f in nii_files_all
                    if not any(x in f.name.lower() for x in ["post_3", "post_4", "post_5"])
                ]

                if len(selected_files) == 0:
                    continue

                sample = {
                    "image": [str(f) for f in selected_files],
                    "uid": f"{uid}_{breast}"  # ✅ REQUIRED FOR CSV
                }

                test_data.append(sample)

    print(f"\nRSH Test samples: {len(test_data)}")

    return test_data