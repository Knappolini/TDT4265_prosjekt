import os
from pathlib import Path
import pandas as pd

#Splits the given dataset into training, validation and test sets where all of the samples have 5 images omitting the post3/4/5 in order to keep the channels of the tensors consistent for the network.
#Only the paths
#RSH is left out because i have no labels and the final submission is my prediction on these images. 

data_dir = 'C:/Users/joh-k/OneDrive/Dokumenter/Fag_vaaren_26/TDT4265/Dataset/ODELIA2025/data'

def build_datasets(root_dir):
    root = Path(root_dir)

    train_data = []
    val_data = []
    test_data = []

    for hospital_dir in root.iterdir():
        if not hospital_dir.is_dir():
            continue

        print(f"\nProcessing {hospital_dir.name}")

        split_path = hospital_dir / "metadata_unilateral" / "split.csv"
        anno_path = hospital_dir / "metadata_unilateral" / "annotation.csv"

        df_split = pd.read_csv(split_path)

        has_labels = anno_path.exists()

        if has_labels:
            df_anno = pd.read_csv(anno_path)
            df = df_split.merge(df_anno, on="UID")
        else:
            print(f"No labels for {hospital_dir.name} (Hidden test set)")
            continue


        for _, row in df.iterrows():
            uid = row["UID"]
            split = row["Split"]

            label = None
            if has_labels:
                label = int(row["Lesion"])

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

                selected_files = []

                for f in nii_files_all:
                    name = f.name.lower()

                    if not any(x in name for x in ["post_3", "post_4", "post_5"]):
                        selected_files.append(f)

                nii_files = sorted(selected_files)

                if len(nii_files) == 0:
                    continue

                sample = {
                    "image": [str(f) for f in nii_files],
                    "breast": breast,
                    "uid": uid,
                    "hospital": hospital_dir.name
                }

                if has_labels:
                    sample["label"] = label

                split_lower = split.lower()

                if has_labels:
                    if split_lower == "train":
                        train_data.append(sample)
                    elif split_lower in ["val", "validation"]:
                        val_data.append(sample)
                    elif split_lower == "test":
                        test_data.append(sample)
                else:
                    test_data.append(sample)

    print("\nFinal counts:")
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")
    print(f"Test: {len(test_data)}")

    return train_data, val_data, test_data

