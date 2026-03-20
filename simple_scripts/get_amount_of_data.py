import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.transforms import LoadImage
import pandas as pd
from pathlib import Path

#####Simple script used to analyze how many of each sample i got and how they are distributed across hospitals###


results = []


df = pd.read_csv('../Dataset/ODELIA2025/data/UKA/metadata_unilateral/split.csv')

# Count splits
counts = df["Split"].value_counts()

# Extract counts safely (in case some splits are missing)
train = counts.get("train", 0)
val = counts.get("val", 0)
test = counts.get("test", 0)

results.append({
    "Hospital": "UKA",  # filename used as hospital name
    "train": train,
    "validation": val,
    "test": test,
    "total": train + val + test
})

# Convert to table
summary_df = pd.DataFrame(results)

print(summary_df)