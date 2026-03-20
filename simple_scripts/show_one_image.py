import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.transforms import LoadImage

loader = LoadImage(image_only=True)

PATH = "../Dataset/ODELIA2025/data/CAM/data_unilateral/ODELIA_BRAID1_0158_1_left/Post_1.nii.gz"

image = loader(PATH)

print(image.shape)
print(image.dtype)
print(image[128,128,1].item())


# print("Min:", image.min().item())
# print("Max:", image.max().item())
# print("Mean:", image.mean().item())

# plt.hist(image.flatten(), bins=100)
# plt.show()

for i in range(image.shape[2]):
    plt.imshow(image[:, :, i], cmap="gray")
    plt.title(f"Slice {i}")
    plt.axis("off")
    plt.pause(0.1)




