import torch
from monai.data import Dataset , DataLoader
from Preprocessing.build_datasets import build_datasets
from Preprocessing.build_datasets_5channels import build_datasets_5channels
from torch.utils.data import Subset

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated
)

#Laptop
# data_dir = 'C:/Users/joh-k/OneDrive/Dokumenter/Fag_vaaren_26/TDT4265/Dataset/ODELIA2025/data'

#Cybele
data_dir = '/datasets/tdt4265/ODELIA2025/data'

#IDUN HPC
#data_dir =  '/cluster/projects/vc/courses/TDT17/mic/ODELIA2025'


root_dir = data_dir

# train_data, val_data, test_data = build_datasets(root_dir)

train_data, val_data, test_data = build_datasets_5channels(root_dir)

# Load the images
# Do the transforms
# Convert them into torch tensors
# Loosely inspired by https://www.youtube.com/watch?v=83FLt4fPNGs , https://www.youtube.com/watch?v=hqgZuatm8eE



#Here i will load and augment the data used for training in order to get the most out of the limited data we have available for training
train_transforms = Compose(
    [
        LoadImaged(keys = ['image']), #Load the image by reading all .nii.gz files and converting them to arrays
        EnsureChannelFirstd(keys = ['image']), #Guarantees [C, H, W, D]

        #Normalization
        ScaleIntensityd(keys=['image']),#In order to normalize the data to the range [0, 1] for better performance of the neural network. Because MRI intensities vary across hospitals
        Resized(keys=['image'], spatial_size = (128,128,32)), #Resize in order to deal with variable shapes and restrict the amount of memory needed

        #Augmentation
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5), #Multiplies intensity by random factor to simulate different scanners
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5), #Adds random offset to simulate brightness differences
        RandBiasFieldd(keys=["image"], prob=0.3), #applies smooth intensity distortion to simulate MRI coil artifacts ???
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.01), #Adds random white noise to simulate lower-quality scanners
        RandGaussianSmoothd(keys=["image"], prob=0.2), #Blurs the image to simulate different scanner resolutions
        RandRotated(keys=["image"],range_x=0.1,range_y=0.1,range_z=0.1,prob=0.3), #Small rotation to simulate patient positioning differences

        EnsureTyped(keys =  ['image', 'label']) #Last function to apply, we turn it into tensor after having applied all the transforms
    ]
)

#Load the data, no augmentation as this is validation set
val_transforms = Compose(
    [
        LoadImaged(keys = ['image']),
        EnsureChannelFirstd(keys = ['image']), 

        ScaleIntensityd(keys=['image']),
        Resized(keys=['image'], spatial_size = (128,128,32)),

        EnsureTyped(keys =  ['image', 'label'])
    ]
)

#Load the data, no augmentation as this is the test set
test_transforms = Compose(
    [
        LoadImaged(keys = ['image']),
        EnsureChannelFirstd(keys = ['image']), 

        ScaleIntensityd(keys=['image']),
        Resized(keys=['image'], spatial_size = (128,128,32)),

        EnsureTyped(keys =  ['image', 'label']) 
    ]
)

train_ds = Dataset(data = train_data, transform = train_transforms)
train_loader = DataLoader(train_ds, batch_size = 2, shuffle = True, num_workers=4, pin_memory = True )

small_train_ds =Subset(train_loader.dataset, list(range(8)))
small_train_loader = DataLoader(small_train_ds, batch_size=2,shuffle = True )

val_ds = Dataset(data = val_data ,transform = val_transforms )
val_loader = DataLoader(val_ds, batch_size = 2, shuffle = False, num_workers= 4, pin_memory = True)

small_val_ds = Subset(val_loader.dataset, list(range(8)))
small_val_loader = DataLoader(small_val_ds, batch_size=2)

