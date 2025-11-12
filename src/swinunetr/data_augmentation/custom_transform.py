import torch
import monai
from monai.data import ITKReader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    SpatialPadd,
    Resized,
    RandSpatialCropd,
    RandFlipd,
    EnsureTyped,
)

def get_cache_transform():
    return Compose([
        LoadImaged(keys=["image", "label"],
                   reader=ITKReader()),    
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0), 
            mode=("trilinear", "trilinear"), 
        ),
    ])

def get_train_transform(patch_size, device):
    return Compose([
        SpatialPadd(
            keys=["image", "label"], 
            spatial_size=patch_size, 
            method='end'
        ),    
        RandSpatialCropd(
            keys=["image", "label", "lm"], 
            roi_size=patch_size, 
            random_size=False,
        ),
        RandFlipd(
            keys=["image", "label", "lm"], 
            prob=0.5, 
            spatial_axis=0,
        ),
        RandFlipd(
            keys=["image", "label", "lm"], 
            prob=0.5, 
            spatial_axis=1,
        ),
        EnsureTyped(
            keys=["image", "label"], 
            device=device, 
            track_meta=False
        ),
        EnsureTyped(
            keys=["lm"], 
            device=device,
            track_meta=False,
            dtype=torch.float32
        )
    ])


def get_val_transform(patch_size, device):
    return Compose([
        Resized(
            keys=["image", "label"],
            spatial_size=patch_size,
            mode=("trilinear", "trilinear"), 
        ),
        EnsureTyped(
            keys=["image", "label"], 
            device=device, 
            track_meta=True
        ),
        EnsureTyped(
            keys=["lm"], 
            device=device,
            track_meta=False,
            dtype=torch.float32
        )
    ])