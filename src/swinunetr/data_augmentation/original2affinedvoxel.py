import numpy as np
import torch
from monai.transforms import MapTransform

from monai.config import KeysCollection
from nibabel.affines import apply_affine
from src.swinunetr.utils.affines import apply_inverse_affine

class UpdateVoxelCoordsd(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 image_key: str = "image",
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.image_key = image_key

    def __call__(self, data):
        d = dict(data)
        meta_key = f"{self.image_key}_meta_dict"
        
        if "original_affine" not in d[meta_key]:
            raise KeyError(f"'original_affine' not in {meta_key}. Ensure LoadImaged ran.")
        if "affine" not in d[meta_key]:
            raise KeyError(f"'affine' not in {meta_key}. Ensure spatial transform ran.")
            
        original_affine = d[meta_key]["original_affine"]
        new_affine = d[meta_key]["affine"]
        
        for key in self.key_iterator(d):
            # 1. (z,y,x) -> (x,y,z) 
            original_voxel_zyx = d[key]
            if isinstance(original_voxel_zyx, torch.Tensor):
                original_voxel_zyx = original_voxel_zyx.numpy()
            original_voxel_xyz = original_voxel_zyx[:, [2, 1, 0]]
            
            # 2. オリジナルピクセル -> 物理座標
            phys_coords_xyz = apply_affine(original_affine, original_voxel_xyz)
            
            # 3. 物理座標 -> 新ピクセル座標 (Spacingd後)
            new_voxel_xyz = apply_inverse_affine(new_affine, phys_coords_xyz)
            
            # 4. (x,y,z) -> (z,y,x) に
            new_voxel_zyx = new_voxel_xyz[:, [2, 1, 0]]
            d[key] = new_voxel_zyx.astype(np.float32)
            
        return d