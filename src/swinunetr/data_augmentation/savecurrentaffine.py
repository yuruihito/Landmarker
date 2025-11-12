import numpy as np
import torch
from monai.transforms import MapTransform

from monai.config import KeysCollection
from nibabel.affines import apply_affine
from src.swinunetr.utils.affines import apply_inverse_affine


class SaveCurrentAffined(MapTransform):
    def __init__(self, 
                 image_key: str, 
                 new_key: str = "previous_affine") -> None:
        
        super().__init__(keys=[])
        self.image_key = image_key
        self.new_key = new_key

    def __call__(self, data):
        d = dict(data)
        meta_key = f"{self.image_key}_meta_dict"
        
        if "affine" not in d[meta_key]:
            raise KeyError(f"'affine' not in {meta_key}. Ensure spatial transform ran.")
        
        # Spacingd後のアフィンを 'previous_affine' としてバックアップ
        d[meta_key][self.new_key] = d[meta_key]["affine"]
        
        return d

class UpdateVoxelCoordsFromPreviousd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        image_key: str = "image",
        previous_affine_key: str = "previous_affine",
        allow_missing_keys: bool = False) -> None:

        super().__init__(keys, allow_missing_keys)
        self.image_key = image_key
        self.previous_affine_key = previous_affine_key

    def __call__(self, data):
        d = dict(data)
        meta_key = f"{self.image_key}_meta_dict"
        
        if self.previous_affine_key not in d[meta_key]:
            raise KeyError(f"'{self.previous_affine_key}' not in {meta_key}. Ensure SaveCurrentAffined ran.")
        if "affine" not in d[meta_key]:
            raise KeyError(f"'affine' not in {meta_key}. Ensure spatial transform (e.g., Resized) ran.")
            
        previous_affine = d[meta_key][self.previous_affine_key] # (Spacingd後のアフィン)
        new_affine = d[meta_key]["affine"]                     # (Resized後のアフィン)
        
        for key in self.key_iterator(d):
            # 1. (z,y,x) -> (x,y,z) に
            prev_voxel_zyx = d[key]
            if isinstance(prev_voxel_zyx, torch.Tensor):
                prev_voxel_zyx = prev_voxel_zyx.cpu().numpy()
            prev_voxel_xyz = prev_voxel_zyx[:, [2, 1, 0]]
            
            # 2. Spacingd後のピクセル -> 物理座標
            phys_coords_xyz = apply_affine(previous_affine, prev_voxel_xyz)
            
            # 3. 物理座標 -> Resized後のピクセル座標
            new_voxel_xyz = apply_inverse_affine(new_affine, phys_coords_xyz)
            
            # 4. (x,y,z) -> (z,y,x) に
            new_voxel_zyx = new_voxel_xyz[:, [2, 1, 0]]
            d[key] = new_voxel_zyx.astype(np.float32)
            
        return d