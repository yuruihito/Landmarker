import os
import numpy as np
import SimpleITK as sitk 
from glob import glob

from src.swinunetr.utils.fcsv import get_coords_from_fcsv_center_list
from src.swinunetr.preproceser.kfoldspliter import get_kfold_splits

class SwinUNETRPreProcesser():
    def __init__(self, 
                 dataset_base_dir, 
                 project,
                 k_fold,
                 n_keys):
        self.dataset_dir = os.path.join(dataset_base_dir, project)
        self.k_fold = k_fold
        self.n_keys = n_keys

    # ------ create path ------
    def get_img_ids(self):
        return os.listdir(os.path.join(self.dataset_dir, 'input'))
    
    def get_img_path(self, img_id: str):
        return os.path.join(self.dataset_dir, 'input', img_id, 'raw_mri_cropped.mhd')
    
    def get_lm_path(self, img_id: str):
        return os.path.join(self.dataset_dir, 'fcsv', img_id, 'landmark.fcsv')
    
    def get_output_dir(self, img_id: str):
        output_path = os.path.join(self.dataset_dir, 'preprocessed', img_id)
        os.makedirs(output_path, exist_ok=True)
        return output_path
    
    def get_output_dir_for_kfold_split(self):
        output_dir = os.path.join(self.dataset_dir, 'kfold')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir


    def get_all_path_dict(self, img_ids: str) -> dict:
        path_dict = {}
        for id in img_ids:
            path_dict[id] = {
                'img_path': self.get_img_path(id),
                'lm_path': self.get_lm_path(id),
                'output_dir': self.get_output_dir(id)
            }
        return path_dict
    
    # ------ process data ------
    @staticmethod
    def img_load(img_path: str):
        img_sitk = sitk.ReadImage(img_path)
        size = img_sitk.GetSize()
        spacing = img_sitk.GetSpacing()
        origin = img_sitk.GetOrigin()
        direction = img_sitk.GetDirection()
        img_np = sitk.GetArrayFromImage(img_sitk)
        return img_np, img_sitk, size, spacing, origin, direction

    @staticmethod
    def get_lm_dict(fcsv_path: str, n_keys: list):
        # Get x, y, z coordinates
        return {key: get_coords_from_fcsv_center_list(fcsv_path, key) for key in n_keys}
    
    @staticmethod
    def convert_physical_to_pixel_lm_list(sitk_img: sitk.Image, fcsv_path: str, n_keys: list):
        lm_list = []
        for key in n_keys:
            lm = get_coords_from_fcsv_center_list(fcsv_path, key)
            continuous_idx = sitk_img.TransformPhysicalPointToContinuousIndex(lm)
            lm_list.append([abs(idx) for idx in continuous_idx])
        return lm_list
              
    @staticmethod
    def convert_physical_to_pixel_indices(sitk_img: sitk.Image, physical_points: dict) -> dict:
        pixel_points_dict = {}
        for key, point_xyz_mm in physical_points.items():
            continuous_index = sitk_img.TransformPhysicalPointToContinuousIndex(point_xyz_mm)
            pixel_points_dict[key] = [round(abs(idx)) for idx in continuous_index]
            
        return pixel_points_dict

    def get_coordinate_points_dict(self, img_ids: list, all_path_dict: dict) -> dict:
        each_id_points_dict = {}
        for id in img_ids:
            each_id_points_dict[id] = self.get_lm_dict(all_path_dict[id]['lm_path'], self.n_keys)
        return each_id_points_dict
    
    # ------ saver ------
    @staticmethod
    def set_meta(img, spacing, origin, direction):
        if isinstance(img, np.ndarray):
            if img.ndim == 3:
                new_sitk = sitk.GetImageFromArray(img)
            elif img.ndim == 4:
                new_sitk = sitk.GetImageFromArray(img, isVector=True)
            new_sitk.SetSpacing(spacing)
            new_sitk.SetOrigin(origin)
            new_sitk.SetDirection(direction)
        return new_sitk
    
    @staticmethod
    def save_mhd_img(img, output_path):
        sitk.WriteImage(img, output_path)
    
    def save_txt_file_name(self, output_dir, filename, file_ids):
        with open(os.path.join(output_dir, f'{filename}.txt'), 'w', encoding='utf-8') as f:
            for id in file_ids:
                f.write(f'{id}\n')

    def save_kfold_split(self):
        datalist = self.get_img_ids()
        for i, (train_files, val_files, test_files) in enumerate(get_kfold_splits(datalist=datalist, 
                                                                             n_splits=self.k_fold, 
                                                                             val_cases=3)):
            output_path = os.path.join(self.get_output_dir_for_kfold_split(), f'fold{i+1}')
            os.makedirs(output_path, exist_ok=True)
            self.save_txt_file_name(output_path, 'train', train_files)
            self.save_txt_file_name(output_path, 'valid', val_files)
            self.save_txt_file_name(output_path, 'test', test_files)