import os
import numpy as np
import SimpleITK as sitk
from src.swinunetr.preproceser.preprocesser import SwinUNETRPreProcesser

class BuilderDataList:
    def __init__(self, 
                 dataset_dir,
                 project,
                 k_fold,
                 lm_keys,
                 ):
        self.dataset_base_dir = os.path.join(dataset_dir, f'{project}_{k_fold}fold')
        self.lm_keys = lm_keys
        self.pre = SwinUNETRPreProcesser(dataset_dir, project, k_fold, lm_keys)

    @staticmethod
    def load_ids_from_txt(txt_file_path: str) -> list:
        
        with open(txt_file_path, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        return ids

    def get_file_list(self, ids: list) -> list:

        files_list = []
        for id in ids:
            id_dir = os.path.join(self.dataset_base_dir, 'preprocessed', id)
            coord_dir = os.path.join(self.dataset_base_dir, 'fcsv', id)
            
            img_path = os.path.join(id_dir, 'raw.mhd')
            hm_path = os.path.join(id_dir, 'label.mhd')
            lm_path = os.path.join(coord_dir, 'landmark.fcsv')

            files_list.append({"image": img_path, "label": hm_path, 
                               'lm': np.array(self.pre.convert_physical_to_pixel_lm_list(sitk.ReadImage(img_path), lm_path, self.lm_keys))})

        return files_list
    
    def get_k_fold_txt_path(self, fold: int, target: str):
        return os.path.join(self.dataset_base_dir, 'kfold', f'fold{fold}', f'{target}.txt')
    
    def get_kfold_file_list_from_txt_path(self, fold, target_name):
        target_id_file_path = self.get_k_fold_txt_path(fold=fold, target=target_name)
        target_ids = self.load_ids_from_txt(target_id_file_path)
        target_files = self.get_file_list(target_ids)
        return target_files