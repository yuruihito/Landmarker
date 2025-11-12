

import logging
from tqdm import tqdm
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
)
logger = logging.getLogger("project.loading")

class CacheDataset(Dataset):
    def __init__(self, 
                 data_dicts: list, 
                 pre_cache_transform: Compose, 
                 runtime_transform: Compose):
        '''
        args 
        data_dict(list) [{'image': './path', label: './path'}]
        pre_cash_dataset: pre_cache_transform (Compose): Transforms applied only once during cache creation (e.g., LoadImageD)
        runtime_transform (Compose): Transforms applied every time in __getitem__ (e.g., RandSpatialCropSamplesD)
        '''
        self.data_dicts = data_dicts
        self.pre_cache_transform = pre_cache_transform
        self.runtime_transform = runtime_transform
        self.cache = []

        logger.info('starting cache data into RAM ...')

        for data_dict in tqdm(self.data_dicts, desc='Cashing Data'):
            preprocessed_data = self.pre_cache_transform(data_dict)
            self.cache.append(preprocessed_data)
        
    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        cache_data = self.cache[idx]
        runtime_data = self.runtime_transform(cache_data)
        return runtime_data