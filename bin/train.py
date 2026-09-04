import os
import logging

import glob
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

import monai
from monai.data import CacheDataset, DataLoader, list_data_collate

from monai.networks.nets import SwinUNETR
from src.swinunetr.preproceser.preprocesser import SwinUNETRPreProcesser
from src.swinunetr.trainer.trainer import SwinUNETRTrainer
from src.swinunetr.utils.logging import Logger

def main(args):

    output_path = os.path.join(args.output_dir, 
                               f'{args.project_name}_{args.k_fold}fold')
    os.makedirs(output_path, exist_ok=True)

    Logger(log_dir=os.path.join(output_path, 'log'),                 
           filename=f"{args.project_name}_{args.k_fold}fold_{args.max_epoch}epoch_{args.lr}lr.log",         
           console_level=logging.INFO,
           file_level=logging.DEBUG)
    
    trainer = SwinUNETRTrainer(dataset_dir=args.dataset_dir,
                               output_path=output_path,
                               project_name=args.project_name,
                               lm_keys=args.lm_keys,
                               patch_size=args.patch_size,
                               batch_size=args.batch_size,
                               lr=args.lr,
                               max_epoch=args.max_epoch,
                               k_fold=args.k_fold,
                               model_each_epoch_save=args.model_each_save_epoch,)

    trainer.run_kfold()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='"Script to train a SwinUNETR model \' \
                                    for heatmap-based landmark prediction. \
                                    This script handles data loading, K-fold cross-validation,  \
                                    model training, and checkpoint saving.')
    parser.add_argument('--dataset_dir', type=str,
                        default='')
    parser.add_argument('--output_dir', type=str,
                        default='./workspace')
    parser.add_argument('--project_name', type=str,
                        default='project')
    parser.add_argument('--lm_keys', nargs='+',
                        default=['head_center', 'Acetabular_outermost', 'tear_drop'])
    parser.add_argument('--patch_size', type=int,
                        default=96)
    parser.add_argument('--batch_size', type=int,
                        default=2)
    parser.add_argument('--k_fold', type=int,
                        default=4)
    parser.add_argument('--lr', type=float, 
                        default=1e-4)
    parser.add_argument('--max_epoch', type=int,
                        default=400)
    parser.add_argument('--model_each_save_epoch', type=int,
                        default=20)
    args=parser.parse_args()
    main(args)
