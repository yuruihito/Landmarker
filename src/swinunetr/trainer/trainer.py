import os
import torch
import logging
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.data import list_data_collate
from monai.transforms.utils import apply_affine_to_points
from monai.networks.nets import SwinUNETR
import inspect
print(f"DEBUG: Importing SwinUNETR from: {inspect.getfile(SwinUNETR)}")

from src.swinunetr.loss.loss import GLiPLoss
from torch.optim import Adam
from src.swinunetr.dataloading.build_data_list import BuilderDataList
from src.swinunetr.dataloading.cachedataset import CacheDataset
from src.swinunetr.data_augmentation.custom_transform import (
    get_cache_transform,
    get_train_transform,
    get_val_transform
)
from src.swinunetr.utils.visualizer import ( 
    _create_hotspot_overlay_image,
    _save_concatenated_overlays
)
from src.swinunetr.postprocesser.getcoordsfromheatmap import SoftArgMax

logger = logging.getLogger("project.train")

class SwinUNETRTrainer:
    def __init__(self, 
                 dataset_dir,
                 output_path,
                 project_name,
                 lm_keys,
                 patch_size,
                 batch_size,
                 lr,
                 max_epoch,
                 k_fold,
                 model_each_epoch_save):
        
        self.dataset_dir = dataset_dir
        self.output_path = output_path
        self.project = project_name
        self.lm_keys = lm_keys
        self.lr = lr
        self.batch_size = batch_size
        self.patch_size = [patch_size]*3
        self.max_epoch = max_epoch
        self.k_fold = k_fold
        self.model_each_epoch_save = model_each_epoch_save 

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SwinUNETR(
            in_channels=1, 
            out_channels=len(self.lm_keys), 
            feature_size=48, 
            use_checkpoint=True
        ).to(self.device)
        self.loss_function = GLiPLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.postprocessor = SoftArgMax()
        self.train_loss_values = []
        self.val_loss_values = []


    def get_dataloaders(self, fold):

        logger.info('data loading...')
        builder = BuilderDataList(self.dataset_dir, self.project, self.k_fold, self.lm_keys)
        # get image and label paths dict of train and valid  each k fold
        train_files = builder.get_kfold_file_list_from_txt_path(fold, 'train')
        val_files = builder.get_kfold_file_list_from_txt_path(fold, 'valid')

        train_ds = CacheDataset(data_dicts=train_files, 
                                pre_cache_transform=get_cache_transform(),
                                runtime_transform=get_train_transform(self.patch_size, self.device))
        train_loader = DataLoader(train_ds, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    num_workers=0,
                                    collate_fn= list_data_collate)
        val_ds = CacheDataset(data_dicts=val_files,
                                pre_cache_transform=get_cache_transform(),
                                runtime_transform=get_val_transform(self.patch_size, self.device))
        val_loader = DataLoader(val_ds, 
                                batch_size=self.batch_size, 
                                shuffle=False, 
                                num_workers=0,
                                collate_fn=list_data_collate)
        logger.info('data loading done.')
        self._print_data_info(fold, train_ds, val_ds)
        return train_loader, val_loader
    
    def run_kfold(self):

        self._print_system_info()
        self._print_args_info()

        for fold in range(self.k_fold):
            writer_save_path=os.path.join(self.output_path, 'summary', f'fold{fold+1}')
            os.makedirs(writer_save_path, exist_ok=True)

            vis_save_path=os.path.join(self.output_path, 'visualize', f'fold{fold+1}')
            os.makedirs(vis_save_path, exist_ok=True)

            logger.info(f'fold {fold+1} / {self.k_fold}')
            train_loader, val_loader = self.get_dataloaders(fold+1)
            self.run(train_loader, val_loader, writer_save_path, vis_save_path)

    def run(self, train_loader, val_loader, writer_save_path, vis_save_path):
        writer = SummaryWriter(writer_save_path)

        for epoch in range(1, self.max_epoch+1):
            logger.info(f'epoch {epoch} / {self.max_epoch}')
            avg_train_loss = self.train_one_epoch(train_loader)
            avg_val_loss, avg_dist_per_lm = self.valid_one_epoch(epoch, val_loader, vis_save_path)

            self.train_loss_values.append(avg_train_loss)
            self.val_loss_values.append(avg_val_loss)  

            # log to tensorboard
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)    
            writer.add_scalar('Loss/Valid', avg_val_loss, epoch)  
            for lm_name, dist in zip(self.lm_keys, avg_dist_per_lm):
                tag = f"Val_Dist/{lm_name}"
                writer.add_scalar(tag, dist, epoch)  

            # save checkpoint
            if epoch % self.model_each_epoch_save == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': avg_val_loss
                }
                save_path = os.path.join(self.output_path, 'checkpoint')
                os.makedirs(save_path, exist_ok=True)
                torch.save(checkpoint, os.path.join(save_path, 
                        f'checkpoint_{self.project}_epoch{self.max_epoch}_{self.lr}_{time.strftime("%Y%m%d-%H%M%S")}'))
                logger.info(f'f"Epoch {epoch}: Val loss improved to {avg_val_loss:.4f}. Model saved to {save_path}"')
            
            writer.close()

    def train_one_epoch(self, train_loader):
        self.model.train() 

        for batch_data in tqdm(train_loader, desc='iterate',leave=False):
            step += 1
            inputs, labels = (
                batch_data["image"].to(self.device),
                batch_data["label"].to(self.device),
            )
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / step

    def valid_one_epoch(self, epoch, val_loader, vis_save_path):

        self.model.eval() 
        epoch_loss = 0.0
        epoch_mm_distance = 0.0 
        epoch_mm_distance_per_landmark = None 
        step = 0
        validation_image_saved_this_epoch = False
        
        with torch.no_grad():
            for idx, batch_data in tqdm(enumerate(val_loader), desc='validate', leave=False):
                step += 1
                inputs, labels, gt_coords_mm, affine = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device),
                    batch_data["lm"].to(self.device),
                    batch_data['image'].meta['affine'].to(self.device)
                )
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                epoch_loss += loss.item()

                pred_coords = self.postprocessor(outputs)

                # evaluation by calcurating euclidean dist
                pred_coords_mm = apply_affine_to_points(pred_coords, affine) # voxel to mm
                distance_mm = torch.norm(pred_coords_mm - gt_coords_mm, p=2, dim=-1)
                epoch_mm_distance += torch.mean(distance_mm).item()

                if epoch_pixel_distance_per_landmark is None:
                    num_landmarks = gt_coords_mm.shape[1] 
                    epoch_pixel_distance_per_landmark = np.zeros(num_landmarks)

                batch_mean_dist_per_landmark_mm = torch.mean(distance_mm, dim=0).cpu().numpy()
                
                epoch_pixel_distance_per_landmark += batch_mean_dist_per_landmark_mm

                if not validation_image_saved_this_epoch and idx == 0 and epoch%self.model_each_epoch_save == 0:
                    
                    num_images_to_save = min(3, inputs.shape[0]) 
                    overlay_images_list = [] 

                    overlay_images_list=self._visualize(inputs, 
                                                        labels, 
                                                        outputs, 
                                                        overlay_images_list, 
                                                        num_images_to_save)

                    _save_concatenated_overlays(
                        image_list=overlay_images_list,
                        epoch=epoch,
                        save_dir=vis_save_path
                    )
                    
                    validation_image_saved_this_epoch = True

            avg_loss = epoch_loss / step
            avg_distance_per_landmark_mm = epoch_mm_distance_per_landmark / step
            avg_distance_total_mm = np.mean(avg_distance_per_landmark_mm)
            logger.info(
                f"Epoch {epoch} Val Loss: {avg_loss:.4f}, "
                f"Avg Euclidean Distance (Total): {avg_distance_total_mm:.4f} mm"
            )

            dist_str_list = []
            for i, dist in enumerate(avg_distance_per_landmark_mm):
                if hasattr(self, 'lm_keys') and i < len(self.lm_keys):
                    lm_name = self.lm_keys[i]
                    dist_str_list.append(f"{lm_name}: {dist:.4f}")
                else:
                    dist_str_list.append(f"LM_{i}: {dist:.4f}")
            
            logger.info(f"Avg Distance per Landmark (pixels): [{', '.join(dist_str_list)}]")
                                
        return avg_loss, avg_distance_per_landmark_mm
    

    def _print_system_info(self):
        if self.device == "cuda":
            logger.info("--- GPU Information ---")
            logger.info(f"Device Name:     {torch.cuda.get_device_name(self.device)}")
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            total_mem_gb = total_mem / (1024**3)
            logger.info(f"Total Memory:    {total_mem_gb:.2f} GB")
        else:
            logger.info("--- Device Information ---")
            logger.info(f"Using device: {self.device}") 
    
    def _print_args_info(self):
        logger.info("--- Hyperparameters ---")
        logger.info(f'landmark num: {len(self.lm_keys)}')
        logger.info(f'patch size: {self.patch_size}')
        logger.info(f'batch size: {self.batch_size}')
        logger.info(f'max epoch: {self.lr}')
        logger.info(f'k fold: {self.k_fold if self.k_fold != 0 else False}')
        logger.info(f'model each epoch save: {self.model_each_epoch_save}')
        logger.info("-------------------------")

    def _print_data_info(self, current_fold_num, train_ds, val_ds):
        logger.info(f"Fold {current_fold_num}: DataLoaders are ready.")
        logger.info(f"  Training dataset size:   {len(train_ds)} samples")
        logger.info(f"  Validation dataset size: {len(val_ds)} samples")

    def _visualize(self, inputs, labels, outputs, overlay_images_list, num_images_to_save):
        for i in range(num_images_to_save):
            raw_img = inputs[i].detach().cpu()
            label_img = labels[i].detach().cpu()
            pred_img = outputs[i].detach().cpu()
            if raw_img.dim() == 4: # (C, D, H, W)
                center_slice_idx = raw_img.shape[1] // 2
                raw_img_slice = raw_img[:, center_slice_idx, :, :]
                label_img_slice = label_img[:, center_slice_idx, :, :]
                pred_img_slice = pred_img[:, center_slice_idx, :, :]
            if raw_img_slice.shape[0] == 1:
                raw_img_3ch = raw_img_slice.repeat(3, 1, 1)
            else:
                raw_img_3ch = raw_img_slice

            if label_img_slice.shape[0] == 1:
                label_img_3ch = label_img_slice.repeat(3, 1, 1)
            else:
                label_img_3ch = label_img_slice

            if pred_img_slice.shape[0] == 1:
                pred_img_3ch = pred_img_slice.repeat(3, 1, 1)
            else:
                pred_img_3ch = pred_img_slice
            
            overlay_pil = _create_hotspot_overlay_image(
                raw_img_tensor=raw_img_3ch,
                label_img_tensor=label_img_3ch,
                pred_img_tensor=pred_img_3ch,
                crop_size=256
            )
            overlay_images_list.append(overlay_pil)
        return overlay_images_list