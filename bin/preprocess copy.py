import os
import json
import numpy as np
import torch
import SimpleITK as sitk
import argparse
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter
from scripts.utils.fcsv_utils import get_coords_from_fcsv_center_list
from scripts.preprocesser.visualizer_for_preprocess import VisualizeForSwinUNET
from scripts.preprocesser.preprocesser import SwinUNetPreProcesser

def process_mri_files(args):
    """
    Process all MRI files in the input folder using global windowing.

    Args:
        input_folder (str): Path to the folder containing MRI images.
        output_folder (str): Path to save processed .npy files.
        
    """
    pre = SwinUNetPreProcesser(args)
    saver = VisualizeForSwinUNET()

    # get config 
    input_folders = pre.get_directory_paths()
    file_paths = pre.get_raw_image_paths(input_folders)
    output_heatmap_dir = pre.get_output_dir("label")
    output_seg_dir = pre.get_output_dir('seg')
    output_mri_dir = pre.get_output_dir("mri")
      
    image_ids = []
    for file_path in file_paths:

        # get image_id and output_path, fcsv path
        image_id, image_ids = pre.get_image_ids(file_path, image_ids)
        seg_path = pre.get_seg_label_path(args.input_dir, image_id)
        output_heatmap_file_folder = pre.get_output_dir(image_id, output_heatmap_dir)
        output_mri_file_folder = pre.get_output_dir(image_id, output_mri_dir)
        output_seg_file_folder = pre.get_output_dir(image_id, output_seg_dir)
        fcsv_path = pre.get_fcsv_path(image_id)

        array_data, image = pre.get_image(file_path)
        
        # Get landmarks addapted to spacing
        points = pre.get_coordinate_points(fcsv_path, image)

        # get ww and wl
        global_lower_bound, global_upper_bound = pre.compute_global_window_bounds(file_path)
        ww, wl = pre.get_ww_and_wl(global_lower_bound, global_upper_bound)

        # window processing
        processed_data = pre.window(array_data, ww, wl)
        heatmaps = pre.create_3d_heatmap(processed_data, points)
        saver.save_concat_image(array_data, heatmaps, output_heatmap_file_folder, points)

        # seg label
        label, _ = pre.get_image(seg_path)
        seg_label = pre.get_seg_label(array_data, label)
        saver.save_label_vis(label, output_seg_file_folder)

        pre.save_mha_file(seg_label, image, output_seg_file_folder, 'seg')
        pre.save_mha_file(heatmaps, image, output_heatmap_file_folder, 'label')
        pre.save_mha_file(processed_data, image, output_mri_file_folder, 'raw')
        print(f"heatmap Saved: {output_heatmap_file_folder}")  
        print(f"seg Saved: {output_seg_file_folder}")
        print(f"mri Saved: {output_mri_file_folder}")

    pre.split_and_save_folds(image_ids, os.path.join(pre.output_dir, f'{pre.dataset_dirname}_sigma{pre.sigma}'), pre.n_splits)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply windowing to MRI images and save as .npy.")
    parser.add_argument("--input_dir", type=str, 
                        default = "/win/scallop/user/kameda/datasets/ONFH_data", help="Path to the folder containing MRI images.")
                        #default = r"Z:\kameda\datasets\ONFH_data\half_cropped_20240609", help="Path to the folder containing MRI images.")
                        #default = "/win/scallop/user/kameda/datasets/ONFH_data/63cases", help="Path to the folder containing MRI images.")
    parser.add_argument("--output_dir", type=str, 
                        default = "/win/scallop/user/kameda/Swin-UNETR/dataset/landmark/40cases", help="Path to save processed dataset.")
                        #default = r"Z:\kameda\Swin-UNETR\dataset\landmark\40cases", help="Path to save processed dataset.")
    parser.add_argument("--fcsv_folder", type=str,
                        #default = "/win/scallop/user/kameda/datasets/ONFH_data/63cases_Landmark", help="Path to the folder containing .fcsv files.")
                        default = "/win/scallop/user/kameda/datasets/ONFH_data/kono_landmarks", help="Path to the folder containing .fcsv files.")
                        #default = r"Z:\kameda\datasets\ONFH_data\kono_landmarks", help="Path to the folder containing .fcsv files.")
    parser.add_argument('--train_valid_dir', type=str, 
                       )
    parser.add_argument('--train_test_dir', type=str, 
                       )
    parser.add_argument('--dataset_dirname', default="40cases", type = str,  
                        help='to save every epochs you set')
    parser.add_argument("--points_name", type=str, nargs='+',
                        default = "head_center", help="Path to the folder containing .fcsv files.")
                        #default = ["FHC"], help="Path to the folder containing .fcsv files.")
    parser.add_argument("--n_splits", type=int,
                        default = 4, help="Path to the folder containing .fcsv files.")
    parser.add_argument("--sigma", type=float,
                        default = 3, help="heatmap gausian sigma")
    parser.add_argument('--taskname', type=str,
                        default='heatmap')
    
    args = parser.parse_args()
    process_mri_files(args)
