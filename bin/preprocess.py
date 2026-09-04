import os

import argparse
import numpy as np
from tqdm import tqdm

from src.swinunetr.preproceser.preprocesser import SwinUNETRPreProcesser
from src.swinunetr.preproceser.window import window
from src.swinunetr.preproceser.mkheatmap import create_multi_channels_heatmaps_3d
from src.swinunetr.utils.visualizer import preprocesss_visualizer

"""
The structure of the dataset to be prepared
dataset | 
    project |
        input |
            id001 |
                raw.mhd
            id002 |
                raw.mhd 
        fcsv | 
            id001 |
                landmark.fcsv
            id002 |
                landmark.fcsv
        preprocessed | (output)
            id001 |
                raw.mhd(preprocessed)
                label.mhd
                preprocessed.png
            id002 |
                raw.mhd
                label.mhd
                preprocessed.png
        kfold |
            fold1 |
                train.txt
                val.txt
                test.txt
            fold2 |
                train.txt
                val.txt
                test.txt           
"""
def main(args): 

    pre = SwinUNETRPreProcesser(args.dataset_dir, f'{args.project_name}_{args.k_fold}fold', args.k_fold, args.lm_keys)
    img_ids = pre.get_img_ids()
    paths_dict = pre.get_all_path_dict(img_ids)
    points_dict = pre.get_coordinate_points_dict(img_ids, paths_dict)

    for id in tqdm(img_ids, desc='prepraring'):
        img_np, img_sitk, size, spacing, origin, direction = pre.img_load(paths_dict[id]['img_path'])
        id_point = pre.convert_physical_to_pixel_indices(img_sitk, points_dict[id])

        # adapting window process against raw img 
        normalized_np_img = window(img_np)

        # creating heatmaps
        heatmaps = create_multi_channels_heatmaps_3d(size, id_point, args.sigma)
        heatmaps_itk_order = np.transpose(heatmaps, (1, 2, 3, 0))

        # visualize
        preprocesss_visualizer(normalized_np_img, heatmaps, id_point.values(), 
                               id_point['head_center'], paths_dict[id]['output_dir'])

        # set meta 
        set_meta_normalized_img = pre.set_meta(normalized_np_img, 
                                               spacing, origin, direction)
        set_meta_heatmaps = pre.set_meta(heatmaps_itk_order, 
                                         spacing, origin, direction)
        
        # save img
        pre.save_mhd_img(set_meta_normalized_img, 
                         os.path.join(paths_dict[id]['output_dir'], 'raw.mhd'))
        pre.save_mhd_img(set_meta_heatmaps, 
                         os.path.join(paths_dict[id]['output_dir'], 'label.mhd'))

        # save each txt file writed how to divide the data 
        pre.save_kfold_split()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to preprocess images \
                                      and landmarks for training. Reads images, \
                                     normalizes them, and creates Gaussian heatmaps, \
                                     saving the output to a specified path.")
    parser.add_argument('--dataset_dir', type=str,
                        default=r'/mnt/Users/names')
    parser.add_argument('--project_name', type=str,
                        default='project')
    parser.add_argument('--k_fold', type=int,
                        default=4)
    parser.add_argument('--lm_keys', nargs='+',
                        default=['head_center', 'Acetabular_outermost', 'tear_drop'])
    parser.add_argument('--sigma', type=float,
                        default=3.0)
    args = parser.parse_args()
    main(args)
