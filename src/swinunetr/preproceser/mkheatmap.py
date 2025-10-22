import os
import numpy as np
from typing import Tuple, Union, List

ShareType = Union[Tuple[int, ...], List[int]]

def generate_heatmap_3d(size, landmark, sigma):
    depth, height, width = size
    x_lm, y_lm, z_lm = landmark
    z_grid, y_grid, x_grid = np.mgrid[0:depth, 0:height, 0:width]
    dist = (x_grid - x_lm)**2 + (y_grid - y_lm)**2 + (z_grid - z_lm)**2
    heatmap = np.exp(-dist/(2*sigma**2))
    return heatmap

def create_multi_channels_heatmaps_3d(size: ShareType, landmarks: ShareType, sigma=3.0):

    num_landmarks = len(landmarks)
    depth, height, width = size
    # multi_channels_heatmaps
    heatmaps = np.zeros((num_landmarks, depth, height, width), dtype=np.float32)

    for i, lm in enumerate(landmarks):
        heatmap = generate_heatmap_3d(size, lm, sigma)
        heatmaps[i] = heatmap

    return heatmaps
