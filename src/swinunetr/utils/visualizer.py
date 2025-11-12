import os 
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def preprocesss_visualizer(img, heatmaps, landmark_list, target_point, output_dir):
    
    target_slice_idx = target_point[2]
    img_slice = img[int(target_slice_idx), :, :]
    heatmap_slices_all_channels = heatmaps[:, int(target_slice_idx), :, :]
    if heatmap_slices_all_channels.shape[0] > 0:
        combined_heatmap_slice = np.max(heatmap_slices_all_channels, axis=0)
    else:
        combined_heatmap_slice = np.zeros_like(img_slice)

    coords_x = [int(lm[0]) for lm in landmark_list]
    coords_y = [int(lm[1]) for lm in landmark_list]

    plt.figure(figsize=(10, 10))
    plt.imshow(img_slice, cmap='gray', aspect='equal', interpolation='nearest')
    plt.imshow(
        combined_heatmap_slice, 
        cmap='jet',        
        alpha=0.5,         
        aspect='equal',    
        interpolation='nearest'
    )
    plt.scatter(coords_x, coords_y, color='red', marker='x')
    plt.xlabel("Y coordinate")
    plt.ylabel("Z coordinate")
    plt.savefig(os.path.join(output_dir, 'preprocessed.png'))
    plt.close()

def _create_hotspot_overlay_image(
    raw_img_tensor, 
    label_img_tensor, 
    pred_img_tensor, 
    crop_size=256, 
    padding=10):

    label_ch1 = label_img_tensor[0]
    max_flat_idx = torch.argmax(label_ch1)
    center_y, center_x = torch.unravel_index(max_flat_idx, label_ch1.shape)
    center_y = center_y.item()
    center_x = center_x.item()

    raw_pil = TF.to_pil_image(raw_img_tensor)
    label_pil = TF.to_pil_image(label_img_tensor)
    pred_pil = TF.to_pil_image(pred_img_tensor)

    overlay_A = Image.blend(raw_pil, label_pil, alpha=0.5)
    overlay_B = Image.blend(raw_pil, pred_pil, alpha=0.5)

    half_crop = crop_size // 2
    top = center_y - half_crop
    left = center_x - half_crop
    crop_A = TF.crop(overlay_A, top, left, crop_size, crop_size)
    crop_B = TF.crop(overlay_B, top, left, crop_size, crop_size)

    total_width = (crop_size * 2) + padding
    height = crop_size
    canvas = Image.new('RGB', (total_width, height), color=(255, 255, 255))
    canvas.paste(crop_A, (0, 0))
    canvas.paste(crop_B, (crop_size + padding, 0))
    
    return canvas

def _save_concatenated_overlays(
    image_list, 
    epoch, 
    save_dir, 
    padding=10):

    max_width = max(img.width for img in image_list)
    total_height = sum(img.height for img in image_list) + (padding * (len(image_list) - 1))

    canvas = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))

    current_y = 0
    for img in image_list:
        x_offset = (max_width - img.width) // 2
        canvas.paste(img, (x_offset, current_y))
        current_y += img.height + padding

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}_validation_overlays.png")
    canvas.save(save_path)