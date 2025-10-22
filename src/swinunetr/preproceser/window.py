import numpy as np

def window(img, lower_percentile=5, upper_percentile=95):
    # calculate 5-95% and clip img
    lower_bound = np.percentile(img, lower_percentile)
    upper_bound = np.percentile(img, upper_percentile)
    clipped_img = np.clip(img, lower_bound, upper_bound)

    scaled_img = (clipped_img - lower_bound) / (upper_bound - lower_bound) * 255

    return scaled_img.astype(np.int8)