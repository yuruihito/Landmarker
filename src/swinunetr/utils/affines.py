import numpy as np
from nibabel.affines import apply_affine

#逆変換用ヘルパー関数
def apply_inverse_affine(affine, pts):
    inv_affine = np.linalg.inv(affine)
    return apply_affine(inv_affine, pts)