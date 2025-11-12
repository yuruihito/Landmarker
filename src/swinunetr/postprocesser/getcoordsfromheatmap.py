import torch
import torch.nn.functional as F

class SoftArgMax: 
    def __init__(self, percentile: float = 0.95):
        self.percentile = percentile

        self.z_grid = None
        self.y_grid = None
        self.x_grid = None
        self._cached_shape = None
        self._cached_device = None

    def forward(self, heatmap_tensor: torch.Tensor) -> torch.Tensor:
        if not heatmap_tensor.is_cuda:
            print("Warning: Heatmap is not on CUDA. Soft-Argmax can be slow on CPU.")

        self._check_and_create_grids(
            spatial_dims=heatmap_tensor.shape[2:],
            device=heatmap_tensor.device,
            dtype=heatmap_tensor.dtype
        )
        
        processed_heatmap = self._apply_threshold(heatmap_tensor)
        probs_reshaped = self._calculate_probabilities(processed_heatmap)
        final_coords = self._calculate_expected_coords(probs_reshaped)
        
        return final_coords
    
    def _check_and_create_grids(self, spatial_dims: tuple, device: torch.device, dtype: torch.dtype):
        D, H, W = spatial_dims
        new_shape = (D, H, W)
        
        if self._cached_shape != new_shape or self._cached_device != device:
            
            z_coords = torch.linspace(0.0, D - 1.0, D, device=device, dtype=dtype)
            y_coords = torch.linspace(0.0, H - 1.0, H, device=device, dtype=dtype)
            x_coords = torch.linspace(0.0, W - 1.0, W, device=device, dtype=dtype)
            
            self.z_grid = z_coords.view(1, 1, D, 1, 1)
            self.y_grid = y_coords.view(1, 1, 1, H, 1)
            self.x_grid = x_coords.view(1, 1, 1, 1, W)
            
            self._cached_shape = new_shape
            self._cached_device = device

    def _apply_threshold(self, heatmap_tensor: torch.Tensor) -> torch.Tensor:

        if self.percentile is None:
            return heatmap_tensor
        
        B, N, D, H, W = heatmap_tensor.shape
        device = heatmap_tensor.device
        dtype = heatmap_tensor.dtype
        
        heatmap_flat = heatmap_tensor.view(B, N, -1)
        thresholds = torch.quantile(heatmap_flat, self.percentile, dim=2, keepdim=True)
        thresholds = thresholds.view(B, N, 1, 1, 1)
        processed_heatmap = torch.where(
            heatmap_tensor >= thresholds, 
            heatmap_tensor, 
            torch.tensor(-float('inf'), device=device, dtype=dtype)
        )
        return processed_heatmap

    def _calculate_probabilities(self, processed_heatmap: torch.Tensor) -> torch.Tensor:

        B, N, D, H, W = processed_heatmap.shape
        heatmap_flat = processed_heatmap.view(B, N, -1)
        probs_flat = F.softmax(heatmap_flat, dim=2)

        return probs_flat.view(B, N, D, H, W)

    def _calculate_expected_coords(self, probs_reshaped: torch.Tensor) -> torch.Tensor:

        expected_z = torch.sum(probs_reshaped * self.z_grid, dim=(2, 3, 4))
        expected_y = torch.sum(probs_reshaped * self.y_grid, dim=(2, 3, 4))
        expected_x = torch.sum(probs_reshaped * self.x_grid, dim=(2, 3, 4))
        
        return torch.stack([expected_z, expected_y, expected_x], dim=2)