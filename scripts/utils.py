import numpy as np
import torch
import cv2
from torchvision import transforms as T

# Constants
INPUT_IMAGE_SIZE = (1024, 1024)
HEATMAP_SIZE = (256, 256)  # (height, width)
GAUSSIAN_RADIUS = 3        # Standard deviation in pixels

def get_transforms(input_size=INPUT_IMAGE_SIZE):
    return T.Compose([
        T.Resize(input_size),
        T.ToTensor()
    ])

def create_heatmap(keypoints, heatmap_size=HEATMAP_SIZE, radius=GAUSSIAN_RADIUS):
    """
    Create a single-channel heatmap from a list of (x, y) keypoints.
    """
    heatmap = np.zeros(heatmap_size, dtype=np.float32)
    H, W = heatmap_size

    for x, y in keypoints:
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        x = int(x)
        y = int(y)
        tmp = np.zeros((H, W), dtype=np.float32)
        tmp[y, x] = 1
        tmp = cv2.GaussianBlur(tmp, (0, 0), sigmaX=radius, sigmaY=radius)
        tmp = tmp / tmp.max()  # Normalize peak to 1
        heatmap = np.maximum(heatmap, tmp)

    return torch.tensor(heatmap).unsqueeze(0)  # shape: (1, H, W)

def extract_keypoints_from_heatmap(heatmap_tensor, threshold=0.3, min_distance=4):
    """
    Convert predicted heatmap to list of (x, y) center points
    Input:
        heatmap_tensor: torch.Tensor of shape (1, H, W)
    Returns:
        List of (x, y) keypoints
    """
    import numpy as np
    from skimage.feature import peak_local_max

    heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-6)
    heatmap_thresh = np.where(heatmap_np > threshold, heatmap_np, 0)

    keypoints = peak_local_max(
        heatmap_thresh,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False
    )

    # Convert (row, col) to (x, y)
    return [(int(x), int(y)) for y, x in keypoints]
