import numpy as np
import torch


def numpy_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a NumPy image array to a PyTorch tensor.

    Args:
        image (np.ndarray): Input image as a NumPy array with shape (H, W, C).

    Returns:
        torch.Tensor: Image as a PyTorch tensor with shape (C, H, W).
    """
    if image.ndim != 3:
        msg = "Input image must have 3 dimensions (H, W, C)."
        raise ValueError(msg)
    # Convert from HWC to CHW format
    image_chw = np.transpose(image, (2, 0, 1))
    # Convert to torch tensor
    return torch.from_numpy(image_chw).float()


def tensor_image_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor image to a NumPy array.

    Args:
        image_tensor (torch.Tensor): Input image as a PyTorch tensor with shape (C, H, W).

    Returns:
        np.ndarray: Image as a NumPy array with shape (H, W, C).
    """
    if image_tensor.ndim != 3:
        msg = "Input image tensor must have 3 dimensions (C, H, W)."
        raise ValueError(msg)
    # Convert to NumPy array
    image_np = image_tensor.cpu().numpy()
    # Convert from CHW to HWC format
    return np.ascontiguousarray(np.transpose(image_np, (1, 2, 0)))
