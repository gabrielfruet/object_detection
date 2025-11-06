import numpy as np
import torch


def image_numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
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
    # Use from_numpy to avoid copy if possible
    return torch.from_numpy(image).permute(2, 0, 1)


def image_tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
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


def image_scale_to_uint8_numpy(image: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8 in [0, 255].

    Args:
        image (np.ndarray): Input image as a NumPy array with float values in [0, 1].

    Returns:
        np.ndarray: Image as a NumPy array with uint8 values in [0, 255].
    """
    if not np.issubdtype(image.dtype, np.floating):
        msg = "Input image must have float dtype."
        raise ValueError(msg)
    if image.min() < 0.0 or image.max() > 1.0:
        msg = "Input image values must be in the range [0, 1]."
        raise ValueError(msg)
    return (image * 255).astype(np.uint8)
