from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import torch
from torchtyping import TensorType

ImageTensor = TensorType["C", "H", "W"]
"""
Shall be normalized to [0, 1] range.
"""

LabelsTensor = TensorType["N"]


class BoxFormat(str, Enum):
    """
    Bounding box format enumeration.

    Formats:
    - AABB: Axis-Aligned Bounding Box - (x1, y1, x2, y2) format, shape (N, 4)
    - XYWHR: Oriented box - (center_x, center_y, width, height, rotation) format, shape (N, 5)
    """

    AABB = "aabb"  # Axis-Aligned: (x1, y1, x2, y2), shape (N, 4)
    XYWHR = "xywhr"  # Oriented: (cx, cy, w, h, rotation), shape (N, 5)


# Tensor type aliases for different box formats
AlignedBoxesTensor = TensorType["N", 4]
"""
Axis-Aligned Bounding Box tensor: (x1, y1, x2, y2) format.
Shape: (N, 4) where N is the number of boxes.
"""

OrientedBoxesTensor = TensorType["N", 5]
"""
Oriented Bounding Box tensor: (center_x, center_y, width, height, rotation) format.
Shape: (N, 5) where N is the number of boxes.
Rotation is in radians.
"""


class DetectionBundleDict(TypedDict):
    """Legacy TypedDict for type hints only. Use DetectionBundle class instead."""

    image: ImageTensor
    boxes: AlignedBoxesTensor
    labels: LabelsTensor


@dataclass(frozen=True)
class DetectionBundle:
    """
    A validated bundle containing image, bounding boxes, and labels for object detection.

    Supports multiple box formats:
    - AABB: Axis-aligned boxes (x1, y1, x2, y2) - shape (N, 4)
    - XYWHR: Oriented boxes (cx, cy, w, h, rotation) - shape (N, 5)

    This class provides:
    - Runtime validation of tensor shapes and types at construction
    - Format-specific validation (AABB vs XYWHR)
    - Explicit attribute access (.image, .boxes, .labels, .box_format)
    - Factory methods for creating from various sources
    - as_dict() method for converting to dictionary when needed

    Examples:
        >>> # AABB format (axis-aligned)
        >>> bundle = DetectionBundle(
        ...     image=img_tensor,
        ...     boxes=aabb_tensor,  # (N, 4)
        ...     labels=labels_tensor,
        ...     box_format=BoxFormat.AABB,
        ... )
        >>>
        >>> # XYWHR format (oriented)
        >>> bundle = DetectionBundle(
        ...     image=img_tensor,
        ...     boxes=xywhr_tensor,  # (N, 5)
        ...     labels=labels_tensor,
        ...     box_format=BoxFormat.XYWHR,
        ... )
        >>>
        >>> # Attribute access
        >>> image = bundle.image
        >>>
        >>> # Convert to dict for unpacking (e.g., for ONNX export)
        >>> model(**bundle.as_dict())
    """

    image: ImageTensor
    boxes: AlignedBoxesTensor | OrientedBoxesTensor
    labels: LabelsTensor
    box_format: BoxFormat

    def __post_init__(self) -> None:
        """Validate tensor shapes and constraints at construction time."""
        # Validate image tensor: must be CHW format (3D)
        if self.image.dim() != 3:
            msg = f"Image must be CHW format (3D tensor), got {self.image.dim()}D tensor with shape {self.image.shape}"
            raise ValueError(msg)

        C, _H, _W = self.image.shape
        if C not in (1, 3, 4):
            msg = f"Image channels must be 1, 3, or 4, got {C}"
            raise ValueError(msg)

        # Validate boxes tensor based on format
        if self.box_format == BoxFormat.AABB:
            if self.boxes.dim() != 2:
                msg = (
                    f"AABB boxes must be 2D tensor (N, 4), got {self.boxes.dim()}D tensor with shape {self.boxes.shape}"
                )
                raise ValueError(msg)

            if self.boxes.shape[1] != 4:
                msg = f"AABB boxes must have 4 columns (x1, y1, x2, y2), got {self.boxes.shape[1]}"
                raise ValueError(msg)

            N_boxes = self.boxes.shape[0]

            # Validate AABB box coordinates are reasonable (x2 > x1, y2 > y1)
            if N_boxes > 0:
                invalid_boxes = (self.boxes[:, 2] <= self.boxes[:, 0]) | (self.boxes[:, 3] <= self.boxes[:, 1])
                if invalid_boxes.any():
                    num_invalid = invalid_boxes.sum().item()
                    warnings.warn(
                        f"Found {num_invalid} invalid AABB boxes where x2 <= x1 or y2 <= y1. "
                        "Consider filtering these before creating DetectionBundle.",
                        UserWarning,
                        stacklevel=2,
                    )

        elif self.box_format == BoxFormat.XYWHR:
            if self.boxes.dim() != 2:
                msg = f"XYWHR boxes must be 2D tensor (N, 5), got {self.boxes.dim()}D tensor with shape {self.boxes.shape}"
                raise ValueError(msg)

            if self.boxes.shape[1] != 5:
                msg = f"XYWHR boxes must have 5 columns (cx, cy, w, h, rotation), got {self.boxes.shape[1]}"
                raise ValueError(msg)

            N_boxes = self.boxes.shape[0]

            # Validate XYWHR box dimensions are positive
            if N_boxes > 0:
                invalid_widths = self.boxes[:, 2] <= 0
                invalid_heights = self.boxes[:, 3] <= 0
                if invalid_widths.any() or invalid_heights.any():
                    num_invalid = (invalid_widths | invalid_heights).sum().item()
                    warnings.warn(
                        f"Found {num_invalid} invalid XYWHR boxes with non-positive width or height. "
                        "Consider filtering these before creating DetectionBundle.",
                        UserWarning,
                        stacklevel=2,
                    )
        else:
            msg = f"Unknown box format: {self.box_format}"
            raise ValueError(msg)

        # Validate labels tensor: must be 1D with N elements
        if self.labels.dim() != 1:
            msg = f"Labels must be 1D tensor (N,), got {self.labels.dim()}D tensor with shape {self.labels.shape}"
            raise ValueError(msg)

        N_labels = self.labels.shape[0]

        # Validate that boxes and labels have matching N
        if N_boxes != N_labels:
            msg = f"Mismatch: boxes has {N_boxes} entries but labels has {N_labels} entries"
            raise ValueError(msg)

        # Warn if image values are in an unexpected range
        # Note: After ImageNet normalization ((x - mean) / std), values will be outside [0, 1]
        # Typical ImageNet-normalized range is approximately [-2.5, 2.5]
        # Values > 10 or < -10 suggest improper normalization or unnormalized input
        img_min = self.image.min().item()
        img_max = self.image.max().item()
        # Only warn for clearly problematic ranges (suggests unnormalized [0, 255] input or normalization error)
        if img_max > 10.0 or img_min < -10.0:
            warnings.warn(
                f"Image values in unexpected range: min={img_min:.3f}, max={img_max:.3f}. "
                "Expected normalized values (typically [0, 1] for raw images or [-2.5, 2.5] after ImageNet normalization). "
                "Values > 10 suggest input may not have been properly scaled to [0, 1] before normalization.",
                UserWarning,
                stacklevel=2,
            )

    def as_dict(self) -> dict[str, torch.Tensor | BoxFormat]:
        """
        Convert to a plain dictionary.

        Useful for explicit conversion when needed (e.g., for ONNX export).
        Note: box_format is included in the dict.
        """
        return {
            "image": self.image,
            "boxes": self.boxes,
            "labels": self.labels,
            "box_format": self.box_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, torch.Tensor | BoxFormat]) -> DetectionBundle:
        """
        Create DetectionBundle from a dictionary.

        Args:
            data: Dictionary with 'image', 'boxes', 'labels', and 'box_format' keys

        Returns:
            Validated DetectionBundle instance
        """
        return cls(
            image=data["image"],
            boxes=data["boxes"],
            labels=data["labels"],
            box_format=data.get("box_format", BoxFormat.AABB),  # Default to AABB for backward compat
        )

    @classmethod
    def from_tensors(
        cls,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        box_format: BoxFormat = BoxFormat.AABB,
    ) -> DetectionBundle:
        """
        Create DetectionBundle from individual tensors.

        Args:
            image: Image tensor in CHW format
            boxes: Bounding boxes tensor - shape (N, 4) for AABB or (N, 5) for XYWHR
            labels: Labels tensor in N format
            box_format: Format of the boxes (AABB or XYWHR)

        Returns:
            Validated DetectionBundle instance
        """
        return cls(image=image, boxes=boxes, labels=labels, box_format=box_format)

    def to(self, device: torch.device | str) -> DetectionBundle:
        """
        Move all tensors to the specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device('cuda:0'))

        Returns:
            New DetectionBundle with tensors on the target device
        """
        return DetectionBundle(
            image=self.image.to(device),
            boxes=self.boxes.to(device),
            labels=self.labels.to(device),
            box_format=self.box_format,
        )


BatchedImageTensor = TensorType["B", "C", "H", "W"]
BatchedAlignedBoxesTensor = list[AlignedBoxesTensor]
BatchedOrientedBoxesTensor = list[OrientedBoxesTensor]
BatchedLabelsTensor = list[LabelsTensor]


class BatchedDetectionBundleDict(TypedDict):
    """Legacy TypedDict for type hints only. Use BatchedDetectionBundle class instead."""

    image: BatchedImageTensor
    boxes: BatchedAlignedBoxesTensor
    labels: BatchedLabelsTensor


@dataclass(frozen=True)
class BatchedDetectionBundle:
    """
    A validated batch of DetectionBundle objects.

    Provides the same benefits as DetectionBundle but for batched data.
    All items in the batch must have the same box_format.
    """

    image: BatchedImageTensor
    boxes: BatchedAlignedBoxesTensor | BatchedOrientedBoxesTensor
    labels: BatchedLabelsTensor
    box_format: BoxFormat

    def __post_init__(self) -> None:
        """Validate batched tensor shapes and constraints."""
        # Validate batched image tensor: must be BCHW format (4D)
        if self.image.dim() != 4:
            msg = f"Batched image must be BCHW format (4D tensor), got {self.image.dim()}D tensor with shape {self.image.shape}"
            raise ValueError(msg)

        B, C, _H, _W = self.image.shape
        if C not in (1, 3, 4):
            msg = f"Image channels must be 1, 3, or 4, got {C}"
            raise ValueError(msg)

        # Validate boxes and labels are lists
        if not isinstance(self.boxes, list):
            msg = f"Boxes must be a list of tensors, got {type(self.boxes)}"
            raise ValueError(msg)

        if not isinstance(self.labels, list):
            msg = f"Labels must be a list of tensors, got {type(self.labels)}"
            raise ValueError(msg)

        # Validate batch size consistency
        if len(self.boxes) != B:
            msg = f"Mismatch: image batch size is {B} but boxes list has {len(self.boxes)} items"
            raise ValueError(msg)

        if len(self.labels) != B:
            msg = f"Mismatch: image batch size is {B} but labels list has {len(self.labels)} items"
            raise ValueError(msg)

        # Validate each box/label pair in the batch
        expected_cols = 4 if self.box_format == BoxFormat.AABB else 5
        format_name = "AABB" if self.box_format == BoxFormat.AABB else "XYWHR"

        for i, (box_tensor, label_tensor) in enumerate(zip(self.boxes, self.labels)):
            if box_tensor.dim() != 2:
                msg = f"Boxes[{i}] must be 2D tensor (N, {expected_cols}), got shape {box_tensor.shape}"
                raise ValueError(msg)

            if box_tensor.shape[1] != expected_cols:
                msg = (
                    f"Boxes[{i}] must have {expected_cols} columns for {format_name} format, got {box_tensor.shape[1]}"
                )
                raise ValueError(msg)

            if label_tensor.dim() != 1:
                msg = f"Labels[{i}] must be 1D tensor (N,), got shape {label_tensor.shape}"
                raise ValueError(msg)

            if box_tensor.shape[0] != label_tensor.shape[0]:
                msg = (
                    f"Mismatch in batch item {i}: boxes has {box_tensor.shape[0]} entries "
                    f"but labels has {label_tensor.shape[0]} entries"
                )
                raise ValueError(msg)

    def as_dict(self) -> dict[str, torch.Tensor | list[torch.Tensor] | BoxFormat]:
        """Convert to a plain dictionary."""
        return {
            "image": self.image,
            "boxes": self.boxes,
            "labels": self.labels,
            "box_format": self.box_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, torch.Tensor | list[torch.Tensor] | BoxFormat]) -> BatchedDetectionBundle:
        """Create BatchedDetectionBundle from a dictionary."""
        return cls(
            image=data["image"],
            boxes=data["boxes"],
            labels=data["labels"],
            box_format=data.get("box_format", BoxFormat.AABB),  # Default to AABB for backward compat
        )

    def to(self, device: torch.device | str) -> BatchedDetectionBundle:
        """Move all tensors to the specified device."""
        return BatchedDetectionBundle(
            image=self.image.to(device),
            boxes=[box.to(device) for box in self.boxes],
            labels=[label.to(device) for label in self.labels],
            box_format=self.box_format,
        )
