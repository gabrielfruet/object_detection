from __future__ import annotations

from typing import TypedDict

from torchtyping import TensorType

ImageTensor = TensorType["C", "H", "W"]
"""
Shall be normalized to [0, 1] range.
"""

AlignedBoxesTensor = TensorType["N", 4]
LabelsTensor = TensorType["N"]


class DetectionBundle(TypedDict):
    image: ImageTensor
    boxes: AlignedBoxesTensor
    labels: LabelsTensor


BatchedImageTensor = TensorType["B", "C", "H", "W"]
BatchedAlignedBoxesTensor = list[AlignedBoxesTensor]
BatchedLabelsTensor = list[LabelsTensor]


class BatchedDetectionBundle(TypedDict):
    image: BatchedImageTensor
    boxes: BatchedAlignedBoxesTensor
    labels: BatchedLabelsTensor
