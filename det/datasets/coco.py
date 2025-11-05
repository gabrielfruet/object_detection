from __future__ import annotations

from typing import TYPE_CHECKING, cast

import supervision as sv
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToDtype, ToImage, Transform

from det.utils.img import image_numpy_to_tensor

if TYPE_CHECKING:
    from pathlib import Path

    from det.types.detection import (
        AlignedBoxesTensor,
        BatchedDetectionBundle,
        BatchedImageTensor,
        DetectionBundle,
        ImageTensor,
        LabelsTensor,
    )


class ResizeWithBoundingBox(Transform):
    size: tuple[int, int]

    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.size = size

    def forward(self, detection_bundle: DetectionBundle) -> DetectionBundle:
        image = detection_bundle["image"]
        boxes = detection_bundle["boxes"]
        labels = detection_bundle["labels"]

        original_height, original_width = image.shape[1], image.shape[2]
        resized_image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        scale_x = self.size[1] / original_width
        scale_y = self.size[0] / original_height

        resized_boxes = boxes.clone()
        resized_boxes[:, 0] = boxes[:, 0] * scale_x
        resized_boxes[:, 1] = boxes[:, 1] * scale_y
        resized_boxes[:, 2] = boxes[:, 2] * scale_x
        resized_boxes[:, 3] = boxes[:, 3] * scale_y

        return {
            "image": cast("ImageTensor", resized_image),
            "boxes": cast("AlignedBoxesTensor", resized_boxes),
            "labels": cast("LabelsTensor", labels),
        }


class CocoDataset(Dataset):
    images_path: Path
    annotations_path: Path
    detection_dataset: sv.DetectionDataset
    transforms: Compose

    def __init__(self, dataset_path: Path, transforms=None, resize: tuple[int, int] | None = None) -> None:
        self.images_path = dataset_path
        resize = resize or (720, 1280)
        try:
            self.annotations_path = next(dataset_path.glob("*.json"))
        except StopIteration:
            msg = f"No JSON annotation file found in {dataset_path}"
            raise FileNotFoundError(msg) from None
        self.transforms = transforms or Compose(
            [
                ToImage(),
                ToDtype(torch.float32, scale=True),
                ResizeWithBoundingBox(size=resize),
            ]
        )

        self.detection_dataset = self.to_detection_dataset()

    def to_detection_dataset(self) -> sv.DetectionDataset:
        return sv.DetectionDataset.from_coco(str(self.images_path), str(self.annotations_path))

    @staticmethod
    def collate_fn(batch: list[DetectionBundle]) -> BatchedDetectionBundle:
        images = cast("BatchedImageTensor", torch.stack([item["image"] for item in batch]))
        boxes = [item["boxes"] for item in batch]
        labels = [item["labels"] for item in batch]

        return {
            "image": images,
            "boxes": boxes,
            "labels": labels,
        }

    @property
    def class_names(self) -> list[str]:
        return self.detection_dataset.classes

    def _filter_degraded_boxes(self, detection_bundle: DetectionBundle) -> DetectionBundle:
        boxes = detection_bundle["boxes"]
        labels = detection_bundle["labels"]

        valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

        filtered_boxes = boxes[valid_indices]
        filtered_labels = labels[valid_indices]

        return {
            "image": cast("ImageTensor", detection_bundle["image"]),
            "boxes": cast("AlignedBoxesTensor", filtered_boxes),
            "labels": cast("LabelsTensor", filtered_labels),
        }

    def __getitem__(self, index: int) -> DetectionBundle:
        _path, image, detections = self.detection_dataset[index]

        detection_bundle: DetectionBundle = {
            "image": cast("ImageTensor", image_numpy_to_tensor(image)),
            "boxes": cast(
                "AlignedBoxesTensor",
                torch.tensor(detections.xyxy, dtype=torch.float32),
            ),
            "labels": cast("LabelsTensor", torch.tensor(detections.class_id)),
        }

        detection_bundle = self._filter_degraded_boxes(detection_bundle)

        return self.transforms(detection_bundle)

    def __len__(self) -> int:
        return len(self.detection_dataset)
