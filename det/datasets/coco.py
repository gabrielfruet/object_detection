from __future__ import annotations

from typing import TYPE_CHECKING, cast

import supervision as sv
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image

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
                Resize(resize),
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
            "image": cast("ImageTensor", Image(torch.tensor(image).permute(2, 0, 1))),
            "boxes": cast(
                "AlignedBoxesTensor",
                BoundingBoxes._wrap(
                    torch.tensor(detections.xyxy).to(torch.float32),
                    format=BoundingBoxFormat.XYXY,
                    canvas_size=(int(image.shape[0]), int(image.shape[1])),
                ),
            ),
            "labels": cast("LabelsTensor", torch.tensor(detections.class_id)),
        }

        detection_bundle = self._filter_degraded_boxes(detection_bundle)

        return self.transforms(detection_bundle)

    def __len__(self) -> int:
        return len(self.detection_dataset)
