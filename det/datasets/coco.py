from __future__ import annotations

from typing import TYPE_CHECKING

import albumentations as A
import numpy as np
import supervision as sv
import torch
from torch.utils.data import Dataset

from det.types.detection import BatchedAlignedBoxesTensor, BatchedImageTensor, BatchedLabelsTensor, BoxFormat
from det.utils.img import image_numpy_to_tensor

if TYPE_CHECKING:
    from pathlib import Path

    from det.types.detection import (
        BatchedDetectionBundle,
        DetectionBundle,
    )
    from det.types.split import DatasetSplit
else:
    from det.types.detection import (
        BatchedDetectionBundle,
        BoxFormat,
        DetectionBundle,
    )


class CocoDataset(Dataset):
    COCO_SPLIT_MAPPING: dict[DatasetSplit, str] = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }

    images_path: Path
    annotations_path: Path
    detection_dataset: sv.DetectionDataset
    transforms: A.Compose | None

    def __init__(
        self,
        dataset_path: Path,
        split: DatasetSplit,
        transforms: A.Compose | None = None,
        resize: tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> None:
        self.images_path = self.get_coco_split_path(dataset_path, split)
        resize = resize or (720, 1280)
        try:
            self.annotations_path = next(self.images_path.glob("*.json"))
        except StopIteration:
            msg = f"No JSON annotation file found in {self.images_path}"
            raise FileNotFoundError(msg) from None

        # Build albumentations transform pipeline
        if transforms is None:
            transform_list = [
                # Convert to float32 and scale [0, 255] -> [0, 1]
                A.ToFloat(max_value=255.0),
                # Resize with random crop (handles both image and boxes)
                # RandomResizedCrop already resizes, so no need for separate Resize
                A.RandomResizedCrop(size=resize, scale=(0.7, 1.0), p=1.0),
                # Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]

            # Add normalization if requested (must be after ToFloat)
            if normalize:
                transform_list.append(
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=1.0,
                    )
                )

            self.transforms = A.Compose(
                transform_list,
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            )
        else:
            self.transforms = transforms

        self.detection_dataset = self.to_detection_dataset()

    @classmethod
    def get_coco_split_path(cls, dataset_path: Path, dataset_split: DatasetSplit) -> Path:
        return dataset_path / cls.COCO_SPLIT_MAPPING[dataset_split]

    def to_detection_dataset(self) -> sv.DetectionDataset:
        return sv.DetectionDataset.from_coco(str(self.images_path), str(self.annotations_path))

    @staticmethod
    def collate_fn(batch: list[DetectionBundle]) -> BatchedDetectionBundle:
        images: BatchedImageTensor = torch.stack([item.image for item in batch])
        boxes: BatchedAlignedBoxesTensor = [item.boxes for item in batch]
        labels: BatchedLabelsTensor = [item.labels for item in batch]

        # Ensure all items have the same box_format
        box_formats = [item.box_format for item in batch]
        if len(set(box_formats)) > 1:
            msg = f"All items in batch must have the same box_format, got {set(box_formats)}"
            raise ValueError(msg)
        box_format = box_formats[0]

        return BatchedDetectionBundle(
            image=images,
            boxes=boxes,
            labels=labels,
            box_format=box_format,
        )

    @property
    def class_names(self) -> list[str]:
        return self.detection_dataset.classes

    def _filter_degraded_boxes(self, detection_bundle: DetectionBundle) -> DetectionBundle:
        boxes = detection_bundle["boxes"]
        labels = detection_bundle["labels"]
        box_format = detection_bundle["box_format"]

        # Filtering currently only supports AABB format
        if box_format != BoxFormat.AABB:
            msg = f"_filter_degraded_boxes only supports AABB format, got {box_format}"
            raise ValueError(msg)

        # using same degenerate box validation as torchvision FCOS
        # Handle empty boxes case
        if boxes.shape[0] == 0:
            # No boxes to filter, return as-is
            filtered_boxes = boxes
            filtered_labels = labels
        else:
            valid_indices = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
            filtered_boxes = boxes[valid_indices]
            filtered_labels = labels[valid_indices]

        return DetectionBundle(
            image=detection_bundle["image"],
            boxes=filtered_boxes,
            labels=filtered_labels,
            box_format=BoxFormat.AABB,
        )

    def __getitem__(self, index: int) -> DetectionBundle:
        _path, image, detections = self.detection_dataset[index]

        # Supervision returns numpy arrays (H, W, C) in [0, 255] uint8 format
        # Convert to format expected by albumentations
        image_np = np.asarray(image, dtype=np.uint8)
        boxes_np = detections.xyxy.astype(np.float32)
        labels_np = detections.class_id.astype(np.int64)

        # Apply albumentations transforms
        if self.transforms is not None:
            augmented = self.transforms(
                image=image_np,
                bboxes=boxes_np,
                class_labels=labels_np,
            )
            image_np = augmented["image"]
            boxes_np = np.array(augmented["bboxes"], dtype=np.float32)
            labels_np = np.array(augmented["class_labels"], dtype=np.int64)

        # Convert to DetectionBundle
        # Albumentations returns image in (H, W, C) format, convert to (C, H, W) tensor
        image_tensor = image_numpy_to_tensor(image_np)
        boxes_tensor = torch.tensor(boxes_np, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_np, dtype=torch.int64)

        detection_bundle = DetectionBundle(
            image=image_tensor,
            boxes=boxes_tensor,
            labels=labels_tensor,
            box_format=BoxFormat.AABB,  # COCO uses AABB format
        )

        # Filter degraded boxes
        return self._filter_degraded_boxes(detection_bundle)

    def __len__(self) -> int:
        return len(self.detection_dataset)
