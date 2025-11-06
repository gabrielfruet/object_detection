from __future__ import annotations

from typing import TYPE_CHECKING, cast

import albumentations as A
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
    from det.types.split import DatasetSplit


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


class AlbumentationsTransform(Transform):
    def __init__(self, aug: A.BaseCompose) -> None:
        super().__init__()
        self.aug = aug

    def forward(self, detection_bundle: DetectionBundle) -> DetectionBundle:
        image = detection_bundle["image"]
        boxes = detection_bundle["boxes"]
        labels = detection_bundle["labels"]

        image_np = image.permute(1, 2, 0).cpu().numpy()

        bboxes = boxes.cpu().numpy()
        class_labels = labels.cpu().numpy()
        albumentations_bboxes = bboxes

        # Apply augmentation
        augmented = self.aug(
            image=image_np,
            bboxes=albumentations_bboxes,
            class_labels=class_labels,
        )

        # Convert back to tensor
        augmented_image = image_numpy_to_tensor(augmented["image"])

        # Extract augmented bounding boxes and labels
        augmented_bboxes = augmented["bboxes"]
        augmented_class_labels = augmented["class_labels"]

        augmented_boxes_tensor = torch.tensor(augmented_bboxes, dtype=torch.float32)
        augmented_labels_tensor = torch.tensor(augmented_class_labels, dtype=torch.int64)

        return {
            "image": cast("ImageTensor", augmented_image),
            "boxes": cast("AlignedBoxesTensor", augmented_boxes_tensor),
            "labels": cast("LabelsTensor", augmented_labels_tensor),
        }


class CocoDataset(Dataset):
    COCO_SPLIT_MAPPING: dict[DatasetSplit, str] = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }

    images_path: Path
    annotations_path: Path
    detection_dataset: sv.DetectionDataset
    transforms: Compose

    def __init__(
        self,
        dataset_path: Path,
        split: DatasetSplit,
        transforms=None,
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

        normalization = A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=1.0,
        )

        augmentations = A.Compose(
            [
                A.RandomResizedCrop(size=resize, scale=(0.7, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                normalization if normalize else A.NoOp(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        self.transforms = transforms or Compose(
            [
                ToImage(),
                ToDtype(torch.float32, scale=True),
                ResizeWithBoundingBox(size=resize),
                AlbumentationsTransform(aug=augmentations),
            ]
        )

        self.detection_dataset = self.to_detection_dataset()

    @classmethod
    def get_coco_split_path(cls, dataset_path: Path, dataset_split: DatasetSplit) -> Path:
        return dataset_path / cls.COCO_SPLIT_MAPPING[dataset_split]

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

        # using same degenerate box validation as torchvision FCOS
        valid_indices = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)

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
