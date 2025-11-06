from __future__ import annotations

from typing import TYPE_CHECKING

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from det.datasets.coco import CocoDataset

if TYPE_CHECKING:
    from pathlib import Path


class CocoDataModule(LightningDataModule):
    def __init__(self, dataset_path: Path, batch_size: int, num_workers: int, resize: tuple[int, int] | None = None):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.train_dataset: CocoDataset | None = None
        self.val_dataset: CocoDataset | None = None
        self.class_names: list[str] = []
        self.num_classes: int = 0

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = CocoDataset(self.dataset_path, split="train", resize=self.resize)
            self.val_dataset = CocoDataset(self.dataset_path, split="val", resize=self.resize)
            self.class_names = self.train_dataset.class_names
            self.num_classes = len(self.class_names)

    def train_dataloader(self):
        if not self.train_dataset:
            msg = "Train dataset not initialized. Call setup() first."
            raise RuntimeError(msg)
        return DataLoader[CocoDataset](
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,  # Assuming your dataset has a collate_fn
        )

    def val_dataloader(self):
        if not self.val_dataset:
            msg = "Validation dataset not initialized. Call setup() first."
            raise RuntimeError(msg)
        return DataLoader[CocoDataset](
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,  # Assuming your dataset has a collate_fn
        )
