#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pytorch_lightning as pl
import rich
import supervision as sv
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from supervision.metrics.core import MetricTarget
from supervision.metrics.mean_average_precision import MeanAveragePrecision
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import FCOS
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.rpn import AnchorGenerator

# Assuming these imports are in your project structure
from det.datasets.coco import CocoDataset

if TYPE_CHECKING:
    from det.types.detection import BatchedDetectionBundle

DEBUG = os.environ.get("DEBUG", "0") == "1" or "pdb" in sys.argv
print(f"{DEBUG=}")


# --- Utility Functions (Unchanged) ---


def sv_detection_from_dict(data: dict) -> sv.Detections:
    xyxy = data["boxes"].cpu().numpy()
    scores = data.get("scores")

    if scores is not None:
        scores = scores.cpu().numpy()

    labels = data.get("labels")

    if labels is not None:
        labels = labels.cpu().numpy()

    return sv.Detections(xyxy=xyxy, confidence=scores, class_id=labels)


def batched_detection_bundle_to_sv_detection(batch: BatchedDetectionBundle) -> list[sv.Detections]:
    detections_list = []
    for i in range(len(batch["image"])):
        xyxy = batch["boxes"][i].cpu().numpy()
        labels = batch["labels"][i].cpu().numpy()
        detections = sv.Detections(xyxy=xyxy, class_id=labels)
        detections_list.append(detections)
    return detections_list


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


class FCOSDetector(LightningModule):
    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        epochs: int,
        learning_rate: float = 1e-3,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.3,
        train_eval_subset: int = 200,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        self.train_eval_subset = train_eval_subset

        # Store class names for visualization
        self.class_names = class_names

        # Define model components
        backbone = mobilenet_backbone(
            backbone_name="mobilenet_v3_large",
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            fpn=True,
        )
        # backbone = resnet_fpn_backbone(
        #     backbone_name="resnet50", weights=ResNet50_Weights.IMAGENET1K_V2, returned_layers=[1, 2, 4]
        # )
        # anchor_sizes = (
        #     (8.0),
        #     (16.0,),
        #     (32.0,),
        #     (128.0),
        # )
        anchor_sizes = (
            (64.0,),
            (128.0,),
            (256.0,),
        )
        aspect_ratios = ((1.0,),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        # Initialize the model
        self.model = FCOS(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
        )

        # Metric: Will be initialized in on_validation_epoch_start
        self.map_metric: MeanAveragePrecision | None = None

        # Annotators for visualization
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.3, text_padding=2, text_thickness=1)
        for param in self.model.parameters():
            param.requires_grad = True

    def on_validation_epoch_start(self):
        """Initialize the metric at the start of each validation epoch."""
        self.map_metric = MeanAveragePrecision(metric_target=MetricTarget.BOXES)

    def training_step(self, batch: BatchedDetectionBundle, batch_idx: int):
        images = batch["image"]
        targets = [
            {"boxes": box.to(self.device), "labels": label.to(self.device)}
            for box, label in zip(batch["boxes"], batch["labels"])
        ]

        # FCOS returns losses when in training mode
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())

        # Log losses
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, logger=True)

        return loss

    def validation_step(self, batch: BatchedDetectionBundle, batch_idx: int):
        images = batch["image"]
        targets = [
            {"boxes": box.to(self.device), "labels": label.to(self.device)}
            for box, label in zip(batch["boxes"], batch["labels"])
        ]

        # FCOS returns predictions when in eval mode
        predictions: list[dict] = self.model(images, targets)

        # Convert to supervision Detections
        pred_detections = [sv_detection_from_dict(pred) for pred in predictions]
        gt_detections = batched_detection_bundle_to_sv_detection(batch)

        # Update metric
        if self.map_metric:
            self.map_metric.update(pred_detections, gt_detections)

    def on_validation_epoch_end(self):
        """Compute and log the final metric."""
        if self.map_metric:
            metrics = self.map_metric.compute()
            self.log("val_mAP50", metrics.map50, prog_bar=True, logger=True)
            rich.print(f"\n[bold blue]Epoch {self.current_epoch} mAP@0.5: {metrics.map50:.4f}[/bold blue]")

    def on_train_epoch_end(self):
        """Compute training mAP on a small subset after each epoch."""
        if not hasattr(self.trainer, "train_dataloader") or self.trainer.train_dataloader is None:
            return

        dataloader = self.trainer.train_dataloader
        self.model.eval()

        metric = MeanAveragePrecision(metric_target=MetricTarget.BOXES)
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                predictions = self.model(images)
                pred_det = [sv_detection_from_dict(p) for p in predictions]
                gt_det = batched_detection_bundle_to_sv_detection(batch)
                metric.update(pred_det, gt_det)

                count += len(images)
                if count >= self.train_eval_subset:
                    break

        results = metric.compute()
        self.log("train_mAP50", results.map50, prog_bar=True)
        rich.print(f"[green]Train mAP@0.5 (subset): {results.map50:.4f}[/green]")
        self.model.train()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.hparams.epochs)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
            },
        }


# --- Main execution ---


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--epochs", type=int, default=10, help="Number of training epochs.")
@click.option("--batch-size", type=int, default=4, help="Batch size for training.")
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use for training (becomes 'accelerator' in Lightning).",
)
@click.option("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
@click.option("--visualize", is_flag=True, default=False, help="Show predictions on last epoch.")
@click.option(
    "--resume-from-checkpoint",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a checkpoint file to resume training from.",
)
@click.option(
    "--work-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
def main(
    dataset_path: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
    visualize: bool,
    resume_from_checkpoint: Path | None,
    work_dir: Path | None,
):
    print("Training started with PyTorch Lightning...")

    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        if resume_from_checkpoint.suffix != ".ckpt":
            msg = "Checkpoint file must have a .ckpt extension."
            raise ValueError(msg)
    else:
        print("Starting new training session...")

    work_dir = work_dir or Path.cwd() / "work_dir" / "fcos_detector"
    work_dir.mkdir(parents=True, exist_ok=True)

    log_dir = work_dir.name
    default_root_dir = work_dir.parent

    # Set seed
    pl.seed_everything(42, workers=True)

    # 1. Initialize DataModule
    data_module = CocoDataModule(
        dataset_path=dataset_path, batch_size=batch_size, num_workers=num_workers, resize=(360, 640)
    )
    # Run setup manually to get num_classes and class_names
    data_module.setup("fit")
    print(f"Found {data_module.num_classes} classes: {data_module.class_names}")

    # 2. Initialize LightningModule
    model = FCOSDetector(
        num_classes=data_module.num_classes,
        class_names=data_module.class_names,
        epochs=epochs,
    )

    # 3. Initialize Callbacks
    progress_bar = RichProgressBar()

    # 4. Initialize Trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=device,
        log_every_n_steps=1,
        callbacks=[
            progress_bar,
            RichModelSummary(max_depth=3),
        ],
        logger=TensorBoardLogger(save_dir=default_root_dir, name=log_dir),
        deterministic=True,  # For reproducibility
        enable_checkpointing=True,  # Saves checkpoints by default
        default_root_dir=default_root_dir,
    )

    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=data_module, mode="binsearch", init_val=batch_size, max_trials=5)
    if trainer.train_dataloader:
        print(f"Using batch size: {trainer.train_dataloader.batch_size}")

    tuner.lr_find(model, datamodule=data_module, max_lr=0.1)

    # 5. Start Training
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_from_checkpoint)

    print("Training finished.")


if __name__ == "__main__":
    main()
