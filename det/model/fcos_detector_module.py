from __future__ import annotations

from typing import TYPE_CHECKING

import supervision as sv
import torch
from lightning import LightningModule
from supervision.metrics.core import MetricTarget
from supervision.metrics.mean_average_precision import MeanAveragePrecision
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import FCOS
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.rpn import AnchorGenerator

if TYPE_CHECKING:
    from det.types.detection import BatchedDetectionBundle
else:
    from det.types.detection import BoxFormat


def batched_detection_bundle_to_sv_detection(batch: BatchedDetectionBundle) -> list[sv.Detections]:
    detections_list = []
    for i in range(len(batch.image)):
        xyxy = batch.boxes[i].cpu().numpy()
        labels = batch.labels[i].cpu().numpy()
        detections = sv.Detections(xyxy=xyxy, class_id=labels)
        detections_list.append(detections)
    return detections_list


def sv_detection_from_dict(data: dict) -> sv.Detections:
    xyxy = data["boxes"].cpu().numpy()
    scores = data.get("scores")

    if scores is not None:
        scores = scores.cpu().numpy()

    labels = data.get("labels")

    if labels is not None:
        labels = labels.cpu().numpy()

    return sv.Detections(xyxy=xyxy, confidence=scores, class_id=labels)


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
        self.validation_predictions: list[list[dict]] = []
        self.validation_ground_truth: list[BatchedDetectionBundle] = []

    def training_step(self, batch: BatchedDetectionBundle, batch_idx: int):
        # FCOS only supports AABB format
        if batch.box_format != BoxFormat.AABB:
            msg = f"FCOSDetector only supports AABB format, got {batch.box_format}"
            raise ValueError(msg)

        images = batch.image
        targets = [
            {"boxes": box.to(self.device), "labels": label.to(self.device)}
            for box, label in zip(batch.boxes, batch.labels)
        ]

        # FCOS returns losses when in training mode
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())

        # Log losses
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch: BatchedDetectionBundle, batch_idx: int):
        # FCOS only supports AABB format
        if batch.box_format != BoxFormat.AABB:
            msg = f"FCOSDetector only supports AABB format, got {batch.box_format}"
            raise ValueError(msg)

        images = batch.image
        targets = [
            {"boxes": box.to(self.device), "labels": label.to(self.device)}
            for box, label in zip(batch.boxes, batch.labels)
        ]

        predictions: list[dict] = self.model(images, targets)
        self.validation_predictions.append(predictions)
        self.validation_ground_truth.append(batch.to("cpu"))

    def on_validation_epoch_end(self):
        """Compute and log the final metric after gathering all predictions from all processes."""
        if not self.validation_predictions or not self.map_metric:
            return

        # Gather from all processes if distributed
        all_predictions = self._gather_list(self.validation_predictions)
        all_ground_truth = self._gather_list(self.validation_ground_truth)

        # Convert to Detections and update metric
        all_pred_detections = []
        all_gt_detections = []
        for pred_batch, gt_batch in zip(all_predictions, all_ground_truth):
            all_pred_detections.extend([sv_detection_from_dict(p) for p in pred_batch])
            all_gt_detections.extend(batched_detection_bundle_to_sv_detection(gt_batch))

        self.map_metric.update(all_pred_detections, all_gt_detections)
        metrics = self.map_metric.compute()
        self.log("mAP50/val", metrics.map50, prog_bar=True, logger=True, sync_dist=False)

        self.validation_predictions.clear()
        self.validation_ground_truth.clear()

    def _gather_list(self, data: list) -> list:
        """Gather list from all processes if distributed, otherwise return as-is."""
        if self.trainer.num_devices > 1 and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, data)
            result = []
            for sublist in gathered:
                if sublist:
                    result.extend(sublist)
            return result
        return data

    def on_train_epoch_end(self):
        """Compute training mAP on a small subset after each epoch."""
        if not hasattr(self.trainer, "train_dataloader") or self.trainer.train_dataloader is None:
            return

        self.model.eval()
        train_predictions: list[list[dict]] = []
        train_ground_truth: list[BatchedDetectionBundle] = []
        count = 0

        with torch.no_grad():
            for batch in self.trainer.train_dataloader:
                if batch.box_format != BoxFormat.AABB:
                    msg = f"FCOSDetector only supports AABB format, got {batch.box_format}"
                    raise ValueError(msg)

                predictions = self.model(batch.image.to(self.device))
                train_predictions.append(predictions)
                train_ground_truth.append(batch.to("cpu"))

                count += len(batch.image)
                if count >= self.train_eval_subset:
                    break

        # Gather and convert
        all_predictions = self._gather_list(train_predictions)
        all_ground_truth = self._gather_list(train_ground_truth)

        all_pred_detections = []
        all_gt_detections = []
        for pred_batch, gt_batch in zip(all_predictions, all_ground_truth):
            all_pred_detections.extend([sv_detection_from_dict(p) for p in pred_batch])
            all_gt_detections.extend(batched_detection_bundle_to_sv_detection(gt_batch))

        metric = MeanAveragePrecision(metric_target=MetricTarget.BOXES)
        metric.update(all_pred_detections, all_gt_detections)
        self.log("mAP50/train", metric.compute().map50, prog_bar=True, sync_dist=False)
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
