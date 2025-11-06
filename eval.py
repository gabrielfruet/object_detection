#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import click
import pytorch_lightning as pl
import rich
import supervision as sv
import torch
from supervision.metrics.core import MetricTarget
from supervision.metrics.mean_average_precision import MeanAveragePrecision
from torch.utils.data import DataLoader

from det.datasets.coco import CocoDataset
from det.model.fcos_detector_module import (
    FCOSDetector,
    batched_detection_bundle_to_sv_detection,
    sv_detection_from_dict,
)

pl.seed_everything(42, workers=True)


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--checkpoint", type=click.Path(exists=True, file_okay=True, path_type=Path), required=True)
@click.option("--split", type=click.Choice(["train", "test", "val"]), default="test")
@click.option("--batch-size", type=int, default=4)
@click.option("--num-workers", type=int, default=2)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--visualize", is_flag=True, help="Visualize predictions vs ground truth")
@click.option("--score-thresh", type=float, default=0.5, help="Score threshold for predictions")
def main(
    dataset_path: Path,
    checkpoint: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    visualize: bool,
    score_thresh: float,
    split: str,
):
    """Evaluate a trained FCOS model on the test set."""
    rich.print(f"[bold green]Loading model from checkpoint:[/bold green] {checkpoint}")
    # Load dataset
    dataset = CocoDataset(dataset_path, split=split)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )

    # Load model from checkpoint (class must match your training script)
    model = FCOSDetector.load_from_checkpoint(
        checkpoint_path=checkpoint, map_location=device, score_thresh=score_thresh
    )
    model = model.to(device)
    model.eval()

    # Metric
    metric = MeanAveragePrecision(metric_target=MetricTarget.BOXES)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.RichLabelAnnotator(font_size=16)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = [img.to(device) for img in batch.image]
            [
                {"boxes": box.to(device), "labels": label.to(device)}
                for box, label in zip(batch.boxes, batch.labels)
            ]

            preds = model.model(images)

            preds_detections = [sv_detection_from_dict(pred) for pred in preds]
            gt_detections = batched_detection_bundle_to_sv_detection(batch)

            metric.update(preds_detections, gt_detections)

            if visualize:
                import cv2
                import numpy as np

                for img_tensor, pred_det, gt_det in zip(batch.image, preds_detections, gt_detections):
                    img = img_tensor.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype("uint8")

                    pred_frame = box_annotator.annotate(scene=img.copy(), detections=pred_det)
                    pred_frame = label_annotator.annotate(
                        scene=pred_frame,
                        detections=pred_det,
                        labels=[model.class_names[cid] for cid in pred_det.class_id],
                    )

                    gt_frame = box_annotator.annotate(scene=img.copy(), detections=gt_det)
                    gt_frame = label_annotator.annotate(
                        scene=gt_frame,
                        detections=gt_det,
                        labels=[model.class_names[cid] for cid in gt_det.class_id],
                    )

                    cv2.putText(
                        pred_frame, "Prediction", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        gt_frame, "Ground Truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                    )

                    combined = np.hstack((pred_frame, gt_frame))
                    cv2.imshow("Prediction (Left) vs GT (Right)", combined)
                    key = cv2.waitKey(0)
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        return

    # Compute final metrics
    results = metric.compute()
    rich.print("\n[bold blue]Evaluation Results:[/bold blue]")
    rich.print(f"mAP@0.5: {results.map50:.4f}")
    rich.print(f"mAP@0.5:0.95: {results.map50_95:.4f}")
    rich.print("Per-class AP:")
    for class_name, ap in zip(model.class_names, results.ap_per_class):
        rich.print(f"  {class_name:20s}: {ap.mean():.4f}")

    rich.print("[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
