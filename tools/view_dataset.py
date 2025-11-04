import atexit
from pathlib import Path

import click
import cv2
import supervision as sv
import torch

from det.datasets.coco import CocoDataset

atexit.register(cv2.destroyAllWindows)


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def main(dataset_path: Path):
    dataset_path = Path(dataset_path)
    coco_dataset = CocoDataset(dataset_path)
    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
    box_annotator = sv.BoxAnnotator(thickness=2)
    for detection_bundle in coco_dataset:
        image = (detection_bundle["image"] * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        xyxy = detection_bundle["boxes"].numpy()
        labels = detection_bundle["labels"].numpy()

        annotated_image = box_annotator.annotate(scene=image, detections=sv.Detections(xyxy=xyxy, class_id=labels))

        cv2.imshow("Dataset Viewer", annotated_image)
        key = cv2.waitKey(0)
        if key == ord("n"):
            continue
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
