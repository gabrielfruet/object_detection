import atexit
from pathlib import Path

import click
import cv2
import supervision as sv

from det.datasets.coco import CocoDataset
from det.utils.img import image_scale_to_uint8_numpy, image_tensor_to_numpy

atexit.register(cv2.destroyAllWindows)


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def main(dataset_path: Path):
    dataset_path = Path(dataset_path)
    coco_dataset = CocoDataset(dataset_path)
    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
    box_annotator = sv.BoxAnnotator(thickness=2)
    for detection_bundle in coco_dataset:
        image = image_scale_to_uint8_numpy(image_tensor_to_numpy(detection_bundle["image"]))
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
