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
@click.option("-s", "--split", type=click.Choice(["train", "val", "test"]), default="val")
def main(dataset_path: Path, split: str) -> None:
    dataset_path = Path(dataset_path)
    coco_dataset = CocoDataset(dataset_path, split=split, normalize=False, resize=(360, 640))
    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()
    for detection_bundle in coco_dataset:
        image = image_tensor_to_numpy(detection_bundle.image)
        if image.dtype != "uint8":
            image = image_scale_to_uint8_numpy(image)
        xyxy = detection_bundle.boxes.numpy()
        labels = detection_bundle.labels.numpy()

        annotated_image = box_annotator.annotate(scene=image, detections=sv.Detections(xyxy=xyxy, class_id=labels))
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=sv.Detections(xyxy=xyxy, class_id=labels),
            labels=[coco_dataset.class_names[cid] for cid in labels],
        )

        cv2.imshow("Dataset Viewer", annotated_image)
        key = cv2.waitKey(0)
        if key == ord("n"):
            continue
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
