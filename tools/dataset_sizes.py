import atexit
from pathlib import Path

import click
import cv2
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track

from det.datasets.coco import CocoDataset

atexit.register(cv2.destroyAllWindows)


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def main(dataset_path: Path):
    dataset_path = Path(dataset_path)
    coco_dataset = CocoDataset(dataset_path)
    sizes_wh = []
    for detection_bundle in track(iter(coco_dataset), description="Processing images..."):
        xyxy = detection_bundle["boxes"].numpy()
        xy1 = xyxy[:, :2]
        xy2 = xyxy[:, 2:]
        sizes_wh.append(xy2 - xy1)

    _fig, ax = plt.subplots(1, 2)

    sizes_wh = np.concatenate(sizes_wh, axis=0)

    ax[0].hist([size[0] for size in sizes_wh], bins=200, color="blue", alpha=0.7)

    ax[0].set_xlim(0, np.percentile(sizes_wh[:, 0], 90))

    ax[0].set_title("Width Distribution")

    ax[1].hist([size[1] for size in sizes_wh], bins=200, color="green", alpha=0.7)

    ax[1].set_xlim(0, np.percentile(sizes_wh[:, 1], 90))

    ax[1].set_title("Height Distribution")

    plt.show()


if __name__ == "__main__":
    main()
