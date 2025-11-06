#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import cv2
import pytorch_lightning as pl
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner import Tuner

from det.datasets.datamodule import CocoDataModule
from det.model.fcos_detector_module import FCOSDetector

cv2.setNumThreads(0)

# Assuming these imports are in your project structure

DEBUG = os.environ.get("DEBUG", "0") == "1" or "pdb" in sys.argv


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
@click.option("--num-workers", type=int, default=min(8, os.cpu_count() or 1), help="Number of DataLoader workers.")
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
@click.option(
    "-lr",
    "--learning-rate",
    type=float,
    default=1e-5,
    help="Learning rate for the optimizer.",
)
@click.option(
    "--tune-batch-size",
    is_flag=True,
    default=False,
    help="Automatically tune the batch size before training.",
)
@click.option(
    "--tune-lr",
    is_flag=True,
    default=False,
    help="Automatically tune the learning rate before training.",
)
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help="Enable PyTorch Profiler for TensorBoard performance analysis.",
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
    tune_batch_size: bool,
    tune_lr: bool,
    learning_rate: float,
    profile: bool,
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
        learning_rate=learning_rate,
    )

    # 3. Initialize Callbacks
    progress_bar = RichProgressBar()

    # 4. Setup Profiler if enabled
    profiler = None
    if profile:
        profiler_dir = work_dir / "profiler"
        profiler_dir.mkdir(parents=True, exist_ok=True)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        profiler = PyTorchProfiler(
            dirpath=str(profiler_dir),
            filename="profile",
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            export_to_chrome=True,
        )
        print(f"Profiling enabled. Traces will be saved to: {profiler_dir}")
        print(f"View in TensorBoard: tensorboard --logdir {profiler_dir}")

    # 5. Initialize Trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=device,
        log_every_n_steps=10,
        callbacks=[
            progress_bar,
            RichModelSummary(max_depth=3),
            LearningRateMonitor(),
        ],
        logger=TensorBoardLogger(save_dir=default_root_dir, name=log_dir),
        profiler=profiler,
        deterministic=True,  # For reproducibility
        enable_checkpointing=True,  # Saves checkpoints by default
        default_root_dir=default_root_dir,
    )

    tuner = Tuner(trainer)
    if tune_batch_size:
        tuner.scale_batch_size(model, datamodule=data_module, mode="binsearch", init_val=batch_size, max_trials=5)
    if trainer.train_dataloader:
        print(f"Using batch size: {trainer.train_dataloader.batch_size}")

    if tune_lr:
        tuner.lr_find(model, datamodule=data_module, max_lr=0.1)

    # 6. Start Training
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_from_checkpoint)

    if profile:
        print("\nProfiling complete! View traces in TensorBoard:")
        print(f"  tensorboard --logdir {work_dir / 'profiler'}")

    print("Training finished.")


if __name__ == "__main__":
    main()
