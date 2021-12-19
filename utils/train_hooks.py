"""Implements hook functions used in utils/train_utils.py"""

import logging
import logging.config
import os
from typing import Dict

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn as nn
from omegaconf import DictConfig
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_ddi(config, model, ema, optimizer, scheduler, train_dataloader, device):
    """Runs DDI and saves this as checkpoint 0"""
    from utils.commons import get_model, to_device

    if not hasattr(model, "ddi") or not callable(model.ddi):
        logging.warning("Skipping DDI, not supported by model.")
    else:
        logging.info("Running DDI ...")
        batch = None
        for batch in train_dataloader:
            batch = to_device(batch, device)
            break
        if batch is None:
            raise RuntimeError("`train_dataloader` exited without returning a batch")
        model.ddi(batch)

    # Update starting point of EMA
    _, ema = get_model(config, device, rank=0)
    config.train.load_ckpt = save_checkpoint(config, 0, 0, model, ema, optimizer, scheduler)
    logger.info("Finished DDI, checkpointed as ckpt.0.pt")


def accumulate_stats(
    config: DictConfig,
    loss_dict: Dict[str, torch.Tensor],
    metrics_dict: Dict[str, torch.Tensor],
    accumulated_loss: Dict[str, torch.Tensor],
    accumulated_metrics: Dict[str, torch.Tensor],
):
    """Accumulates loss into `accumulated_loss` and metrics into `accumulated_metrics` in-place"""
    for key in loss_dict.keys():
        if "loss" in key:
            accumulated_loss[key] += loss_dict[key].cpu().item() / config.train.log_every_n_steps
    for key in metrics_dict.keys():
        accumulated_metrics[key] += metrics_dict[key].cpu().item() / config.train.log_every_n_steps


def log_stats(
    step_or_epoch: int,
    writer: SummaryWriter,
    losses: Dict[str, torch.Tensor],
    metrics: Dict[str, torch.Tensor],
):
    """Logs loss and metrics into tensorboard"""
    for key in losses.keys():
        writer.add_scalar(f"loss/train_{key}", losses[key], step_or_epoch)
    for key in metrics.keys():
        writer.add_scalar(f"metrics/train_{key}", metrics[key], step_or_epoch)


def save_checkpoint(
    config: DictConfig,
    global_step: int,
    epoch: int,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> None:
    """Saves checkpoint. Pass in epoch=-1 to save the last checkpoint"""
    ckpt_path = os.path.join(config.train.log_dir, "ckpts", f"ckpt.{'last' if epoch == -1 else global_step}.pt")
    torch.save(
        {
            "config": config,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "step": global_step,
            "epoch": config.train.total_epochs if epoch == -1 else epoch,
        },
        ckpt_path,
    )
    return ckpt_path


def spects_to_grid(ys: np.ndarray, yhs: np.ndarray, n: int = 4) -> np.ndarray:
    fig, axes = plt.subplots(n, 2, figsize=(64, 16))
    for i in range(n * 2):
        j = i % 2
        i //= 2
        ax = axes[i, j]
        spect = ys[i] if j == 0 else yhs[i]
        im = ax.imshow(spect, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=ax)
        if j == 0:
            ax.set_xlabel("ground truth")
        else:
            ax.set_xlabel("predicted")
        ax.set_ylabel(str(i))

    fig.tight_layout()
    fig.canvas.draw()
    grid = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    grid = grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return grid


def save_spect_and_inverted_audio(
    config: DictConfig,
    global_step: int,
    writer: SummaryWriter,
    spect: np.ndarray,
    spect_pred: np.ndarray,
    n: int = 4,
) -> None:
    # Save spectrogram as grid
    grid = spects_to_grid(spect, spect_pred, n=n)
    Image.fromarray(grid).save(os.path.join(config.train.log_dir, "spect", f"val_spect_{global_step}.jpg"))
    writer.add_image("mel/val", grid, global_step, dataformats="HWC")

    # Save audio individually
    for i, (gt, pred) in enumerate(zip(spect, spect_pred)):
        if i == n:
            break
        gt = librosa.feature.inverse.mel_to_audio(
            np.exp(gt),
            sr=config.dataset.sample_rate,
            n_fft=config.dataset.n_fft,
            hop_length=config.dataset.hop_length,
            win_length=config.dataset.win_length,
            window="hann",
            pad_mode="constant",
        )
        pred = librosa.feature.inverse.mel_to_audio(
            np.exp(pred),
            sr=config.dataset.sample_rate,
            n_fft=config.dataset.n_fft,
            hop_length=config.dataset.hop_length,
            win_length=config.dataset.win_length,
            window="hann",
            pad_mode="constant",
        )
        if i == 0:
            soundfile.write(
                os.path.join(config.train.log_dir, "audio", f"val_audio_{global_step}_gt.wav"),
                gt,
                config.dataset.sample_rate,
            )
            writer.add_audio("audio/val_gt", gt, global_step=global_step, sample_rate=config.dataset.sample_rate)
            soundfile.write(
                os.path.join(config.train.log_dir, "audio", f"val_audio_{global_step}_syn.wav"),
                pred,
                config.dataset.sample_rate,
            )
            writer.add_audio("audio/val_pred", pred, global_step=global_step, sample_rate=config.dataset.sample_rate)


def save_audio_and_computed_spect(
    config: DictConfig,
    global_step: int,
    writer: SummaryWriter,
    audio: np.ndarray,
    audio_pred: np.ndarray,
    n: int = 4,
) -> None:
    spect, spect_pred = [], []
    for i, (gt, pred) in enumerate(zip(audio, audio_pred)):
        if i == n:
            break
        gt = np.clip(gt, a_min=-1, a_max=1)
        pred = np.clip(pred, a_min=-1, a_max=1)

        if i == 0:
            soundfile.write(
                os.path.join(config.train.log_dir, "audio", f"val_audio_{global_step}_gt.wav"),
                gt,
                config.dataset.sample_rate,
            )
            writer.add_audio("audio/val_gt", gt, global_step=global_step, sample_rate=config.dataset.sample_rate)
            soundfile.write(
                os.path.join(config.train.log_dir, "audio", f"val_audio_{global_step}_pred.wav"),
                pred,
                config.dataset.sample_rate,
            )
            writer.add_audio("audio/val_pred", pred, global_step=global_step, sample_rate=config.dataset.sample_rate)

        gt = librosa.feature.melspectrogram(
            gt,
            sr=config.dataset.sample_rate,
            n_fft=config.dataset.n_fft,
            hop_length=config.dataset.hop_length,
            win_length=config.dataset.win_length,
            window="hann",
            pad_mode="constant",
        )
        pred = librosa.feature.melspectrogram(
            pred,
            sr=config.dataset.sample_rate,
            n_fft=config.dataset.n_fft,
            hop_length=config.dataset.hop_length,
            win_length=config.dataset.win_length,
            window="hann",
            pad_mode="constant",
        )
        spect += [librosa.power_to_db(gt)]
        spect_pred += [librosa.power_to_db(pred)]

    spect = np.array(spect)
    spect_pred = np.array(spect_pred)
    grid = spects_to_grid(spect, spect_pred, n=n)
    Image.fromarray(grid).save(os.path.join(config.train.log_dir, "spect", f"val_spect_{global_step}.jpg"))
    writer.add_image("mel/val", grid, global_step, dataformats="HWC")
