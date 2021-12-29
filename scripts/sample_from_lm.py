"""
Script to sample from token language model

Sample usage:
python -m scripts.sample_from_lm \
    --log_dir ./logs/transformer_lm \
    --ckpt_num 3000 \
    --dump_dir ./outputs \
    --n_samples 4 \
    --n_steps 1024
"""

import argparse
import logging
import logging.config
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
from omegaconf import OmegaConf
from PIL import Image
from tabulate import tabulate

from utils.commons import get_model

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory of training")
    parser.add_argument("--ckpt_num", type=int, required=True, help="Checkpoint number to load")
    parser.add_argument("--dump_dir", type=str, required=False, default="./outputs", help="Directory to dump VQ dataset")

    parser.add_argument("--n_samples", type=int, required=False, default=4, help="Batch size for inference")
    parser.add_argument("--n_steps", type=int, required=False, default=1024, help="Batch size for inference")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    config.train.n_gpus = 1 if device == "cuda" else 0
    logger.info("Loaded config")

    # Load model
    ckpt = torch.load(os.path.join(args.log_dir, "ckpts", f"ckpt.{args.ckpt_num}.pt"), map_location=device)
    model, _ = get_model(config, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info("Loaded checkpoint")

    args.dump_dir = os.path.join(args.dump_dir, f"{model.__class__.__name__}@{args.ckpt_num}")
    os.makedirs(args.dump_dir, exist_ok=True)

    # Sample
    x_samples, q_samples = model.sample(batch_size=args.n_samples, n_steps=args.n_steps, device=device)
    logger.info("Generated token samples")

    # Save audio and spect
    spects = []
    for i in range(args.n_samples):
        audio = x_samples[i].cpu().numpy()
        soundfile.write(
            os.path.join(args.dump_dir, f"sample_{i}.wav"),
            audio,
            config.dataset.sample_rate,
        )

        spect = librosa.feature.melspectrogram(
            audio,
            sr=config.dataset.sample_rate,
            n_fft=config.dataset.n_fft,
            hop_length=config.dataset.hop_length,
            win_length=config.dataset.win_length,
            window="hann",
            pad_mode="constant",
        )
        spect = librosa.power_to_db(spect)
        spects += [spect]

    fig, axes = plt.subplots(args.n_samples, 1, figsize=(16 * args.n_samples, 4 * args.n_samples))
    for i in range(args.n_samples):
        ax = axes[i]
        im = ax.imshow(spects[i], aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("sample spectrograms")
        ax.set_ylabel(str(i))
    fig.tight_layout()
    fig.canvas.draw()
    grid = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    grid = grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    Image.fromarray(grid).save(os.path.join(args.dump_dir, "mel_spectrograms.png"))
    logger.info("Saved audio and spectrograms")

    # Save token
    with open(os.path.join(args.dump_dir, "tokens.txt"), "w") as f:
        f.write(tabulate(q_samples.tolist(), headers=range(args.n_steps)))

    logger.info("Done")


if __name__ == "__main__":
    main()
