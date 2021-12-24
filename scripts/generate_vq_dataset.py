"""
Script to generate datasets from VQVAE latents

Sample usage:
python -m scripts.generate_vq_dataset \
    --log_dir ./logs/vqvae \
    --ckpt_num 10000 \
    --dump_dir ./data/VQ-Latent \
    --batch_size 32 \
    --n_processes 32
"""

import argparse
import json
import logging
import logging.config
import multiprocessing
import os
import pickle
import random
from collections import Counter

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import models.glow_tts.submodules as submodules
from models.vqvae.vqvae import VQVAE
from utils.commons import get_dataloaders, to_device
from utils.train_utils import spects_to_grid

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory of training")
    parser.add_argument("--ckpt_num", type=int, required=True, help="Checkpoint number to load")
    parser.add_argument(
        "--dump_dir", type=str, required=False, default="./data/VQ-Latent", help="Directory to dump VQ dataset"
    )

    parser.add_argument("--batch_size", type=int, required=False, default=128, help="Batch size for inference")
    parser.add_argument(
        "--n_processes", type=int, required=False, default=4, help="Number of processes to save pickle files with"
    )
    return parser.parse_args()


class ConvenientVQVAE(VQVAE):

    @torch.no_grad()
    def encode_and_quantize(self, x, x_lengths):
        x_mask = torch.unsqueeze(submodules.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        # Encode
        q, q_mask = self.encoders[VQVAE.LEVEL](x, x_mask)

        # Quantize
        q = self.bottleneck.level_blocks[VQVAE.LEVEL].encode(q, q_mask)
        return {"x": x.cpu(), "q": q.cpu(), "l": q_mask.sum(-1).long().cpu()}

    @torch.no_grad()
    def dequantize_and_decode(self, q, q_lengths):
        # Dequantize
        x = self.bottleneck.level_blocks[VQVAE.LEVEL].decode(q)

        # Decode
        x_mask = torch.unsqueeze(submodules.sequence_mask(q_lengths, x.size(2)), 1).to(x.dtype)
        x, _ = self.decoders[VQVAE.LEVEL]([x], [x_mask], all_levels=False)
        return {"xh": x.cpu() * x_mask}


def dump_batch_to_pickle(index: int, x: torch.FloatTensor, q: torch.LongTensor, l: torch.LongTensor, dump_dir: str):
    x = x.tolist()
    q = q[:l].tolist()
    with open(os.path.join(dump_dir, f"{index:05d}.pkl"), "wb") as f:
        pickle.dump({"x": x, "q": q}, f)
    return Counter(q)


def generate_and_dump_dataset(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    pool: multiprocessing.Pool,
    dump_dir: str,
    split: str,
    device: str = "cuda",
):
    os.makedirs(os.path.join(dump_dir, split), exist_ok=True)
    batch_size = dataloader.batch_size
    dataset_counter = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Generating {split} dataset"):
        batch = to_device(batch, device)
        _, _, _, _, x, x_lengths, _ = batch
        out_dict = model.encode_and_quantize(x, x_lengths)

        # Dump batch and accumulate token counters for each example
        batch_counter = pool.starmap(
            dump_batch_to_pickle,
            zip(
                range(i * batch_size, i * batch_size + len(batch[0])),
                out_dict["x"],
                out_dict["q"],
                out_dict["l"],
                [os.path.join(dump_dir, split)] * len(batch[0]),
            ),
        )
        # Aggregate token counters per batch
        dataset_counter += [sum(batch_counter, Counter())]

    # Aggregate token counters across batches
    dataset_counter = sum(dataset_counter, Counter())

    # Plot histogram of learned tokens
    keys = list(sorted(dataset_counter.keys()))
    values = [dataset_counter[k] for k in keys]
    plt.bar(keys, values)
    plt.savefig(os.path.join(dump_dir, f"{split}_histogram.png"))
    plt.clf()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    config.train.n_gpus = 1 if device == "cuda" else 0
    config.train.batch_size = args.batch_size
    config.dataset.segment_length = -1
    logger.info("Loaded config")

    # Init model and load checkpoint
    ckpt = torch.load(os.path.join(args.log_dir, "ckpts", f"ckpt.{args.ckpt_num}.pt"), map_location=device)
    model = ConvenientVQVAE(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info("Loaded checkpoint")

    # Init dataloaders
    train_dataloader, val_dataloader = get_dataloaders(config)
    train_dataloader.shuffle = False
    logger.info("Loaded dataloaders")

    # Run inference and dump Q-datasets
    with multiprocessing.Pool(processes=args.n_processes) as pool:
        generate_and_dump_dataset(
            dataloader=train_dataloader,
            model=model,
            pool=pool,
            dump_dir=args.dump_dir,
            split="train",
            device=device,
        )
        generate_and_dump_dataset(
            dataloader=val_dataloader,
            model=model,
            pool=pool,
            dump_dir=args.dump_dir,
            split="val",
            device=device,
        )
    logger.info("Finished generating datasets")

    # Run sanity check, save audio and spect
    sanity = random.sample(os.listdir(os.path.join(args.dump_dir, "train")), 1)[0]
    with open(os.path.join(args.dump_dir, "train", sanity), "rb") as f:
        data = pickle.load(f)
        q = torch.tensor(data["q"], dtype=torch.long, device=device).unsqueeze(0)
        q_lengths = torch.tensor((q.shape[-1],), dtype=torch.long, device=device)
        x = np.array(data["x"], dtype=np.float32)

    xh = model.dequantize_and_decode(q, q_lengths)["xh"].flatten().numpy()
    soundfile.write(os.path.join(args.dump_dir, "sanity.wav"), xh, config.dataset.sample_rate)
    x = x[:len(xh)]
    sh = librosa.feature.melspectrogram(
        xh,
        sr=config.dataset.sample_rate,
        n_fft=config.dataset.n_fft,
        hop_length=config.dataset.hop_length,
        win_length=config.dataset.win_length,
        window="hann",
        pad_mode="constant",
    )
    sh = librosa.power_to_db(sh)
    s = librosa.feature.melspectrogram(
        x,
        sr=config.dataset.sample_rate,
        n_fft=config.dataset.n_fft,
        hop_length=config.dataset.hop_length,
        win_length=config.dataset.win_length,
        window="hann",
        pad_mode="constant",
    )
    s = librosa.power_to_db(s)
    grid = spects_to_grid(sh[None, ...], sh[None, ...], n=1)
    Image.fromarray(grid).save(os.path.join(args.dump_dir, "sanity.png"))
    logger.info("Finished sanity check")

    # Save metadata
    metadata = {}
    metadata["compression_factor"] = int(np.prod(np.array(config.model.strides_t)**np.array(config.model.downs_t)))
    metadata["vocab_size"] = config.model.l_bins
    with open(os.path.join(args.dump_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    logger.info("Saved metadata")

    logger.info("Done")


if __name__ == "__main__":
    main()
