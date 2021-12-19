import json
import math
import os
import pickle
import random

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset

from datasets.transforms import MelSpectrogram, STFT


class VQLatent(Dataset):

    PAD = -100  # <pad> token
    BOS = 0  # <bos> token
    OFFSET = 1  # number of special tokens to offset original vocabulary by

    def __init__(self, config: DictConfig, split: str):
        super().__init__()

        self.dataset_path = config.dataset.dataset_path
        self.pkl_files = list(os.listdir(os.path.join(config.dataset.dataset_path, split)))
        with open(os.path.join(config.dataset_path, "metadata.json"), "w", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.segment_length = config.dataset.segment_length

        assert config.model.vocab_size == self.metadata["vocab_size"], \
            "Need to specify correct model vocab size for this dataset"

        self.use_token = config.dataset.use_token
        self.use_spect = config.dataset.use_spect
        self.use_audio = config.dataset.use_audio

        self.transforms = {
            "stft":
                STFT(
                    n_fft=config.dataset.n_fft,
                    hop_length=config.dataset.hop_length,
                    win_length=config.dataset.win_length,
                    window="hann",
                ),
            "spect":
                MelSpectrogram(
                    sample_rate=config.dataset.sample_rate,
                    n_fft=config.dataset.n_fft,
                    win_length=config.dataset.win_length,
                    hop_length=config.dataset.hop_length,
                    n_mels=config.dataset.n_mels,
                    f_min=0.0,
                    f_max=8000.,
                ),
        }

    def __getitem__(self, index):
        # Load pickle
        pkl_file = self.pkl_files[index]
        with open(os.path.join(self.dataset_path, pkl_file), "rb") as f:
            pkl = pickle.load(f)
            audio = pkl["x"]
            token = pkl["q"]
            speaker = torch.tensor((pkl["speaker"],), dtype=torch.long) if "speaker" in pkl else None

        # Truncate to segment length
        if self.segment_length > 0:
            if token.shape[-1] > self.segment_length:
                # Trim tokens
                random_start = random.randint(0, token.shape[-1] - self.segment_length)
                token = token[random_start:random_start + self.segment_length]
                # Trim audio (longer by a factor of compression_factor)
                random_start = random_start * self.metadata["compression_factor"]
                audio = audio[random_start:random_start + self.segment_length * self.metadata["compression_factor"]]

        # Prepend <BOS> and append <EOS> tokens and tensorize
        token = [VQLatent.BOS] + token
        token = torch.tensor(token, dtype=torch.long)
        audio = torch.tensor(audio, dtype=torch.float32)
        token_len = token.shape[-1]
        audio_len = audio.shape[-1]

        if self.segment_length > 0:
            # Pad to segment_length in case we have shorter examples
            token = F.pad(token, (0, self.segment_length + 2 - len(token)), mode="constant", value=VQLatent.PAD)
            audio = F.pad(audio, (0, self.segment_length * self.metadata["compression_factor"] - len(audio)))

        if self.use_spect:
            spect = self.transforms["spect"](audio)
            spect_len = spect.shape[-1]
        else:
            spect = spect_len = None

        if not self.use_audio:
            audio = audio_len = None

        if not self.use_token:
            token = token_len = None

        return token + VQLatent.OFFSET, token_len, spect, spect_len, audio, audio_len, speaker

    def __len__(self):
        return len(self.pickles)

    @staticmethod
    def collate(batch):
        """None entries indicate elements that aren't needed by the config and are returned as None"""
        token, token_len, spect, spect_len, audio, audio_len, speaker = zip(*batch)

        token_len = torch.tensor(token_len, dtype=torch.long) if token_len[0] is not None else None
        token = torch.stack(
            [F.pad(x, (0, token_len.max() - x.shape[-1]), mode="constant", value=VQLatent.PAD) for x in token],
            dim=0,
        ) if token[0] is not None else None

        spect_len = torch.tensor(spect_len, dtype=torch.long) if spect_len[0] is not None else None
        spect = torch.stack(
            [F.pad(x, (0, spect_len.max() - x.shape[-1]), value=math.log(1e-7)) for x in spect],
            dim=0,
        ) if spect[0] is not None else None

        audio_len = torch.tensor(audio_len, dtype=torch.long) if audio_len[0] is not None else None
        audio = torch.stack(
            [F.pad(x, (0, audio_len.max() - x.shape[-1])) for x in audio],
            dim=0,
        ).unsqueeze(1) if audio[0] is not None else None

        speaker = torch.stack(speaker, dim=0) if speaker[0] is not None else None

        return token, token_len, spect, spect_len, audio, audio_len, speaker
