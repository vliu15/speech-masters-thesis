import math
import os
import random

import librosa
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset

from datasets.transforms import MelSpectrogram, STFT
from models.parser import CMUDictParser

TRUNC_MOD = 512


class LJSpeech(Dataset):

    def __init__(self, config: DictConfig, split: str):
        super().__init__()
        self.root = config.dataset.dataset_path
        self.intersperse_blanks = config.dataset.intersperse_blanks

        if config.dataset.segment_length > 0:
            assert config.dataset.segment_length % TRUNC_MOD == 0, \
                f"config.dataset.segment_length={config.dataset.segment_length} must be a multiple of TRUNC_MOD={TRUNC_MOD}"
        self.segment_length = config.dataset.segment_length
        self.use_token = config.dataset.use_token
        self.use_spect = config.dataset.use_spect
        self.use_audio = config.dataset.use_audio

        # Read in audio paths and transcriptions
        self.audio = []
        self.token = []
        with open(os.path.join(self.root, "metadata.csv"), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                self.audio.append(os.path.join(self.root, "wavs", f"{parts[0]}.wav"))
                self.token.append(parts[2])
        if split == "train":
            self.audio = self.audio[10:]
            self.token = self.token[10:]
        elif split == "val":
            self.audio = self.audio[:10]
            self.token = self.token[:10]
        else:
            raise ValueError(f"LJSpeech not implemented for split {split}")

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
        self.parser = CMUDictParser(config.dataset.cmudict_path)

    def __getitem__(self, index):
        ## Load and truncate audio
        audio, _ = librosa.load(self.audio[index])
        audio = torch.from_numpy(audio)

        if self.segment_length > 0:
            # Truncate to segment_length, which should be modulo TRUNC_MOD
            if audio.shape[-1] > self.segment_length:
                random_start = random.randint(0, audio.shape[-1] - self.segment_length)
                audio = audio[random_start:random_start + self.segment_length]
                audio_len = audio.shape[-1]
            else:
                # In case entire batch of examples is shorter than self.segment_length, pad here
                audio_len = audio.shape[-1]
                audio = F.pad(audio, (0, self.segment_length - len(audio)))
        else:
            # Truncate to modulo compression factor to avoid up/down-sampling length mismatches
            audio = audio[:len(audio) - len(audio) % TRUNC_MOD]
            audio_len = audio.shape[-1]

        # Compute spectrogram
        if self.use_spect:
            spect = self.transforms["spect"](audio).squeeze(0)
            spect_len = spect.shape[-1]
        else:
            spect = spect_len = None

        # Preprocess tokens
        if self.use_token:
            token = self.token[index]
            token = token.strip()
            if token[-1] not in [".", "!", "?"]:
                token = token + "."

            token = self.parser(token)
            if self.intersperse_blanks:
                interspersed = [len(self.parser.symbols)] * (len(token) * 2 + 1)
                interspersed[1::2] = token
                token = interspersed
            token = torch.tensor(token, dtype=torch.long)
            token_len = token.shape[-1]
        else:
            token = token_len = None

        if not self.use_audio:
            audio = audio_len = None

        return token, token_len, spect, spect_len, audio, audio_len, None

    def __len__(self):
        return len(self.audio)

    @staticmethod
    def collate(batch):
        """None entries indicate elements that aren't needed by the config and are returned as None"""
        token, token_len, spect, spect_len, audio, audio_len, _ = zip(*batch)

        token_len = torch.tensor(token_len, dtype=torch.long) if token_len[0] is not None else None
        token = torch.stack(
            [F.pad(x, (0, token_len.max() - x.shape[-1])) for x in token],
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

        return token, token_len, spect, spect_len, audio, audio_len, None
