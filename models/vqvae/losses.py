from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.transforms import STFT
from utils.torch_utils import safe_log


class MultiResolutionSpectralLoss(nn.Module):

    def __init__(
        self,
        n_ffts: Iterable[int],
        hop_lengths: Iterable[int],
        win_lengths: Iterable[int] = None,
        window: str = "hann",
        log: bool = False,
    ):
        super().__init__()
        if win_lengths is None:
            win_lengths = n_ffts
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)

        self.stfts = nn.ModuleList()
        self.hop_lengths = []
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.hop_lengths += [hop_length]
            self.stfts += [STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)]

        self.log = log

    @staticmethod
    def downsample_mask(mask, stft):
        mask = F.pad(mask, (stft.pad_amount, 0, 0, 0), mode="constant", value=1)
        mask = F.pad(mask, (0, stft.pad_amount, 0, 0), mode="constant", value=0)
        return mask[:, :, stft.n_fft // 2:-stft.n_fft // 2:stft.hop_length]

    def forward(self, y, yh, mask):
        loss = 0.0
        for stft in self.stfts:
            y_stft = stft(y)
            yh_stft = stft(yh)

            # NOTE: Mask is applied after STFT and before the norm. Applying mask before STFT causes instability
            mask_stft = self.downsample_mask(mask, stft)

            if self.log:
                y_stft = safe_log(y_stft)
                yh_stft = safe_log(yh_stft)

            y_stft = torch.norm(y_stft * mask_stft, p=2, dim=-1)
            yh_stft = torch.norm(yh_stft * mask_stft, p=2, dim=-1)

            loss += F.mse_loss(y_stft, yh_stft, reduction="none").sum(-1).sqrt().mean(0)
        return loss / len(self.stfts)


class MultiNormReconstructionLoss(nn.Module):

    def __init__(
        self,
        l1: float = 0.0,
        l2: float = 1.0,
        linf: float = 0.02,
        linf_topk: int = 2048,
    ):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.linf = linf
        self.linf_topk = linf_topk

    def forward(self, y, yh, mask):
        y = (y * mask).reshape(y.shape[0], -1)
        yh = (yh * mask).reshape(yh.shape[0], -1)
        return (
            self.l1 * F.l1_loss(y, yh).mean(0).sum() + \
            self.l2 * F.mse_loss(y, yh).mean(0).sum() + \
            self.linf * torch.topk((y - yh) ** 2, self.linf_topk, dim=-1)[0].mean(0).sum()
        )
