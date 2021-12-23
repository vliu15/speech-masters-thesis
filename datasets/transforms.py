"""Pytorch transforms for audio tensors"""

import random

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.util import pad_center, tiny
from scipy.signal import get_window

from utils.torch_utils import safe_log


class MelSpectrogram(nn.Module):

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = None,
        n_mels: int = 80,
        sample_rate: int = 22050,
        f_min: float = 0.0,
        f_max: float = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            win_length = n_fft
        self.stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
        )
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def forward(self, audio, jitter_steps: int = 0):
        assert audio.min() >= -1 and audio.max() <= 1
        # Add batch dimension if not already present
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        # Phase shift by jitter_steps if specified
        if jitter_steps > 0:
            length = audio.size(-1)
            audio = F.pad(audio, (jitter_steps, jitter_steps), "constant")
            random_start = random.randint(0, 2 * jitter_steps)
            audio = audio[:, random_start:random_start + length]

        # Compute stft and log-mel
        magnitudes = self.stft(audio)
        mel = torch.matmul(self.mel_basis, magnitudes)
        mel = safe_log(mel)
        return mel

    def mel_len(self, audio_len):
        return audio_len // self.hop_length


class STFT(nn.Module):

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = None,
        window: str = "hann",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length else n_fft
        self.window = window
        self.forward_transform = None
        self.pad_amount = (self.n_fft - self.hop_length) // 2
        scale = self.n_fft / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.n_fft))

        cutoff = int((self.n_fft / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        assert n_fft >= self.win_length
        # Get window and zero center pad it to n_fft
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, n_fft)
        fft_window = torch.from_numpy(fft_window).float()

        # Window the bases
        forward_basis *= fft_window
        inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis)
        self.register_buffer("inverse_basis", inverse_basis)

    def forward(self, input_data):
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        # Similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)

        input_data = F.pad(input_data.unsqueeze(1), (self.pad_amount, self.pad_amount, 0, 0), mode="reflect")
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(input_data, self.forward_basis, stride=self.hop_length, padding=0)

        cutoff = int((self.n_fft / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        return torch.sqrt(real_part**2 + imag_part**2)

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = librosa.filters.window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.n_fft,
                dtype=np.float32,
            )

            # Remove modulation effects
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # Scale by hop ratio
            inverse_transform *= self.n_fft / self.hop_length

        inverse_transform = inverse_transform[:, :, self.pad_amount:]
        inverse_transform = inverse_transform[:, :, :-self.pad_amount:]

        return inverse_transform
