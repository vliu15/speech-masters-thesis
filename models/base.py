"""Implements abstract base classes for various TTS models"""

import torch.nn as nn


class TokenToWaveformModel(nn.Module):
    """Maps input tokens to audio waveform"""

    def supervised_step(self, batch):
        x, x_lengths, _, _, y, y_lengths, speaker = batch
        loss_dict, metrics_dict = self.forward(x, x_lengths, y, y_lengths, speaker=speaker)
        loss_dict["y"] = y
        return loss_dict, metrics_dict

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"Forward function not implemented for {self.__class__.__name__}")


class WaveformReconstructionModel(nn.Module):
    """Reconstructs audio waveform through encoding/decoding"""

    def supervised_step(self, batch):
        _, _, _, _, x, x_lengths, speaker = batch
        loss_dict, metrics_dict = self.forward(x, x_lengths, speaker=speaker)
        loss_dict["y"] = x.squeeze(1)
        return loss_dict, metrics_dict

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"Forward function not implemented for {self.__class__.__name__}")


class TokenToSpectrogramModel(nn.Module):
    """Maps input tokens to spectrogram model"""

    def supervised_step(self, batch):
        x, x_lengths, y, y_lengths, _, _, speaker = batch
        loss_dict, metrics_dict = self.forward(x, x_lengths, y, y_lengths, speaker=speaker)
        loss_dict["y"] = y
        return loss_dict, metrics_dict

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"Forward function not implemented for {self.__class__.__name__}")


class SpectrogramReconstructionModel(nn.Module):
    """Reconstructs spectrogram through encoding/decoding"""

    def supervised_step(self, batch):
        _, _, y, y_lengths, _, _, speaker = batch
        loss_dict, metrics_dict = self.forward(y, y_lengths, speaker=speaker)
        loss_dict["y"] = y
        return loss_dict, metrics_dict

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"Forward function not implemented for {self.__class__.__name__}")
