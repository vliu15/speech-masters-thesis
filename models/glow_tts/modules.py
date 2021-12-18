import math

import torch
import torch.nn as nn

import models.glow_tts.submodules as submodules


class TextEncoder(nn.Module):

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        filter_channels_dp: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        window_size: int,
        mean_only: bool = False,
        prenet: bool = False,
        gin_channels: int = 0,
    ):
        """
        GlowTTS text encoder. Takes in the input text tokens and produces prior distribution statistics for the latent
        representation corresponding to each token, as well as the log durations (the Duration Predictor is also part of
         this module).
        Architecture is similar to Transformer TTS with slight modifications.
        Args:
            n_vocab (int): Number of tokens in the vocabulary
            out_channels (int): Latent representation channels
            hidden_channels (int): Number of channels in the intermediate representations
            filter_channels (int): Number of channels for the representations in the feed-forward layer
            filter_channels_dp (int): Number of channels for the representations in the duration predictor
            n_heads (int): Number of attention heads
            n_layers (int): Number of transformer layers
            kernel_size (int): Kernels size for the feed-forward layer
            p_dropout (float): Dropout probability
            mean_only (bool): Return zeros for logs if true
            prenet (bool): Use an additional network before the transformer modules
            gin_channels (int): Number of channels in speaker embeddings
        """
        super().__init__()

        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.prenet = prenet
        self.mean_only = mean_only

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if prenet:
            self.pre = submodules.ConvReluNorm(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.1,
            )

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(
                submodules.AttentionBlock(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    window_size=window_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(submodules.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                submodules.FeedForwardNetwork(
                    hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout
                )
            )
            self.norm_layers_2.append(submodules.LayerNorm(hidden_channels))

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_w = submodules.DurationPredictor(hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout)

    def forward(self, text, text_lengths, speaker_embeddings=None):

        x = self.emb(text) * math.sqrt(self.hidden_channels)  # [b, t, h]

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(submodules.sequence_mask(text_lengths, x.size(2)), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)

        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask

        if speaker_embeddings is not None:
            g_exp = speaker_embeddings.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)

        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        logw = self.proj_w(spect=x_dp, mask=x_mask)

        return x_m, x_logs, logw, x_mask


class FlowSpecDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_blocks: int,
        n_layers: int,
        p_dropout: float = 0.0,
        n_split: int = 4,
        n_sqz: int = 2,
        sigmoid_scale: bool = False,
        gin_channels: int = 0,
    ):
        """
        Flow-based invertible decoder for GlowTTS. Converts spectrograms to latent representations and back.
        Args:
            in_channels (int): Number of channels in the input spectrogram
            hidden_channels (int): Number of channels in the intermediate representations
            kernel_size (int): Kernel size in the coupling blocks
            dilation_rate (int): Dilation rate in the WaveNet-like blocks
            n_blocks (int): Number of flow blocks
            n_layers (int): Number of layers within each coupling block
            p_dropout (float): Dropout probability
            n_split (int): Group size for the invertible convolution
            n_sqz (int): The rate by which the spectrograms are squeezed before applying the flows
            sigmoid_scale (bool): Apply sigmoid to logs in the coupling blocks
        """
        super().__init__()

        self.n_sqz = n_sqz

        self.flows = nn.ModuleList()
        for _ in range(n_blocks):
            self.flows.append(submodules.ActNorm(channels=in_channels * n_sqz))
            self.flows.append(submodules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
            self.flows.append(
                submodules.CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                    gin_channels=gin_channels,
                )
            )

    def forward(self, spect, spect_mask, speaker_embeddings=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        x = spect
        x_mask = spect_mask

        if self.n_sqz > 1:
            x, x_mask = self.squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=speaker_embeddings, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=speaker_embeddings, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = self.unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def squeeze(self, x, x_mask=None, n_sqz=2):
        b, c, t = x.size()

        t = (t // n_sqz) * n_sqz
        x = x[:, :, :t]
        x_sqz = x.view(b, c, t // n_sqz, n_sqz)
        x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

        if x_mask is not None:
            x_mask = x_mask[:, :, n_sqz - 1::n_sqz]
        else:
            x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
        return x_sqz * x_mask, x_mask

    def unsqueeze(self, x, x_mask=None, n_sqz=2):
        b, c, t = x.size()

        x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
        x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

        if x_mask is not None:
            x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
        else:
            x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
        return x_unsqz * x_mask, x_mask

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()
