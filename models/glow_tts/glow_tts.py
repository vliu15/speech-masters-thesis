import math

import torch
import torch.nn as nn

import models.glow_tts.submodules as submodules
from models.base import TokenToSpectrogramModel
from models.glow_tts.modules import FlowSpecDecoder, TextEncoder
from models.parser import CMUDictParser


class GlowTTS(TokenToSpectrogramModel):
    """Write a wrapper for abstracting forward passes in training and inference."""

    def __init__(self, config):
        super().__init__()
        if config.model.n_speakers > 1:
            self.emb_g = nn.Embedding(config.model.n_speakers, config.model.gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

        self.encoder = TextEncoder(
            n_vocab=config.model.encoder.n_vocab + int(config.dataset.intersperse_blanks),
            out_channels=config.dataset.n_mels,
            hidden_channels=config.model.encoder.hidden_channels,
            filter_channels=config.model.encoder.filter_channels,
            filter_channels_dp=config.model.encoder.filter_channels,
            n_heads=config.model.encoder.n_heads,
            n_layers=config.model.encoder.n_layers,
            kernel_size=config.model.encoder.kernel_size,
            p_dropout=config.model.encoder.p_dropout,
            window_size=config.model.encoder.window_size,
            mean_only=config.model.encoder.mean_only,
            prenet=config.model.encoder.prenet,
            gin_channels=config.model.gin_channels,
        )
        self.decoder = FlowSpecDecoder(
            in_channels=config.dataset.n_mels,
            hidden_channels=config.model.decoder.hidden_channels,
            kernel_size=config.model.decoder.kernel_size,
            dilation_rate=config.model.decoder.dilation_rate,
            n_blocks=config.model.decoder.n_blocks,
            n_layers=config.model.decoder.n_layers,
            p_dropout=config.model.decoder.p_dropout,
            n_split=config.model.decoder.n_split,
            n_sqz=config.model.decoder.n_sqz,
            sigmoid_scale=config.model.decoder.sigmoid_scale,
            gin_channels=config.model.gin_channels,
        )
        self.parser = CMUDictParser(config.dataset.cmudict_path)

    @torch.no_grad()
    def ddi(self, batch):
        self.train()
        _ = self.supervised_step(batch)
        for f in self.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(True)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        speaker: torch.Tensor = None
    ):
        if speaker is not None:
            speaker_embeddings = self.emb_g(speaker)
        else:
            speaker_embeddings = None

        # Encode: x -> x_enc
        if x_lengths is None:
            x_lengths = torch.ones_like(x).sum(-1)
        x_m, x_logs, logw_enc, x_mask = self.encoder(text=x, text_lengths=x_lengths, speaker_embeddings=speaker_embeddings)

        # Decode inverse: y -> z_dec
        y_max_length = (y.size(2) // self.decoder.n_sqz) * self.decoder.n_sqz
        y = y[:, :, :y_max_length]
        if y_lengths is None:
            y_lengths = torch.ones_like(y).sum(-1)
        y_lengths = (y_lengths // self.decoder.n_sqz) * self.decoder.n_sqz
        y_mask = torch.unsqueeze(submodules.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        z_dec, logdet = self.decoder(spect=y, spect_mask=y_mask, speaker_embeddings=speaker_embeddings, reverse=False)

        # Monotonic alignment search
        with torch.no_grad():
            attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

            x_s_sq_r = torch.exp(-2 * x_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z_dec**2))  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1, 2), z_dec)  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (x_m**2) * x_s_sq_r, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

            attn = submodules.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach().squeeze(1)

        # Align x_enc -> z_enc
        logw_dec = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask.squeeze()
        z_m_enc = torch.matmul(x_m, attn)
        z_logs_enc = torch.matmul(x_logs, attn)

        # Decode forward: z_enc -> yh
        if self.training:
            yh = None
        else:
            with torch.no_grad():
                w = torch.sum(attn, -1) * x_mask.squeeze()
                z_lengths = torch.clamp_min(torch.sum(w, [1]), 1).long()
                z_lengths = (z_lengths // self.decoder.n_sqz) * self.decoder.n_sqz

                z_mask = torch.unsqueeze(submodules.sequence_mask(z_lengths, None), 1).to(x_mask.dtype)
                z_enc = (z_m_enc + torch.exp(z_logs_enc) * torch.randn_like(z_m_enc)) * z_mask
                yh, *_ = self.decoder(spect=z_enc, spect_mask=z_mask, speaker_embeddings=speaker_embeddings, reverse=True)

        # Loss
        logdet = torch.sum(logdet)
        l_mle = 0.5 * math.log(
            2 * math.pi
        ) + (torch.sum(z_logs_enc) + 0.5 * torch.sum(torch.exp(-2 * z_logs_enc) * (z_dec - z_m_enc)**2) - logdet) / (
            torch.sum(y_lengths) * z_dec.shape[1]
        )
        l_length = torch.sum((logw_enc - logw_dec)**2) / torch.sum(x_lengths)

        return {
            "loss_mle": l_mle,
            "loss_length": l_length,
            "loss": l_mle + l_length,
            "yh": yh,
        }, {}

    @torch.no_grad()
    def infer_step(self, t: str, speaker: torch.Tensor = None):
        assert not self.training, f"Put {self.__class__.__name__} in evaluation model before calling this function."

        # Prepare text
        t = t.strip()
        if t[-1] not in [".", "!", "?"]:
            t = t + "."

        # Phonemize text
        x = self.parser(t)
        x = torch.tensor(x).unsqueeze(0).long().to(self.device)
        x_lengths = torch.ones_like(x).sum(-1)

        if speaker is not None:
            speaker_embeddings = self.emb_g(speaker)
        else:
            speaker_embeddings = None

        # Encode: x -> x_enc
        x_m, x_logs, logw_enc, x_mask = self.encoder(text=x, text_lengths=x_lengths, speaker_embeddings=speaker_embeddings)

        # Align x_enc -> z_enc
        w = torch.ceil(torch.exp(logw_enc) * x_mask.squeeze())
        z_lengths = torch.clamp_min(torch.sum(w, [1]), 1).long()
        z_lengths = int(z_lengths // self.decoder.n_sqz) * self.decoder.n_sqz
        z_mask = torch.unsqueeze(submodules.sequence_mask(z_lengths, None), 1).to(x_mask.dtype)

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)
        attn = submodules.generate_path(w.squeeze(1), attn_mask.squeeze(1))
        z_m_enc = torch.matmul(x_m, attn)
        z_logs_enc = torch.matmul(x_logs, attn)

        z_enc = (z_m_enc + torch.exp(z_logs_enc) * torch.randn_like(z_m_enc)) * z_mask
        yh, *_ = self.decoder(spect=z_enc, spect_mask=z_mask, speaker_embeddings=speaker_embeddings, reverse=True)
        return yh
