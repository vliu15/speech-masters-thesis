import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

import models.glow_tts.submodules as submodules
from models.base import TokenToWaveformModel
from models.vqvae.vqvae import VQVAE


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerLM(TokenToWaveformModel):

    PAD = 0  # <pad> token
    BOS = 1  # <bos> token
    OFFSET = 2  # number of special tokens to offset original vocabulary by

    def __init__(self, config):
        super().__init__()
        self.d_model = config.model.d_model

        # Embedding and positional encoding
        self.embedding = nn.Embedding(
            num_embeddings=config.model.vocab_size + TransformerLM.OFFSET,
            embedding_dim=config.model.embed_dim,
            padding_idx=TransformerLM.PAD,
        )
        self.pos_encoding = PositionalEncoding(
            d_model=config.model.d_model,
            dropout=config.model.dropout,
            max_len=config.model.max_len,
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.model.num_layers,
            norm=nn.LayerNorm(config.model.d_model, eps=config.model.layer_norm_eps),
        )

        # Classifier
        self.classifier = nn.Linear(config.model.d_model, config.model.vocab_size)

        # Load VQVAE for audio reconstruction
        self.vqvae = TransformerLM.load_vqvae(config.model.vqvae.log_dir, config.model.vqvae.ckpt_num)

    @staticmethod
    def load_vqvae(log_dir, ckpt_num):
        # Load config and ckpt
        config = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
        ckpt = torch.load(os.path.join(log_dir, "ckpts", f"ckpt.{ckpt_num}.pt"), map_location="cpu")

        # Initialize model
        vqvae = VQVAE(config)
        vqvae.load_state_dict(ckpt["model"])
        return nn.ModuleDict(
            {
                "bottleneck": vqvae.bottleneck.level_blocks[vqvae.LEVEL],
                "decoder": vqvae.decoders[vqvae.LEVEL]
            }
        )

    def reconstruct(self, q, mask):
        # Dequantize
        y = self.vqvae["bottleneck"].decode(q)

        # Decode
        y, mask = self.vqvae["decoder"]([y], [mask.to(y.dtype)], all_levels=False)
        y = y * mask
        return y.squeeze(1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, y: torch.Tensor, y_lengths: torch.Tensor, speaker=None):
        src_key_padding_mask = submodules.sequence_mask(x_lengths, x.size(1)).bool()
        mask = torch.triu(torch.ones(x.size(1), x.size(1)) * float("-inf"), diagonal=1).to(x.device)

        xh = x.permute(1, 0)
        xh = self.embedding(xh) * math.sqrt(self.d_model)
        xh = self.pos_encoding(xh)
        xh = self.transformer(xh, mask=mask.to(xh.dtype), src_key_padding_mask=~src_key_padding_mask)
        xh = self.classifier(xh)
        xh = xh.permute(1, 0, 2)  # B x T x C

        x_flat = x[:, 1:].flatten()
        xh_flat = xh[:, :-1, :].reshape(len(x_flat), -1)

        # Undo offset from special tokens
        loss_mask = (x_flat >= TransformerLM.OFFSET)
        target = x_flat[loss_mask] - TransformerLM.OFFSET
        loss = F.cross_entropy(xh_flat[loss_mask], target, reduction="mean")
        accuracy = torch.sum(target == xh_flat[loss_mask].argmax(1)) / torch.sum(loss_mask)

        if not self.training:
            yh = self.reconstruct(xh[:, :-1, :].argmax(-1), src_key_padding_mask[:, None, :-1])
        else:
            yh = None

        return {"loss": loss, "yh": yh}, {"accuracy": accuracy}
