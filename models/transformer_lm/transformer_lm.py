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

        # Loss function
        if config.model.loss_type == "ce":
            self.loss = nn.CrossEntropyLoss(reduction="mean")
        elif config.model.loss_type == "mmi":
            from models.transformer_lm.losses import MaximumMutualInformationLoss
            self.loss = MaximumMutualInformationLoss(num_classes=config.model.vocab_size)
        elif config.model.loss_type == "focal":
            from models.transformer_lm.losses import FocalLoss
            self.loss = FocalLoss(gamma=10.0, reduction="mean")
        else:
            raise ValueError(f"Loss function {config.model.loss} not supported")

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
        loss = self.loss(xh_flat[loss_mask], target)
        accuracy = torch.sum(target == (xh_flat[loss_mask].argmax(1))).float() / torch.sum(loss_mask)

        if not self.training:
            yh = self.reconstruct(xh[:, :-1, :].argmax(-1), src_key_padding_mask[:, None, :-1])
        else:
            yh = None

        return {"loss": loss, "yh": yh}, {"accuracy": accuracy}

    @torch.no_grad()
    def sample(self, batch_size: int, n_steps: int, device: str = "cuda", sigma: float = 1.0):
        from tqdm import tqdm
        assert sigma > 0, "Temperature scalar must be positive"

        q = torch.tensor([TransformerLM.BOS] * batch_size, dtype=torch.long, device=device).unsqueeze(-1)  # B x T
        for _ in tqdm(range(n_steps), desc=f"Sampling from {self.__class__.__name__}"):
            x = q.permute(1, 0)
            x = self.embedding(x) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            x = self.transformer(x, mask=None, src_key_padding_mask=None)
            x = self.classifier(x)
            x = F.softmax(x[-1, :, :] / sigma, dim=-1)  # B x C
            x = torch.multinomial(x, 1)  # B x 1
            q = torch.cat([q, x], dim=-1)

        q = q[:, 1:]
        xh = self.reconstruct(q, torch.ones_like(q).to(x.dtype).unsqueeze(1))
        return xh, q
