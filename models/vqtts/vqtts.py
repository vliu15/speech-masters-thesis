import torch
import torch.nn as nn
import torch.nn.functional as F

import models.glow_tts.submodules as submodules
from models.base import TokenToWaveformModel
from models.glow_tts.modules import TextEncoder
from models.parser import CMUDictParser
from models.vqvae.bottleneck import BottleneckBlock
from models.vqvae.encdec import Decoder, Encoder
from models.vqvae.losses import MultiNormReconstructionLoss, MultiResolutionSpectralLoss
from models.vqvae.resnet import ResNetBlock
from utils.torch_utils import safe_log


class VQTTS(TokenToWaveformModel):

    def __init__(self, config):
        super().__init__()
        if config.model.n_speakers > 1:
            self.emb_g = nn.Embedding(config.model.n_speakers, config.model.gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

        self.audio_encoder = Encoder(
            input_emb_width=1,
            output_emb_width=config.model.emb_width,
            levels=config.model.levels,
            downs_t=config.model.downs_t,
            strides_t=config.model.strides_t,
            width=config.model.width * config.model.multipliers[-1],
            depth=config.model.depth * config.model.multipliers[-1],
            m_conv=config.model.m_conv,
            block_type=config.model.block_type,
            dilation_growth_rate=config.model.dilation_growth_rate,
            kernel_size_growth_rate=config.model.kernel_size_growth_rate,
            kernel_size_cycle=config.model.kernel_size_cycle,
            dilation_cycle=config.model.dilation_cycle,
            reverse_decoder_dilation=config.model.reverse_decoder_dilation,
            zero_out=config.model.zero_out,
        )
        self.audio_decoder = Decoder(
            input_emb_width=1,
            output_emb_width=config.model.emb_width,
            levels=config.model.levels,
            downs_t=config.model.downs_t,
            strides_t=config.model.strides_t,
            width=config.model.width * config.model.multipliers[-1],
            depth=config.model.depth * config.model.multipliers[-1],
            m_conv=config.model.m_conv,
            block_type=config.model.block_type,
            dilation_growth_rate=config.model.dilation_growth_rate,
            dilation_cycle=config.model.dilation_cycle,
            kernel_size_growth_rate=config.model.kernel_size_growth_rate,
            kernel_size_cycle=config.model.kernel_size_cycle,
            reverse_decoder_dilation=config.model.reverse_decoder_dilation,
            zero_out=config.model.zero_out,
        )
        self.text_parser = CMUDictParser(config.dataset.cmudict_path)
        self.text_encoder = TextEncoder(
            n_vocab=config.model.encoder.n_vocab + int(config.dataset.intersperse_blanks),
            out_channels=config.model.encoder.out_channels,
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
        self.quant_bottleneck = BottleneckBlock(
            config.model.l_bins, config.model.emb_width, config.model.mu, config.model.revival_threshold
        )
        self.quant_decoder = ResNetBlock(
            n_in=config.model.encoder.out_channels,
            n_depth=4,
            m_conv=2.0,
            dilation_growth_rate=3,
            dilation_cycle=None,
            zero_out=True,
            res_scale=False,
            reverse_dilation=True,
        )
        self.quant_proj = nn.Conv1d(config.model.encoder.out_channels, config.model.l_bins, 1)

        self.multi_stft_loss = MultiResolutionSpectralLoss(
            n_ffts=config.model.loss.n_ffts,
            hop_lengths=config.model.loss.hop_lengths,
            win_lengths=config.model.loss.win_lengths,
            window=config.model.loss.window,
            log=config.model.loss.log,
        )
        self.multi_recon_loss = MultiNormReconstructionLoss(
            l1=config.model.loss.l1,
            l2=config.model.loss.l2,
            linf=config.model.loss.linf,
            linf_topk=config.model.loss.linf_topk,
        )

        self.l_bins = config.model.l_bins

        self.l_commit = config.model.loss.commit
        self.l_stft = config.model.loss.multispectral
        self.l_align = config.model.loss.align

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

        # Encode text: x -> x_enc
        if x_lengths is None:
            x_lengths = torch.ones_like(x).sum(-1)
        x_enc, _, logw_enc, x_mask = self.text_encoder(text=x, text_lengths=x_lengths, speaker_embeddings=speaker_embeddings)

        # Encode audio: y -> y_enc
        if y_lengths is None:
            y_lengths = torch.ones_like(y).sum(-1)
        y_mask = torch.unsqueeze(submodules.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype).detach()
        y_enc, q_mask = self.audio_encoder(y, y_mask)

        # Align text and audio: (x_enc, y_enc) -> attn
        with torch.no_grad():
            attn_mask = x_mask.unsqueeze(-1) * q_mask.unsqueeze(2)
            distances = (x_enc.unsqueeze(-1) - y_enc.unsqueeze(2)).pow(2).sum(1).sqrt()
            attn = submodules.maximum_path(-distances, attn_mask.squeeze(1))

        # Quantize/dequantize audio: (y_enc, attn) -> y_q, y_d
        y_q, y_d, loss_commit, _ = self.quant_bottleneck(y_enc, x, attn)

        # Decode quants: (x_enc, attn) -> y_qh
        y_qh, _ = self.quant_decoder(torch.matmul(x_enc, attn).detach(), q_mask)
        y_qh = self.quant_proj(y_qh * q_mask)

        # Decode audio: y_enc -> y_h
        y_h, _ = self.audio_decoder([y_d], [q_mask], all_levels=False)

        # Compute loss
        logw_dec = safe_log(torch.sum(attn, -1)) * x_mask.squeeze()
        align = (x_enc.unsqueeze(-1) - y_enc.unsqueeze(2)).pow(2).sum(1).sqrt()

        loss_recon = self.multi_recon_loss(y, y_h, y_mask)
        loss_stft = self.multi_stft_loss(y, y_h, y_mask)
        loss_dur = torch.sum(torch.pow(logw_enc - logw_dec, 2)) / torch.sum(x_lengths)
        loss_align = (align * attn).sum() / attn_mask.sum()
        loss_ce = F.cross_entropy(y_qh, y_q)

        loss = sum(
            [
                loss_recon,
                self.l_stft * loss_stft,
                self.l_commit * loss_commit,
                loss_dur,
                self.l_align * loss_align,
                loss_ce,
            ]
        )

        # if not self.training:
        #     y_qh_rel = y_qh.argmax(1)
        #     y_qh_abs = torch.matmul(x.to(attn.dtype), attn).squeeze(1).long() * self.l_bins + y_qh_rel
        #     y_d = F.embedding(y_qh_abs, self.quant_bottleneck.k).permute(0, 2, 1)
        #     y_h, _ = self.audio_decoder([y_d], [q_mask], all_levels=False)
        if not self.training:
            y_qh = y_qh.argmax(1)
            y_d = F.embedding(y_qh, self.quant_bottleneck.k).permute(0, 2, 1)
            y_h, _ = self.audio_decoder([y_d], [q_mask], all_levels=False)

        return {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_stft": loss_stft,
            "loss_commit": loss_commit,
            "loss_dur": loss_dur,
            "loss_align": loss_align / (1 + self.l_align),
            "loss_ce": loss_ce,
            "yh": y_h.squeeze(1),
        }, {
            "q_acc": (y_qh.argmax(1) == y_q).float().mean()
        }
