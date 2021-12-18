import torch
import torch.nn as nn

from models.base import WaveformReconstructionModel
from models.glow_tts import submodules
from models.vqvae.bottleneck import Bottleneck, NoBottleneck
from models.vqvae.encdec import Decoder, Encoder
from models.vqvae.losses import MultiNormReconstructionLoss, MultiResolutionSpectralLoss


class VQVAE(WaveformReconstructionModel):

    def __init__(self, config):
        super().__init__()
        self.levels = config.model.levels

        if config.model.multipliers is None:
            self.multipliers = [1] * config.model.levels
        else:
            assert len(config.model.multipliers) == config.model.levels, "Invalid number of multipliers"
            self.multipliers = config.model.multipliers

        encoder = lambda level: Encoder(
            input_emb_width=1,
            output_emb_width=config.model.emb_width,
            levels=level + 1,
            downs_t=config.model.downs_t[:level + 1],
            strides_t=config.model.strides_t[:level + 1],
            width=config.model.width * self.multipliers[level],
            depth=config.model.depth * self.multipliers[level],
            m_conv=config.model.m_conv,
            dilation_growth_rate=config.model.dilation_growth_rate,
            dilation_cycle=config.model.dilation_cycle,
            reverse_decoder_dilation=config.model.reverse_decoder_dilation,
            zero_out=config.model.zero_out,
        )
        decoder = lambda level: Decoder(
            input_emb_width=1,
            output_emb_width=config.model.emb_width,
            levels=level + 1,
            downs_t=config.model.downs_t[:level + 1],
            strides_t=config.model.strides_t[:level + 1],
            width=config.model.width * self.multipliers[level],
            depth=config.model.depth * self.multipliers[level],
            m_conv=config.model.m_conv,
            dilation_growth_rate=config.model.dilation_growth_rate,
            dilation_cycle=config.model.dilation_cycle,
            reverse_decoder_dilation=config.model.reverse_decoder_dilation,
            zero_out=config.model.zero_out,
        )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(config.model.levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        # HACK: Hard-code to last level for now, leave as is until scaling up
        config.model.levels = 1
        config.model.multipliers = config.model.multipliers[:1]
        self.levels = 1
        self.encoders = self.encoders[:1]
        self.decoders = self.decoders[:1]
        # ####

        if config.model.use_bottleneck:
            self.bottleneck = Bottleneck(config.model.l_bins, config.model.emb_width, config.model.mu, config.model.levels)
        else:
            self.bottleneck = NoBottleneck(config.model.levels)

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

        self.downs_t = config.model.downs_t
        self.strides_t = config.model.strides_t
        self.l_bins = config.model.l_bins
        self.commit = config.model.loss.commit
        self.multispectral = config.model.loss.multispectral

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantized = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantized) == end_level - start_level

        # Use only lowest level
        decoder, x_quantized = self.decoders[start_level], xs_quantized[0:1]
        x_out = decoder(x_quantized, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [torch.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return torch.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = torch.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [torch.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples, z_shapes):
        zs = [torch.randint(0, self.l_bins, size=(n_samples, *z_shape), device="cuda") for z_shape in z_shapes]
        return self.decode(zs)

    def forward(self, x, x_lengths, speaker=None):
        x_mask = torch.unsqueeze(submodules.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        # Encode
        xs = []
        for level in range(self.levels):
            xs += [self.encoders[level](x, x_mask)]  # encoder returns x and x_mask

        # Quantize
        xs, x_masks = zip(*xs)
        _, xqs, commit_losses, quantizer_metrics = self.bottleneck(xs, x_masks)

        # Decode
        x_outs = []
        for level in range(self.levels):
            x_out, _ = self.decoders[level](xqs[level:level + 1], x_masks[level:level + 1], all_levels=False)
            assert x_out.shape == x.shape, f"Expected shape {x.shape}, got {x_out.shape}."
            x_outs += [x_out]

        # Loss
        loss_recon, loss_stft = 0., 0.
        for level in reversed(range(self.levels)):
            x_out = x_outs[level]
            loss_recon += self.multi_recon_loss(x, x_out, x_mask)
            loss_stft += self.multi_stft_loss(x, x_out, x_mask)

        loss_commit = sum(commit_losses)
        loss = loss_recon + self.multispectral * loss_stft + self.commit * loss_commit

        return {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_stft": loss_stft,
            "loss_commit": loss_commit,
            "yh": x_out.squeeze(1),
        }, quantizer_metrics[-1] if self.training else {}
