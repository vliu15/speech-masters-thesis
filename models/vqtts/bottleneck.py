import torch
import torch.nn.functional as F

from models.vqvae.bottleneck import BottleneckBlock


class Bottleneck(BottleneckBlock):

    def __init__(self, n_vocab: int, l_bins: int, emb_width: int, mu: float, threshold: float):
        super().__init__(
            k_bins=n_vocab * l_bins,
            emb_width=emb_width,
            mu=mu,
            threshold=threshold,
        )
        self.n_vocab = n_vocab
        self.l_bins = l_bins

    def forward(self, y_enc, x_id, attn, update_k=True):
        b, tx, ty = attn.shape
        c = y_enc.shape[1]

        # Compute mask along ty so as to not corrupt learned latents
        mask = attn.sum(1).reshape(b * ty, 1)
        indices = mask.bool().flatten()

        # Align x_id to y_enc
        x_id = torch.matmul(x_id.to(attn.dtype), attn).long()

        # Preprocess (flatten)
        y_enc = y_enc.permute(0, 2, 1).reshape(b * ty, c)  # (b * ty, 1, c)
        x_id = x_id.reshape(b * ty)  # (b * ty)

        # Init k if not inited
        if update_k and not self.init:
            self.init_k(y_enc)

        # Grab relevant centroids
        k = self.k.reshape(self.n_vocab, self.l_bins, c)
        k = k[x_id]  # (b * ty, l, c)

        # Quantize and dequantize
        with torch.no_grad():
            k_t = k.transpose(1, 2)
            distance = torch.sum(
                y_enc.unsqueeze(1)**2, dim=-1
            ) - 2 * torch.bmm(y_enc.unsqueeze(1), k_t).squeeze(1) + torch.sum(
                k**2, dim=-1
            )  # (b * ty, l)

            # NOTE: codebook indices are relative per group
            min_distance, q_rel = torch.min(distance, dim=-1)

            fit = torch.sum(min_distance * mask) / (mask.sum() * distance.shape[-1])

            # Since we're doing grouped quantization, we need to convert relative
            # group indices (l_bins) into absolute indices (k_bins)
            q_abs = (x_id * self.l_bins + q_rel).long()

            y_d = F.embedding(q_abs, self.k)

        if self.training:
            update_metrics = self.update_k(y_enc[indices], q_abs[indices])
        else:
            update_metrics = {}

        # Loss
        commit_loss = torch.norm(y_d[indices].detach() - y_enc[indices])**2 / (mask.sum() * c)

        # Passthrough
        y_d = y_enc + (y_d - y_enc).detach()

        # Postprocess (unflatten)
        y_d = (y_d * mask).reshape(b, ty, c).permute(0, 2, 1)
        q_rel = q_rel.reshape(b, ty)

        return q_rel, y_d, commit_loss, dict(fit=fit, **update_metrics)
