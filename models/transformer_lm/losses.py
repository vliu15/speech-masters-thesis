import torch
import torch.nn as nn
import torch.nn.functional as F


class MaximumMutualInformationLoss(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, yh, y):
        assert yh.shape == y.shape
        p_zy = F.softmax(yh, dim=-1)
        p_z = p_zy.mean(0)
        h_z = -1. * (p_z * torch.log(p_z)).sum(-1)

        x = p_zy * F.log_softmax(F.one_hot(y, num_classes=self.num_classes), dim=-1)
        h_z_x_ub = -1 * x.sum(-1).mean(0)
        return h_z_x_ub - h_z


class FocalLoss(nn.Module):

    def __init__(self, gamma: int = 0, alpha: float = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        return loss.mean()
