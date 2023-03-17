import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class HeteroscedasticLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predicted_mean, predicted_logvar, target_mean):
        heteroscedastic_loss = (0.5 * predicted_logvar.neg().exp() * torch.pow(predicted_mean - target_mean, 2))\
                               + (0.5 * predicted_logvar)
        if self.reduction == 'sum':
            return torch.sum(heteroscedastic_loss)
        elif self.reduction == 'mean':
            return torch.mean(heteroscedastic_loss)
        elif self.reduction == 'none':
            return heteroscedastic_loss
        else:
            raise ValueError("Loss reduction option not found")


class MdnLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def gaussian_distribution(pred_mean, pred_sigma, target_mean):
        result = -0.5 * ((target_mean.expand_as(pred_mean) - pred_mean) * torch.reciprocal(pred_sigma))**2
        pdf_val = 1.0 / np.sqrt(2.0 * np.pi) * (torch.exp(result) * torch.reciprocal(pred_sigma))
        return pdf_val

    def forward(self, pred_pi, pred_mean, pred_sigma, target_mean):
        pdf_results = pred_pi * self.gaussian_distribution(pred_mean, pred_sigma, target_mean)
        mdn_loss = -torch.log(torch.sum(pdf_results, dim=1))

        if self.reduction == 'sum':
            return torch.sum(mdn_loss)
        elif self.reduction == 'mean':
            return torch.mean(mdn_loss)
        elif self.reduction == 'none':
            return mdn_loss
        else:
            raise ValueError("Loss reduction option not found")
