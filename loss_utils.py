import torch
from torch.nn import functional as F
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio

# avoid negative results
ssim_loss = SSIM(data_range=1.0, nonnegative_ssim=True)
# psnr_loss = PeakSignalNoiseRatio(data_range=1.0)


def psnr_based_loss(output, target):
    mse = F.mse_loss(output, target) # L2 loss - mean squared error
    loss = 10 * torch.log10(1.0/mse)
    return torch.mean(loss).item()
    #return psnr_loss(output, target).item()

def ssim_based_loss(output, target):
    # Calculate SSIM loss
    loss = 1 - ssim_loss(output, target)  # 1 - SSIM since we want to minimize the loss
    return loss
