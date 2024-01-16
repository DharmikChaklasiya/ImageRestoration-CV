import torch
from torch.nn import functional as F
from pytorch_msssim import SSIM
import math
import cv2
import numpy as np


# in oder to avoid negative results
ssim_norm = SSIM(data_range=1., nonnegative_ssim=True)


def psnr_value(output, target):    
        return cv2.PSNR(np.float32(output), np.float32(target)) # R = 255. default

def ssim_value(output, target):
        return  ssim_norm(output, target)

