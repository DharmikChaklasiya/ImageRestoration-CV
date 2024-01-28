from PIL import Image
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from pytorch_msssim import SSIM
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Transform for converting PIL Images to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

def psnr_value(output, target):
    output = np.array(output)
    target = np.array(target)
    # Check if the images have the same size
    if output.shape != target.shape:
        raise ValueError(f"Image sizes are different. Output: {output.shape}, Target: {target.shape}")
        
    return cv2.PSNR(np.float32(output), np.float32(target))

def ssim_value(output, target):
    output_tensor = transform(output).unsqueeze(0)  # Add batch dimension
    target_tensor = transform(target).unsqueeze(0)  # Add batch dimension

    #output_tensor.size(1)  # Grayscale images
    # Initialize SSIM for single-channel images
    ssim_norm = SSIM(data_range=1., nonnegative_ssim=True, channel=1)

    # Calculate SSIM
    ssim_out = ssim_norm(output_tensor, target_tensor)
    return ssim_out.item()

def plot_metrics(psnr_scores,ssim_scores,output_dir):
  # Plotting the PSNR and SSIM values
  plt.figure(figsize=(10, 5))

  # PSNR plot
  plt.subplot(1, 2, 1)
  plt.plot(psnr_scores, label='PSNR')
  plt.title('PSNR Scores')
  plt.xlabel('Image Index')
  plt.ylabel('PSNR')
  plt.xticks([])

  # SSIM plot
  plt.subplot(1, 2, 2)
  plt.plot(ssim_scores, label='SSIM', color='orange')
  plt.title('SSIM Scores')
  plt.xlabel('Image Index')
  plt.ylabel('SSIM')
  plt.xticks([])

  plt.legend()
  plt.tight_layout()
  plt.savefig(f'{output_dir}/plot.png')


        
