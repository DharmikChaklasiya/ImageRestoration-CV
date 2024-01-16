import os
import torch
import unittest
from unittest import TestCase
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np


base_path = "..\\test_images\\"
# Define a torchvision transform to convert PIL image to PyTorch tensor
# transform = transforms.ToTensor()

transform = transforms.Compose([
            transforms.ToTensor() # ,  transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
eps = 7e-02

        
from loss_utils import ssim_value, psnr_value

def generate_noisy_image(x: np.array, sigma: float) -> np.array:
    noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
    return x + noise


class Test(TestCase):
    

    def test_ssim_value(self):
        # https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images
        image1 = Image.open(os.path.join(base_path, "test_ssim1.jpg"))
        image2 = Image.open(os.path.join(base_path, "test_ssim2.jpg"))

        ssim = ssim_value(transform(image1).unsqueeze(0), transform(image2).unsqueeze(0))

        self.assertAlmostEqual(ssim.item(), 0.9639027981846681, delta = eps)

    # https://www.researchgate.net/figure/PSNR-values-for-Lena-test-images-with-different-noise-levels-and-denoised-with-the-SURE_tbl2_352023265
    def test_psnr_value_lena_15(self):
        image1 = Image.open(os.path.join(base_path, "pgd-lena-orig.jpg"))

        image1 = transform(image1)
        image2 = generate_noisy_image(image1 , sigma = 15.)

        cv_psnr = cv2.PSNR(np.float32(image1), np.float32(image2))
        
        psnr = psnr_value(image1, image2)

        self.assertAlmostEqual(psnr, 24.65, delta=eps)
        self.assertEqual(psnr, cv_psnr)

    def test_psnr_value_lena_20(self):
        image1 = Image.open(os.path.join(base_path, "pgd-lena-orig.jpg"))

        image1 = transform(image1)
        image2 = generate_noisy_image(image1 , sigma = 20.)

        cv_psnr = cv2.PSNR(np.float32(image1), np.float32(image2))
        
        psnr = psnr_value(image1, image2)

        self.assertAlmostEqual(psnr, 22.14, delta=eps)
        self.assertEqual(psnr, cv_psnr)

    def test_psnr_value_lena_25(self):
        image1 = Image.open(os.path.join(base_path, "pgd-lena-orig.jpg"))

        image1 = transform(image1)
        image2 = generate_noisy_image(image1 , sigma = 25.)

        cv_psnr = cv2.PSNR(np.float32(image1), np.float32(image2))
        
        psnr = psnr_value(image1, image2)

        self.assertAlmostEqual(psnr, 20.17, delta=eps)
        self.assertEqual(psnr, cv_psnr)

    def test_psnr_value_lena_30(self):
        image1 = Image.open(os.path.join(base_path, "pgd-lena-orig.jpg"))

        image1 = transform(image1)
        image2 = generate_noisy_image(image1 , sigma = 30.)

        cv_psnr = cv2.PSNR(np.float32(image1), np.float32(image2))
        
        psnr = psnr_value(image1, image2)

        self.assertAlmostEqual(psnr, 18.62, delta=eps)
        self.assertEqual(psnr, cv_psnr)


if __name__ == '__main__':
    unittest.main()



        