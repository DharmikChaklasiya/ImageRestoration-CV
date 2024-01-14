import os
import torch
import unittest
from unittest import TestCase
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np


base_path = "C:\\Users\\Ramona\\Documents\\Studium\\WS2023\\UE_CV\\test_images\\"
# Define a torchvision transform to convert PIL image to PyTorch tensor
# transform = transforms.ToTensor()

transform = transforms.Compose([
            transforms.ToTensor()
        ])
eps = 1e-02

        
from loss_utils import ssim_based_loss, psnr_based_loss
class Test(TestCase):
    
    def test_ssim_loss(self):
        # https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images
        image1 = Image.open(os.path.join(base_path, "test_ssim1.jpg"))
        image2 = Image.open(os.path.join(base_path, "test_ssim2.jpg"))

        #0.9704228043556213 != 0.9639027981846681 - slightly diff?
        ssim_loss = ssim_based_loss(transform(image1).unsqueeze(0), transform(image2).unsqueeze(0))

        # get the original SSIM value (1-SSIM_loss)
        self.assertAlmostEqual(1 - ssim_loss.item(), 0.9639027981846681, delta = eps)


    def test_psnr_loss(self):
        preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        psnr_loss = psnr_based_loss(preds, target)
        cv_psnr_loss = cv2.PSNR(np.float32(preds), np.float32(target), R = 1.)
        self.assertAlmostEqual(psnr_loss, cv_psnr_loss, delta=eps)
        # self.assertAlmostEqual(psnr_loss, 2.5527/1., delta=eps)



    def test_psnr_loss2(self):
        # example PSNR values taken from https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        # https://stackoverflow.com/questions/59253688/incorrect-results-for-psnr-calculation
        image1 = Image.open(os.path.join(base_path, "PSNR-example-base.png"))
        image2 = Image.open(os.path.join(base_path, "PSNR-example-comp-10.jpg"))

        image1 = cv2.cvtColor(np.float32(image1), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(np.float32(image2), cv2.COLOR_RGB2BGR)

        cv_psnr_loss = cv2.PSNR(image1, image2, R = 1.)

        preds = transform(image1)
        target = transform(image2)
        
        psnr_loss = psnr_based_loss(preds, target)

        self.assertAlmostEqual(psnr_loss, cv_psnr_loss, delta=eps)
    
    def test_psnr_loss3(self):
        image1 = Image.open(os.path.join(base_path, "PSNR-example-base.png"))
        image2 = Image.open(os.path.join(base_path, "PSNR-example-comp-30.jpg"))

        image1 = cv2.cvtColor(np.float32(image1), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(np.float32(image2), cv2.COLOR_RGB2BGR)

        cv_psnr_loss = cv2.PSNR(image1, image2, R = 1.)

        preds = transform(image1)
        target = transform(image2)
        
        psnr_loss = psnr_based_loss(preds, target)

        self.assertAlmostEqual(psnr_loss, cv_psnr_loss, delta=eps)
        
    def test_psnr_loss4(self):
        image1 = Image.open(os.path.join(base_path, "PSNR-example-base.png"))
        image2 = Image.open(os.path.join(base_path, "PSNR-example-comp-90.jpg"))

        image1 = cv2.cvtColor(np.float32(image1), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(np.float32(image2), cv2.COLOR_RGB2BGR)

        cv_psnr_loss = cv2.PSNR(image1, image2, R = 1.)

        preds = transform(image1)
        target = transform(image2)
        
        psnr_loss = psnr_based_loss(preds, target)

        self.assertAlmostEqual(psnr_loss, cv_psnr_loss, delta=eps)

if __name__ == '__main__':
    unittest.main()



        