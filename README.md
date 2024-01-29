# COMPUTER VISION IMAGE RESTORATION PROJECT D7


- conda env create -f environment.yml

- conda activate compvis-model

If you have GPU support locally, make sure you have cuda installed 
(nvcc --version should emit the currently installed version, if not installed, go to nvidia to install)
and execute something like the following adapted to your environment
(from the pytorch getting started page):

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


## Running test.py 

Variables to be set by the user:
  
    input_img_folder  = '/test_images/'
    output_folder = '/Results'

- "/test_images" - folder, where you can store your images in subfolders that you want to get tested /see example in the "test_images" folder/. It would produce an output that is saved in "Results" folder. 
- "/Results" - folder that saves the all the output predicted images

1. Inputs - focal stacks stored in folder as subfolders /as explained above/. Test.py takes the variable input_img_folder. There are two cases here: if you want to only test the model, then you should not provide GT image and it will just output a predicted image in the "Result" folder. If you have GT images in the same directory, as the focal stacks, named "..._gt", then it will save the predicted output of those images and retrieve a plot with PSNR and SSIM scores.
2. Outputs - files are outputed in "Result" which is provided with the same name as the folder of the focal stack and _pred prefix. Also a plot of metrics is saved as png.

### Github - Full Package Code
https://github.com/DharmikChaklasiya/ImageRestoration-CV.git

The .zip file contains only the main files that you need in order to run our main Restormer model. Since we have started with a U-Net model, bounding boxes and many additional implementations, we are providing the Github to our full code /not only for our main model - Restormer, but also all of the code that we have been working on for the past 2 months, including U-Net, bounding boxes, etc./. 
