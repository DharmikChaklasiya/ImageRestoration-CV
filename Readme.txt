Running test.py :
Variables to be set by the user:

    model_type = 'restormer'
    model_file_name = '/ImageRestoration-CV/restormer_model.pth'
    input_img_folder  = '/input_images/'
    output_folder = '/Results'
    
1. Inputs - focal stacks stored in folder as subfolders test.py takes the variable input_img_folder. If there is a ground truth image in some focal stacks then also metrics are calculated and saved.	Our model uses images at focal planes 0,20,70 cm. If more images are provided in each focal stack we dont guarantee that precisely those planes are taken.	If only one image is provided it is multiplicated 3 times.

2. Outputs - files are outputed in output_folder which is provided with the same name as the folder of the focal stack and _pred prefix. Also a plot of metrics is saved as png.
