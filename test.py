
from model_handler_user import UserModelHandler
from loss_utils import psnr_value,ssim_value,plot_metrics
from PIL import Image
import os

def main():
    model_type = 'restormer'
    model_file_name = '/weights/restormer_model.pth'
    # Restormer is trained on  focal planes of 0,20,70 cm , if you provide just one image it is multiplicated 3x
    
    input_img_folder  = '/input_images/'
    output_folder = '/Results'
    
     # Initialize model handler
    model_handler = UserModelHandler(model_type=model_type, model_file_name=model_file_name)

  
    psnr_scores = []
    ssim_scores= []

    for subdir, dirs, files in os.walk(input_img_folder):
        dirs = sorted(dirs)
        for dir in dirs:          
              
            if dir[0] == '.':
                continue

        

           
    

            # Load inputs
            input,gt,bool_gt = model_handler.load_inputs(f'{subdir}/{dir}')

            # Process inputs   
            output = model_handler.process_inputs(input.unsqueeze(0)).detach()

            # Save output
            model_handler.save_output(output.squeeze(0,1),folder=f'{output_folder}/{dir}_pred.jpg')

            if bool_gt:
                output_image = Image.open(f'{output_folder}/{dir}_pred.jpg').convert('L')
                psnr_score = psnr_value(output_image, gt)
                ssim_score = ssim_value(output_image, gt)
               


                psnr_scores.append(psnr_score)
                ssim_scores.append(ssim_score)


    if len(psnr_scores) > 0:
        plot_metrics(psnr_scores,ssim_scores,output_folder)


if __name__ == '__main__':
    main()
