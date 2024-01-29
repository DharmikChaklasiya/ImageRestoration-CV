# user_model_handler.py

from model_handler_abstraction import AbstractModelHandler
from models.unet_architecture import UNet
from models.restormer import Restormer

from performance_visualization import tensor_to_base64

import os
from PIL import Image
from torchvision import transforms
import torch
import base64
from PIL import Image
from io import BytesIO

import wandb



class UserModelHandler(AbstractModelHandler):
    def __init__(self,model_file_name = None ,model_type='unet',device = 'cpu'):
        self.model_type = model_type
        self.model_file_name = model_file_name
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        if self.model_type.lower() == 'unet':
            model = UNet(in_channels=10)
            state = torch.load(self.model_file_name,map_location=torch.device(self.device))
            model.load_state_dict(state['model_state_dict'])     
            self.in_channels = 10      
            return model.eval()
        
        
        if self.model_type.lower() == 'restormer':

            model = Restormer()
            
            state = torch.load(self.model_file_name,map_location=torch.device(self.device))
            model.load_state_dict(state['model_state_dict'])   
            self.in_channels = 3

           
            return model.eval()

        # You may need to adjust the in_channels parameter
        # Add more model types as needed
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


    
    def load_inputs(self,directory_path):

        def load_single_image(input_image):
            image = Image.open(input_image).convert('L')
            image = image.resize((512,512))
        
            transform = transforms.Compose([
                    transforms.ToTensor() 
                ])
            image_tensor = transform(image)
            return image_tensor

        def align_of_inputs(loaded_image_tensors):
            len_input_images = len(loaded_image_tensors)
            
            if  len_input_images == 1:                
                return loaded_image_tensors[0].repeat(self.in_channels,1,1)
            
            elif len_input_images >= self.in_channels: 
                #take images at equal step size

                step_size = len_input_images// self.in_channels
                selected_indeces =  [i * step_size for i in range(self.in_channels)]
                if len(selected_indeces) < self.in_channels:
                    # I am insuring that we have exactly the same input channel size we need
                    for _ in range(self.in_channels - len(selected_indeces) ):
                        selected_indeces = selected_indeces  + [selected_indeces[-1]]

                selected_images = [loaded_image_tensors[i].squeeze(0) for i in range(len_input_images) if i in selected_indeces]         
                  
                return torch.stack(selected_images)
            else:
                raise Exception('Inputs are not comaptible with the model. Provide one or more than three images')
            

        image_list = []
        if not os.path.isdir(directory_path):
            print(f"Error: {directory_path} is not a valid directory.")
            return image_list
        gt = []
        bool_gt = False
        for filename in sorted(os.listdir(directory_path)):  
            if (filename[0]) =='.' or (filename[-3:] =="txt"):
              continue          
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']) :
                if '_gt' in filename:
                    gt =  Image.open(file_path).convert('L')
                    bool_gt = True
                else:
                    image_list.append(load_single_image(file_path))
            else:
                raise Exception('Images are not extension jpg,jpeg,png,gif')
            
        return align_of_inputs(image_list),gt,bool_gt
    


    def process_inputs(self, inputs):
        # Implement processing inputs based on user's requirements
        # Example implementation using a user-defined model
        

        return self.model(inputs)
    

 


    def save_output(self, output,folder):
        # Implement displaying the output based on user's requirements
        # Example implementation converting output tensor to image
        output_image = tensor_to_base64(output)
        decoded_image = base64.b64decode(output_image)
        pil_image = Image.open(BytesIO(decoded_image))
        pil_image.save(folder)
       
