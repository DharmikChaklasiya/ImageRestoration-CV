
from model_handler_user import UserModelHandler

def main():
    model_type = 'restormer'
    model_file_name = 'restormer_model.pth'

    # Restormer is trained on  focal planes of 0,20,70 cm , if you provide just one image it is multiplicated 3x

    input_img_folder  = 'input_images/'
    output_folder = 'results'

    # Initialize model handler
    model_handler = UserModelHandler(model_type=model_type, model_file_name=model_file_name)

    # Load inputs
    inputs = model_handler.load_inputs(input_img_folder)

    # Process inputs   
    output = model_handler.process_inputs(inputs.unsqueeze(0)).detach()

    # Save output
    model_handler.save_output(output.squeeze(0,1),folder=output_folder)

if __name__ == '__main__':
    main()