import argparse
import torch

from base_model_training import load_dataset_infos
from unet_architecture import UNet

from unet_inner_model_training import train_model_on_one_batch

def comma_separated_list(values):
    return values.split(',')

def main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--batch_size', dest = 'batch_size',required = False,help = 'Load the data in mini batches chose how many focal stack groups to load on one go')
    arg_parser.add_argument('--parts', type=comma_separated_list, help='A comma-separated list of values, list the Parts you want to load for training')
    arg_parser.add_argument('--super_batch', dest = 'super_batch',required = False)
    arg_parser.add_argument('--valid_size', dest = 'valid_size',required = False)
    arg_parser.add_argument('--epochs', dest = 'epochs',required = False)
    arg_parser.add_argument('--path_to_folder', dest = 'path_to_folder',required = True, help = 'The path to the unziped Parts')

    args,_ = arg_parser.parse_known_args()
    
    if args.parts:
        print('List of values:', args.parts)
    else:
        print('No values provided.')

    if args.batch_size is not None:
        train_batch_size = int(args.batch_size)
    else:
        train_batch_size = 4

    if args.super_batch is not None:
        num_super_batches = int(args.batch_size)
    else:
        num_super_batches = 1


    if args.valid_size is not None:
        valid_size = int(args.valid_size)
    else:
        valid_size = 100

    if args.epochs is not None:
        epochs = int(args.epochs)
    else:
        epochs = 50

    path_to_folder = args.path_to_folder
 

    
    print(f"Using pytorch version: {torch.__version__}")
    print(f"Using pytorch cuda version: {torch.version.cuda}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    all_parts = args.parts
    model = UNet(in_channels=10).to(device)

    dataset_parts = load_dataset_infos(all_parts,path_to_folder)


    model_file_name = "unet_model.pth"



    for i in range(1, num_super_batches + 1):
        super_batch_info = f"Super-Batch: {i}/{num_super_batches}"
        print(f"Running the model in super-batches - {super_batch_info}")
        for part, dataset_metainfo in dataset_parts.items():
            train_model_on_one_batch(dataset_metainfo, model, device,super_batch_info,model_file_name,path_to_folder,
                                     train_batch_size,epochs = epochs)

    print("Training complete - printing results.")

if __name__ == "__main__":
    main()