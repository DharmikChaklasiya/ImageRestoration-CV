import os
import json


def remove_losses_from_json(directory, model_name):
    """
    Remove training and validation losses from JSON files in the specified directory for a given model name.
    :param directory: Directory containing JSON files.
    :param model_name: Name of the model (e.g., 'pose_pred_model.pth').
    """
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

            if 'loss_histories' in data and model_name in data['loss_histories']:
                del data['loss_histories'][model_name]

                # Write the modified data back to the file
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)


remove_losses_from_json('dataset_infos', 'pose_pred_model.pth')
