import os
import torch
from torchvision import datasets, transforms

# Define the root directory
root_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "computervision", "integrals"))

# Create a list to store the image file paths
image_file_paths = []

# Iterate over the subfolders
for root, dirs, files in os.walk(root_dir):
    for dir_name in dirs:
        if dir_name.startswith("part"):
            subfolder_path = os.path.join(root, dir_name)
            
            # Get the image file paths in the subfolder
            image_files = [file for file in os.listdir(subfolder_path) if file.endswith(".png") and not file.endswith("_gt.png")]
            
            # Sort the image files based on the pose index
            image_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            
            # Create a dictionary to store the image file paths and ground truth file path
            image_data = {}
            
            # Iterate over the image files
            for file in image_files:
                image_index = file.split("_")[0]
                image_path = os.path.join(subfolder_path, file)
                image_data[image_index] = image_path
            
            # Get the ground truth image file path
            ground_truth_file = f"{dir_name}_gt.png"
            ground_truth_file_path = os.path.join(subfolder_path, ground_truth_file)
            
            # Add the ground truth file path to the image data dictionary
            image_data["gt"] = ground_truth_file_path
            
            # Append the image data dictionary to the list
            image_file_paths.append(image_data)

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add more transformations if needed
])

# Create a custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_file_paths, transform=None):
        self.image_file_paths = image_file_paths
        self.transform = transform
    
    def __getitem__(self, index):
        image_data = self.image_file_paths[index]
        images = []
        
        # Load the image files
        for image_index, image_path in image_data.items():
            if image_index != "gt":
                image = Image.open(image_path)
                if self.transform is not None:
                    image = self.transform(image)
                images.append(image)
        
        # Load the ground truth image file
        ground_truth_image = Image.open(image_data["gt"])
        if self.transform is not None:
            ground_truth_image = self.transform(ground_truth_image)
        
        return torch.stack(images), ground_truth_image
    
    def __len__(self):
        return len(self.image_file_paths)

# Create a custom dataset instance
dataset = CustomDataset(image_file_paths, transform=transform)

# Create a data loader to load the dataset in batches
batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
