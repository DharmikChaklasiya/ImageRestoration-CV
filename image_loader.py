import os
import torch
from torchvision import datasets, transforms
from image_group import ImageGroup
from PIL import Image


class ImageTensorGroup:
    def __init__(self, image_group, focal_stack_indices):
        self.ground_truth_tensor = None
        self.image_tensors = None
        self.image_group = image_group
        self.focal_stack_indices = focal_stack_indices  # Indices of the images in the focal stack to load
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Add any additional transformations here
        ])

    def load_images(self):
        for index in self.focal_stack_indices:
            image_path = self.image_group.output_image_name(index)
            image = Image.open(image_path)
            self.image_tensors.append(self.transform(image))

        self.ground_truth_tensor = self.transform(Image.open(self.image_group.original_ground_truth_file))

    def get_images(self):
        return torch.stack(self.image_tensors), self.ground_truth_tensor


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_groups, focal_stack_indices):
        self.image_tensor_groups = [ImageTensorGroup(img_group, focal_stack_indices) for img_group in image_groups]

    def __getitem__(self, index):
        img_tensor_group = self.image_tensor_groups[index]
        img_tensor_group.load_images()
        return img_tensor_group.get_images()

    def __len__(self):
        return len(self.image_tensor_groups)


# Define the root directory
root_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "computervision", "integrals"))

print(f"Will load data from rootdir: {root_dir}")

# Create a list to store the image file paths
all_image_groups = []

# Iterate over the subfolders
for root, dirs, files in os.walk(root_dir):
    for dir_name in dirs:
        if dir_name.startswith("Part"):
            subfolder_path = os.path.join(root, dir_name)

            for formatted_image_index in os.listdir(subfolder_path):
                img_group = ImageGroup(int(formatted_image_index))
                img_group.initialize_output_only(os.path.join(subfolder_path, formatted_image_index))
                all_image_groups.append(img_group)


dataset = CustomDataset(all_image_groups, [0, 15, 30])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Successfully loaded images - len : {len(dataloader)}")
