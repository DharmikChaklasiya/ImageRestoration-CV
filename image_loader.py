import os
import torch
from torchvision import datasets, transforms
from PIL import Image

from LFR.python.image_group import ImageGroup


class ImageTensorGroup:
    def __init__(self, image_group: ImageGroup, focal_stack_indices):
        self.ground_truth_tensor = None
        self.image_tensors = []
        self.image_group = image_group
        self.focal_stack_indices = focal_stack_indices  # Indices of the images in the focal stack to load
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Add any additional transformations here
        ])

    def load_images(self):
        self.image_tensors = []
        for index in self.focal_stack_indices:
            image_path = self.image_group.output_image_name(index)
            image = Image.open(image_path).convert('L')
            self.image_tensors.append(self.transform(image))

        try:
            opened_image = Image.open(self.image_group.new_ground_truth_file).convert('L')
        except Exception as e:
            raise ValueError(f"Error while loading image from: {self.image_group.new_ground_truth_file}") from e
        self.ground_truth_tensor = self.transform(opened_image)

    def get_images(self):
        stacked_images = torch.cat(self.image_tensors, dim=0)  # Shape: (3, 512, 512)
        assert stacked_images.shape[0] == 3, ("Issue while processing image_group : {} - dimension 1 should be 3, "
                                              "but is {}").format(self.image_group.formatted_image_index,
                                                                  stacked_images.shape[0])
        assert stacked_images.shape[1] == 512, ("Issue while processing image_group : {} - dimension 2 should be 512, "
                                                "but is {}").format(self.image_group.formatted_image_index,
                                                                    stacked_images.shape[1])
        assert stacked_images.shape[2] == 512, ("Issue while processing image_group : {} - dimension 3 should be 512, "
                                                "but is {}").format(self.image_group.formatted_image_index,
                                                                    stacked_images.shape[2])
        return stacked_images, self.ground_truth_tensor, self.image_group.formatted_image_index


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_groups, focal_stack_indices):
        self.image_tensor_groups = [ImageTensorGroup(img_group, focal_stack_indices) for img_group in image_groups]

    def __getitem__(self, index):
        img_tensor_group = self.image_tensor_groups[index]
        img_tensor_group.load_images()
        images = img_tensor_group.get_images()
        return images

    def __len__(self):
        return len(self.image_tensor_groups)


# Define the root directory
root_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "computervision", "integrals"))

print(f"Will load data from rootdir: {root_dir}")

# Create a list to store the image file paths
all_image_groups = []
image_group_map = {}

focal_stack_indices = [0, 15, 30]

# Iterate over the subfolders
for root, dirs, files in os.walk(root_dir):
    for dir_name in dirs:
        if dir_name == "Part1":
            subfolder_path = os.path.join(root, dir_name)

            for formatted_image_index in os.listdir(subfolder_path):
                img_group = ImageGroup(int(formatted_image_index))
                img_group.initialize_output_only(subfolder_path, focal_stack_indices)
                if img_group.valid:
                    all_image_groups.append(img_group)
                    image_group_map[img_group.formatted_image_index] = img_group

                #if len(all_image_groups) > 100:
                   # break

all_image_groups = sorted(all_image_groups, key=lambda img_group: int(img_group.formatted_image_index))

dataset = CustomDataset(all_image_groups, focal_stack_indices)

print(f"Successfully loaded image-batches - len : {len(dataset)}")


def get_dataset():
    return dataset


def get_image_group_map():
    return image_group_map
