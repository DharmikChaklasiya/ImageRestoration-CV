import os
import re
from typing import List

import torch
from torchvision import datasets, transforms
from PIL import Image

from LFR.python.image_group import ImageGroup


class ImageTensorGroup:
    def __init__(self, image_group: ImageGroup, focal_stack_indices):
        self.pose_prediction_labels = None
        self.ground_truth_tensor = None
        self.image_tensors = []
        self.image_group = image_group
        self.focal_stack_indices = focal_stack_indices  # Indices of the images in the focal stack to load
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Add any additional transformations here
        ])
        self.shape_mapping = {'laying': [1, 0, 0, 0, 0], 'sitting': [0, 1, 0, 0, 0], 'standing': [0, 0, 1, 0, 0],
                              'idle': [0, 0, 0, 1, 0], 'no person': [0, 0, 0, 0, 1]}

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

        with open(self.image_group.new_parameter_file, 'r') as file:
            shape_encoded, x, y = self.parse_param_file(self.image_group.new_parameter_file, file.readlines())

        self.pose_prediction_labels = torch.tensor([x])  # torch.tensor(shape_encoded + [x, y]) we start slowly - predicting everything sadly doesn't work at all

    def get_images(self):
        stacked_images = images_to_tensors(self.image_tensors, self.image_group.formatted_image_index)
        return stacked_images, self.ground_truth_tensor, self.image_group.formatted_image_index

    def get_images_with_pose_prediction_labels(self):
        stacked_images = images_to_tensors(self.image_tensors, self.image_group.formatted_image_index)
        return stacked_images, self.pose_prediction_labels, self.image_group.formatted_image_index

    def parse_param_file(self, parameter_file, lines):
        shape_encoded, x, y = self.process_content(lines, parameter_file)
        assert x is not None, "No person pose x found in : " + parameter_file
        assert y is not None, "No person pose y found in : " + parameter_file
        if not shape_encoded:
            self.process_content(lines, parameter_file)
            raise ValueError("No valid person shape found in : " + parameter_file)
        return shape_encoded, x, y

    def process_content(self, lines, parameter_file, x_max_value=10.0, y_max_value=10.0):
        x, y = None, None
        shape_encoded = None
        for line in lines:
            if line.startswith("person pose"):
                match = re.search(r'person pose \(x,y,z,rot x, rot y, rot z\) =\s*([-\d.]+)\s+([-\d.]+)', line)

                if match:
                    x, y = float(match.group(1)) / x_max_value * 0.7, float(match.group(2)) / y_max_value * 0.7 #the picture coordinates do not fill the whole space, whyever that is the case!
                elif "no person" in line:
                    x, y = 0.0, 0.0
                    shape_encoded = self.shape_mapping.get("no person", None)
                else:
                    raise ValueError("x,y invalid for file: " + parameter_file)

            if line.startswith("person shape"):
                shape_match = re.search(r'person shape =\s*(\w+)', line)

                if shape_match:
                    shape = shape_match.group(1)
                    shape_encoded = self.shape_mapping.get(shape, None)
                else:
                    raise ValueError("Invalid person shape in file: " + parameter_file)

                if not shape_encoded:
                    raise ValueError("Invalid person shape in file: " + parameter_file + ", value : " + shape)
        return shape_encoded, x, y


def images_to_tensors(image_tensors, image_index):
    stacked_images = torch.cat(image_tensors, dim=0)  # Shape: (3, 512, 512)
    assert stacked_images.shape[0] == 3, ("Issue while processing image_group : {} - dimension 1 should be 3, "
                                          "but is {}").format(image_index,
                                                              stacked_images.shape[0])
    assert stacked_images.shape[1] == 512, ("Issue while processing image_group : {} - dimension 2 should be 512, "
                                            "but is {}").format(image_index,
                                                                stacked_images.shape[1])
    assert stacked_images.shape[2] == 512, ("Issue while processing image_group : {} - dimension 3 should be 512, "
                                            "but is {}").format(image_index,
                                                                stacked_images.shape[2])
    return stacked_images


class GroundTruthLabelDataset(torch.utils.data.Dataset):
    def __init__(self, image_tensor_groups: List[ImageTensorGroup]):
        self.image_tensor_groups = image_tensor_groups

    def __getitem__(self, index):
        img_tensor_group = self.image_tensor_groups[index]
        return img_tensor_group.get_images()

    def __len__(self):
        return len(self.image_tensor_groups)


class PosePredictionLabelDataset(torch.utils.data.Dataset):
    def __init__(self, image_tensor_groups: List[ImageTensorGroup]):
        self.image_tensor_groups = image_tensor_groups

    def __getitem__(self, index):
        img_tensor_group: ImageTensorGroup = self.image_tensor_groups[index]
        return img_tensor_group.get_images_with_pose_prediction_labels()

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

all_image_groups = sorted(all_image_groups, key=lambda img_group: int(img_group.formatted_image_index))

print(f"Successfully loaded image-batches - len : {len(all_image_groups)}")


def get_image_group_map():
    return image_group_map


def get_sorted_image_groups():
    return all_image_groups
