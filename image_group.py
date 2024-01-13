import os
from typing import List

from pydantic import BaseModel


class ImageGroup(BaseModel):
    formatted_image_index: str
    filenames: List[str] = []
    base_path: str = None
    ground_truth_file: str = None
    parameter_file: str = None
    valid: bool = True

    def output_image_name(self, focal_stack_img_index):
        formatted_focal_stack_img_index = str(focal_stack_img_index).zfill(2)
        return os.path.join(self.base_path,
                            f"{self.formatted_image_index}_{formatted_focal_stack_img_index}.png")

    def initialize_output_only(self, base_output_path: str, validate: bool):
        assert base_output_path is not None and isinstance(base_output_path, str)
        self.base_path = os.path.join(base_output_path, self.formatted_image_index)
        self.ground_truth_file = os.path.join(self.base_path, f"{self.formatted_image_index}_gt.png")
        self.parameter_file = os.path.join(self.base_path, f"{self.formatted_image_index}_params.txt")

        for focal_stack_idx in range(0, 30):
            output_img_name = self.output_image_name(focal_stack_idx)
            self.filenames.append(output_img_name)
            if validate and not os.path.exists(output_img_name):
                raise ValueError("No file found for name : " + output_img_name)
