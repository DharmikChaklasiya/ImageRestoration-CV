import os


class ImageGroup:
    prefix_digit = 100000

    @staticmethod
    def calc_full_image_index(image_prefix, image_index):
        assert image_prefix is not None
        assert image_index is not None
        assert 10 > int(image_prefix) >= 0, "Image prefix {} must be a single digit".format(image_prefix)
        prefix_num = int(image_prefix)*ImageGroup.prefix_digit
        image_index_num = int(image_index)
        assert image_index_num < ImageGroup.prefix_digit, (("Image index {} must be less than {} or our algorithm of "
                                                           "combining indices won't work")
                                                           .format(image_index_num, ImageGroup.prefix_digit))
        return prefix_num+image_index_num

    def __init__(self, image_index):
        assert image_index is not None
        assert isinstance(image_index, int)
        assert image_index >= 0
        self.image_index = image_index
        if self.image_index >= self.prefix_digit:
            prefix = (self.image_index // self.prefix_digit) * self.prefix_digit
            self.image_index_without_prefix = self.image_index - prefix
        else:
            self.image_index_without_prefix = self.image_index
        self.formatted_image_index = str(image_index).zfill(len(str(self.prefix_digit)))
        self.filenames = []
        self.base_output_path = None
        self.original_ground_truth_file = None
        self.new_ground_truth_file = None
        self.original_parameter_file = None
        self.new_parameter_file = None
        self.valid = True
        self.invalid_reason = None
        self.invalid_reason_filenames = None
        self.output_file_names = []

    def add_filename(self, full_filename):
        self.filenames.append(full_filename)

    def _initialize_filenames(self, full_filename, base_output_path):
        leading_digit = os.path.basename(full_filename).split('_')[0]
        base_path = os.path.dirname(full_filename)
        self.base_output_path = os.path.join(base_output_path, self.formatted_image_index)
        self.original_ground_truth_file = os.path.join(base_path, f"{leading_digit}_{self.image_index_without_prefix}_GT_pose_0_thermal.png")
        self.new_ground_truth_file = os.path.join(self.base_output_path, f"{self.formatted_image_index}_gt.png")
        self.original_parameter_file = os.path.join(base_path, f"{leading_digit}_{self.image_index_without_prefix}_Parameters.txt")
        self.new_parameter_file = os.path.join(self.base_output_path, f"{self.formatted_image_index}_params.txt")

    def initialize_and_validate(self, base_output_path):
        assert base_output_path is not None and isinstance(base_output_path, str)
        self._initialize_filenames(self.filenames[0], base_output_path)
        if not os.path.exists(self.original_ground_truth_file):
            self.valid = False
            self.invalid_reason = "" if self.invalid_reason is None else self.invalid_reason
            self.invalid_reason += f"Could not find ground truth file. "
            self.invalid_reason_filenames = [] if self.invalid_reason_filenames is None else self.invalid_reason_filenames
            self.invalid_reason_filenames.append(self.original_ground_truth_file)
        if not os.path.exists(self.original_parameter_file):
            self.valid = False
            self.invalid_reason = "" if self.invalid_reason is None else self.invalid_reason
            self.invalid_reason += f"Could not find parameter file. "
            self.invalid_reason_filenames = [] if self.invalid_reason_filenames is None else self.invalid_reason_filenames
            self.invalid_reason_filenames.append(self.original_parameter_file)
        if len(self.filenames) < 11:
            self.valid = False
            self.invalid_reason = "" if self.invalid_reason is None else self.invalid_reason
            self.invalid_reason += f"Expected 11 images but got less. "
            self.invalid_reason_filenames = [] if self.invalid_reason_filenames is None else self.invalid_reason_filenames
            self.invalid_reason_filenames.append(self.filenames)
        if len(self.filenames) > 11:
            self.valid = False
            self.invalid_reason = "" if self.invalid_reason is None else self.invalid_reason
            self.invalid_reason += f"Expected 11 images but got more. "
            self.invalid_reason_filenames = [] if self.invalid_reason_filenames is None else self.invalid_reason_filenames
            self.invalid_reason_filenames.append(self.filenames)

    def output_image_name(self, focal_stack_img_index):
        formatted_focal_stack_img_index = str(focal_stack_img_index).zfill(2)
        return os.path.join(self.base_output_path, f"{self.formatted_image_index}_{formatted_focal_stack_img_index}.png")

    def initialize_output_only(self, base_output_path, focal_stack_indices):
        assert base_output_path is not None and isinstance(base_output_path, str)
        self.base_output_path = os.path.join(base_output_path, self.formatted_image_index)
        self.new_ground_truth_file = os.path.join(self.base_output_path, f"{self.formatted_image_index}_gt.png")
        self.new_parameter_file = os.path.join(self.base_output_path, f"{self.formatted_image_index}_params.txt")

        self.valid = True
        for focal_stack_idx in focal_stack_indices:
            output_img_name = self.output_image_name(focal_stack_idx)
            self.output_file_names.append(output_img_name)
            if not os.path.exists(output_img_name):
                self.valid = False
                break