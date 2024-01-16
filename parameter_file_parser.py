import math
import re
from enum import Enum
from typing import Optional


class PersonShape(Enum):
    LAYING = 'laying'
    SITTING = 'sitting'
    IDLE = 'idle'
    NO_PERSON = 'no person'

    @property
    def size_factor(self):
        if self == PersonShape.LAYING:
            return 0.05
        elif self == PersonShape.SITTING:
            return 0.08
        elif self == PersonShape.IDLE:
            return 0.075
        elif self == PersonShape.NO_PERSON:
            return 1.0
        else:
            raise ValueError("Invalid PersonShape" + self.value)

    @property
    def offset_x(self):
        if self == PersonShape.LAYING:
            return 0.0
        elif self == PersonShape.SITTING:
            return 0.01
        elif self == PersonShape.IDLE:
            return 0.0
        elif self == PersonShape.NO_PERSON:
            return 0.0
        else:
            raise ValueError("Invalid PersonShape: " + self.value)

    @property
    def offset_y(self):
        if self == PersonShape.LAYING:
            return 0.0
        elif self == PersonShape.SITTING:
            return 0.0
        elif self == PersonShape.IDLE:
            return -0.005
        elif self == PersonShape.NO_PERSON:
            return 0.0
        else:
            raise ValueError("Invalid PersonShape: " + self.value)

    @property
    def width_to_height_factor(self):
        if self == PersonShape.LAYING:
            return 2.545454
        elif self == PersonShape.SITTING:
            return 1.143
        elif self == PersonShape.IDLE:
            return 0.5517
        elif self == PersonShape.NO_PERSON:
            return 1.0
        else:
            raise ValueError("Invalid PersonShape: " + self.value)

    @staticmethod
    def from_string(string_value) -> 'PersonShape':
        if string_value in PersonShape._value2member_map_:
            return PersonShape._value2member_map_[string_value]
        else:
            raise ValueError(f"{string_value} is not a valid value for PersonShape.")


class ImageDimension:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def validate(self):
        assert self.width > 0, "Image dimension is of wrong width : {}".format(self.width)
        assert self.height > 0, "Image dimension is of wrong height : {}".format(self.height)


class BoundingBox:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min: int = int(x_min)
        self.y_min: int = int(y_min)
        self.x_max: int = int(x_max)
        self.y_max: int = int(y_max)

    def correct_the_box(self, img_dim: ImageDimension):
        self.x_min = max(0, min(self.x_min, img_dim.width - 1))
        self.y_min = max(0, min(self.y_min, img_dim.height - 1))
        self.x_max = max(0, min(self.x_max, img_dim.width - 1))
        self.y_max = max(0, min(self.y_max, img_dim.height - 1))
        return self

    def assert_validity(self, img_dim: ImageDimension = None) -> 'BoundingBox':
        assert self.x_min >= 0, "BoundingBox x_min has wrong value : {}".format(self.x_min)
        assert self.x_max > self.x_min, "BoundingBox x_max has wrong value : {}, x_min: {}".format(self.x_max,
                                                                                                   self.x_min)
        assert self.y_min >= 0, "BoundingBox y_min has wrong value : {}".format(self.y_min)
        assert self.y_max > self.y_min, "BoundingBox y_max has wrong value : {}, y_min: {}".format(self.y_max,
                                                                                                   self.y_min)

        if img_dim is not None:
            assert self.x_max < img_dim.width, "BoundingBox x_max has wrong value : {}, img width: {}".format(
                self.x_max, img_dim.width)
            assert self.y_max < img_dim.height, "BoundingBox y_max has wrong value : {}, img height: {}".format(
                self.y_max, img_dim.height)

        return self

    def is_valid(self, img_dim: ImageDimension = None):
        valid: bool = self.x_max > self.x_min >= 0 and self.y_max > self.y_min >= 0
        valid |= valid if img_dim is None else self.x_max < img_dim.width and self.y_max < img_dim.height
        return valid

    def __iter__(self):
        return iter((self.x_min, self.y_min, self.x_max, self.y_max))

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            return NotImplemented

        return (self.x_min == other.x_min and
                self.y_min == other.y_min and
                self.x_max == other.x_max and
                self.y_max == other.y_max)

    def __repr__(self):
        return f"BoundingBox(x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max})"


def calculate_bounding_box(person_shape: PersonShape, x, y, rotz, image_size=512) -> BoundingBox:
    if person_shape == PersonShape.NO_PERSON:
        return BoundingBox(0.0, 0.0, image_size - 1, image_size - 1)

    person_center_x = image_size / 2 - 16.7 * y
    person_center_y = image_size / 2 - 16.7 * x

    original_person_width = image_size * person_shape.size_factor * person_shape.width_to_height_factor
    original_person_height = image_size * person_shape.size_factor

    person_width, person_height = calculate_adjusted_person_size(original_person_width, original_person_height, rotz)

    person_offset_x = person_shape.offset_x * image_size
    person_offset_y = person_shape.offset_y * image_size

    # Calculate the coordinates of the bounding box
    left = int(max(person_center_x - person_width / 2 + person_offset_x, 0))
    right = int(min(person_center_x + person_width / 2 + person_offset_x, image_size))
    top = int(max(person_center_y - person_height / 2 + person_offset_y, 0))
    bottom = int(min(person_center_y + person_height / 2 + person_offset_y, image_size))

    return BoundingBox(left, top, right, bottom).assert_validity(ImageDimension(image_size, image_size))


def calculate_adjusted_person_size(original_width: float, original_height: float, rotz: float) -> (float, float):
    cosine_angle = abs(math.cos(rotz))
    sine_angle = abs(math.sin(rotz))

    apparent_width = cosine_angle * original_width + sine_angle * original_height
    apparent_height = sine_angle * original_width + cosine_angle * original_height

    return apparent_width, apparent_height


def process_content(parameter_file, x_max_value=10.0, y_max_value=10.0):
    with open(parameter_file, 'r') as file:
        lines = file.readlines()

    rotz, shape_encoded, x, y = process_content_lines(lines, parameter_file)

    # epsilon = 1e-3
    # expected_person_shape = PersonShape.NO_PERSON
    # if shape_encoded == expected_person_shape:
    #    print(expected_person_shape.value + " person with interesting coords: " + parameter_file)

    assert x is not None, "No person pose x found in : " + parameter_file
    assert y is not None, "No person pose y found in : " + parameter_file
    if not shape_encoded:
        raise ValueError("No valid person shape found in : " + parameter_file)
    return shape_encoded, x, y, rotz


def process_content_lines(lines, parameter_file):
    x, y, z, rotx, roty, rotz = None, None, None, None, None, None
    shape_encoded: Optional[PersonShape] = None
    for line in lines:
        if line.startswith("person pose"):
            match = re.search(
                        r'person pose \(x,y,z,rot x, rot y, rot z\) ='
                            r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
                line)

            if match:
                # the picture coordinates do not fill the whole space, whyever that is the case!
                x, y, z, rotx, roty, rotz = (int(match.group(1)),  # / x_max_value,
                                             int(match.group(2)),  # / y_max_value * 1.09,
                                             float(match.group(3)),
                                             float(match.group(4)),
                                             float(match.group(5)),
                                             float(match.group(6))
                                             )
            elif "no person" in line:
                x, y = -1.1, -1.1
                shape_encoded = PersonShape.NO_PERSON
            else:
                raise ValueError("x,y invalid for file: " + parameter_file)

        if line.startswith("person shape"):
            shape_match = re.search(r'person shape =\s*(\w+)', line)

            if shape_match:
                shape_encoded = PersonShape.from_string(shape_match.group(1))
            else:
                raise ValueError("Invalid person shape in file: " + parameter_file)

            if not shape_encoded:
                raise ValueError("Invalid person shape in file: " + parameter_file + ", line : " + line)
    return rotz, shape_encoded, x, y
