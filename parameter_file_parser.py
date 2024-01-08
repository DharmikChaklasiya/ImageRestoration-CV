import math
import re
from enum import Enum


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
            raise ValueError("Invalid PersonShape"+self.value)



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
            raise ValueError("Invalid PersonShape: "+self.value)

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
            raise ValueError("Invalid PersonShape: "+self.value)

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
            raise ValueError("Invalid PersonShape: "+self.value)


    @staticmethod
    def from_string(string_value) -> 'PersonShape':
        if string_value in PersonShape._value2member_map_:
            return PersonShape._value2member_map_[string_value]
        else:
            raise ValueError(f"{string_value} is not a valid value for PersonShape.")


def calculate_bounding_box(person_shape: PersonShape, x, y, rotz, image_size=512):
    if(person_shape == PersonShape.NO_PERSON):
        return 0.0, 0.0, image_size-1, image_size-1

    person_center_x = image_size / 2 - 16.7 * y
    person_center_y = image_size / 2 - 16.7 * x

    original_person_width = image_size * person_shape.size_factor * person_shape.width_to_height_factor
    original_person_height = image_size * person_shape.size_factor

    person_width, person_height = calculate_adjusted_person_size(original_person_width, original_person_height, rotz)

    person_offset_x = person_shape.offset_x * image_size
    person_offset_y = person_shape.offset_y * image_size

    # Calculate the coordinates of the bounding box
    left = max(person_center_x - person_width / 2 + person_offset_x, 0)
    right = min(person_center_x + person_width / 2 + person_offset_x, image_size)
    top = max(person_center_y - person_height / 2 + person_offset_y, 0)
    bottom = min(person_center_y + person_height / 2 + person_offset_y, image_size)

    return left, top, right, bottom


def calculate_adjusted_person_size(original_width: float, original_height: float, rotz: float) -> (float, float):
    cosine_angle = abs(math.cos(rotz))
    sine_angle = abs(math.sin(rotz))

    apparent_width = cosine_angle * original_width + sine_angle * original_height
    apparent_height = sine_angle * original_width + cosine_angle * original_height

    return apparent_width, apparent_height


def process_content(parameter_file, x_max_value=10.0, y_max_value=10.0):
    with open(parameter_file, 'r') as file:
        lines = file.readlines()

    x, y, z, rotx, roty, rotz = None, None, None, None, None, None
    shape_encoded: PersonShape = None
    for line in lines:
        if line.startswith("person pose"):
            match = re.search(
                r'person pose \(x,y,z,rot x, rot y, rot z\) =\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
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

    #epsilon = 1e-3
    #expected_person_shape = PersonShape.NO_PERSON
    #if shape_encoded == expected_person_shape:
    #    print(expected_person_shape.value + " person with interesting coords: " + parameter_file)

    assert x is not None, "No person pose x found in : " + parameter_file
    assert y is not None, "No person pose y found in : " + parameter_file
    if not shape_encoded:
        raise ValueError("No valid person shape found in : " + parameter_file)
    return shape_encoded, x, y, rotz
