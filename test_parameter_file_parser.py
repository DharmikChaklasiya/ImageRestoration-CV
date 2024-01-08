import math
import os
from unittest import TestCase
from PIL import Image
from matplotlib import pyplot as plt, patches

from parameter_file_parser import process_content, calculate_bounding_box, calculate_adjusted_person_size

class Test(TestCase):
    def test_calculate_bounding_box(self):
        base_path = "C:\\Users\\marti\\Documents\\computervision\\integrals\\Part1\\"
        formatted_image_index = "003186"  #000790 003186 004748 004921 idle samples no rotation
        formatted_image_index = "002509"  # 000503 002509 sitting samples no rotation
        formatted_image_index = "003029"  # 003029 laying samples no rotation
        formatted_image_index = "001042"  # 000214 000333 000361 000566 001015 001019 001042 diagonal idle samples
        formatted_image_index = "003099"  # 001592 001709 002795 002890 002976 003037 003099 003398 003404 003490 003556 diagonal laying samples
        formatted_image_index = "000004"  # 000004 no person samples

        shape_encoded, x, y, rotz = process_content(
            os.path.join(base_path, formatted_image_index, formatted_image_index + "_params.txt"))

        # Calculate bounding box
        bounding_box = calculate_bounding_box(shape_encoded, x, y, rotz)

        # Open the image to draw the bounding box on
        image = Image.open(os.path.join(base_path, formatted_image_index, formatted_image_index + "_gt.png"))

        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(image)

        print(bounding_box)

        # Create a rectangle patch for the bounding box and add it to the axes
        rect = patches.Rectangle(
            (bounding_box[0], bounding_box[1]),
            bounding_box[2] - bounding_box[0],
            bounding_box[3] - bounding_box[1],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Hide the axis
        ax.axis('off')

        # Show the result
        plt.show()

    def test_calculate_adjusted_width_height_factor(self):
        adjusted_width, adjusted_height = calculate_adjusted_person_size(1.0, 2.0, 0)
        self.assertAlmostEqual(1.0, adjusted_width)
        self.assertAlmostEqual(2.0, adjusted_height)

        adjusted_width, adjusted_height = calculate_adjusted_person_size(1.0, 2.0, math.pi/2)
        self.assertAlmostEqual(2.0, adjusted_width)
        self.assertAlmostEqual(1.0, adjusted_height)

    def test_process_content(self):
        self.fail()
