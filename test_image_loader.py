import unittest
from unittest import TestCase

from image_loader import ImageCropper


class RandomParam:
    def __init__(self, descr, actual_value, min_value, max_value):
        self.descr = descr
        self.actual_value = actual_value
        self.min_value = min_value
        assert self.actual_value >= min_value, "min_value must be smaller equal actual value"
        self.max_value = max_value
        assert self.actual_value <= max_value, "max_value must be greater equal actual value"

    def validate(self, min_value, max_value):
        assert min_value == self.min_value and max_value == self.max_value, \
            f"For {self.descr} expected min/max: {self.min_value}/{self.max_value}, but got: {min_value}/{max_value}"
        return self.actual_value


class TestableImageCropper(ImageCropper):
    def __init__(self, width_param, height_param, x_param, y_param):
        super().__init__()
        self.width_param = width_param
        self.height_param = height_param
        self.x_param = x_param
        self.y_param = y_param

    def _new_random_width(self, min_width, max_width):
        return self.width_param.validate(min_width, max_width)

    def _new_random_height(self, min_height, max_height):
        return self.height_param.validate(min_height, max_height)

    def _new_random_x(self, min_x, max_x):
        return self.x_param.validate(min_x, max_x)

    def _new_random_y(self, min_y, max_y):
        return self.y_param.validate(min_y, max_y)


class ImageCropperTests(TestCase):
    def test_random_crop_1(self):
        width_param = RandomParam("width", 100, 50, 100)
        height_param = RandomParam("height", 100, 50, 100)
        x_param = RandomParam("x", 50, 0, 100)
        y_param = RandomParam("y", 50, 0, 100)

        cropper = TestableImageCropper(width_param, height_param, x_param, y_param)
        img_shape = (3, 200, 200)

        x_start, y_start, x_end, y_end = cropper.randomly_crop_image(img_shape, 0, 0, 200, 200)

        # Assertions to validate cropping
        self.assertEqual(50, x_start)
        self.assertEqual(50, y_start)
        self.assertEqual(150, x_end)  # 50 + 100
        self.assertEqual(150, y_end)  # 50 + 100

    def test_random_crop_2(self):
        width_param = RandomParam("width", 120, 80, 160)
        height_param = RandomParam("height", 100, 80, 160)
        x_param = RandomParam("x", 20, 0, 80)
        y_param = RandomParam("y", 50, 40, 100)

        cropper = TestableImageCropper(width_param, height_param, x_param, y_param)
        img_shape = (3, 200, 200)

        # Bounding box that does not cover the entire image
        x_start, y_start, x_end, y_end = cropper.randomly_crop_image(img_shape, 40, 60, 120, 140)

        # Assertions to validate cropping for specific bounding box
        self.assertEqual(20, x_start)
        self.assertEqual(50, y_start)
        self.assertEqual(140, x_end)  # x_start + crop_width
        self.assertEqual(150, y_end)  # y_start + crop_height

    def test_random_crop_lower_right_corner(self):
        # Edge case where bounding box is in the right lower corner (30x40)
        width_param = RandomParam("width", 60, 30, 60)
        height_param = RandomParam("height", 80, 40, 80)
        x_param = RandomParam("x", 140, 140, 140)
        y_param = RandomParam("y", 120, 120, 120)

        cropper = TestableImageCropper(width_param, height_param, x_param, y_param)
        img_shape = (3, 200, 200)

        # Bounding box in the lower right corner
        x_min, y_min, x_max, y_max = 170, 160, 200, 200

        x_start, y_start, x_end, y_end = cropper.randomly_crop_image(img_shape, x_min, y_min, x_max, y_max)

        self.assertEqual(140, x_start)
        self.assertEqual(120, y_start)
        self.assertEqual(200, x_end)
        self.assertEqual(200, y_end)

    def test_random_crop_from_image_003983(self):
        # Edge case where bounding box is in the right lower corner (30x40)
        width_param = RandomParam("width", 70, 61, 122)
        height_param = RandomParam("height", 80, 59, 118)
        x_param = RandomParam("x", 160, 154, 163)
        y_param = RandomParam("y", 100, 88, 109)

        cropper = TestableImageCropper(width_param, height_param, x_param, y_param)
        img_shape = (3, 512, 512)

        # Bounding box in the lower right corner
        x_min, y_min, x_max, y_max = 163, 109, 224, 168

        x_start, y_start, x_end, y_end = cropper.randomly_crop_image(img_shape, x_min, y_min, x_max, y_max)

        self.assertEqual(160, x_start)
        self.assertEqual(100, y_start)
        self.assertEqual(230, x_end)
        self.assertEqual(180, y_end)


if __name__ == '__main__':
    unittest.main()
