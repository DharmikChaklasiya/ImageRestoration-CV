import unittest
from unittest import TestCase

from image_loader import ImageCropper
from parameter_file_parser import BoundingBox


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

        bbox, orig_bbx_moved = cropper.randomly_crop_image(img_shape, BoundingBox(0, 0, 199, 199))

        self.assertEqual(BoundingBox(50, 50, 150, 150), bbox)
        self.assertEqual(BoundingBox(0, 0, 99, 99), orig_bbx_moved)

    def test_random_crop_2(self):
        width_param = RandomParam("width", 120, 80, 160)
        height_param = RandomParam("height", 100, 80, 160)
        x_param = RandomParam("x", 20, 0, 40)
        y_param = RandomParam("y", 50, 40, 60)

        cropper = TestableImageCropper(width_param, height_param, x_param, y_param)
        img_shape = (3, 200, 200)

        bbox, orig_bbx_moved = cropper.randomly_crop_image(img_shape, BoundingBox(40, 60, 120, 140))

        self.assertEqual(BoundingBox(20, 50, 140, 150), bbox)
        self.assertEqual(BoundingBox(20, 10, 100, 90), orig_bbx_moved)

    def test_random_crop_lower_right_corner(self):
        width_param = RandomParam("width", 58, 29, 58)
        height_param = RandomParam("height", 78, 39, 78)
        x_param = RandomParam("x", 141, 141, 142)
        y_param = RandomParam("y", 121, 121, 122)

        cropper = TestableImageCropper(width_param, height_param, x_param, y_param)
        img_shape = (3, 200, 200)

        bbox, orig_bbx_moved = cropper.randomly_crop_image(img_shape, BoundingBox(170, 160, 199, 199))

        self.assertEqual(BoundingBox(141, 121, 199, 199), bbox)
        self.assertEqual(BoundingBox(29, 39, 57, 77), orig_bbx_moved)

    def test_random_crop_from_image_003983(self):
        width_param = RandomParam("width", 70, 61, 122)
        height_param = RandomParam("height", 80, 59, 118)
        x_param = RandomParam("x", 160, 154, 163)
        y_param = RandomParam("y", 100, 88, 109)

        cropper = TestableImageCropper(width_param, height_param, x_param, y_param)
        img_shape = (3, 512, 512)

        bbox, orig_bbx_moved = cropper.randomly_crop_image(img_shape, BoundingBox(163, 109, 224, 168))

        self.assertEqual(BoundingBox(160, 100, 230, 180), bbox)
        self.assertEqual(BoundingBox(3, 9, 64, 68), orig_bbx_moved)


if __name__ == '__main__':
    unittest.main()
