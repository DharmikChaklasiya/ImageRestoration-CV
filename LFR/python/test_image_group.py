from unittest import TestCase

from LFR.python.image_group import ImageGroup


class TestImageGroup(TestCase):
    def test_calc_full_image_index(self):
        self.assertEqual(100001, ImageGroup.calc_full_image_index(1, 1))
        self.assertEqual(101000, ImageGroup.calc_full_image_index(1, 1000))
        with self.assertRaises(AssertionError):
            ImageGroup.calc_full_image_index(1, 100000)

    def test_init(self):
        img = ImageGroup(10)
        self.assertEqual(img.formatted_image_index, "000010")
