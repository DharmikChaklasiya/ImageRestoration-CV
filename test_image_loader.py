from unittest import TestCase

from image_loader import ImageTensorGroup


class TestImageTensorGroup(TestCase):
    def test_parse_param_file(self):
        ds = ImageTensorGroup(None, [])
        ds.parse_param_file('test', ["person pose (x,y,z,rot x, rot y, rot z) =  -1 -3 0 0 0 1.02974",
                                     "person shape =  sitting"])
