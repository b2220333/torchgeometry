import unittest
import numpy as np

import torch
import torchgeometry as tgm


class Tester(unittest.TestCase):

    def test_tensor_to_image(self):
        tensor = torch.ones(3, 4, 4)
        image = tgm.tensor_to_image(tensor)

        self.assertTrue(image.shape == (4, 4, 3))
        self.assertIsInstance(image, np.ndarray)

    def test_image_to_tensor(self):
        image = np.ones((4, 4, 3))
        tensor = tgm.image_to_tensor(image)

        self.assertTrue(tensor.shape == (3, 4, 4))
        self.assertIsInstance(tensor, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
