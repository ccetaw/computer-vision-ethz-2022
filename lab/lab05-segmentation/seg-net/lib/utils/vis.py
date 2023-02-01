from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

def get_mnist_labels():
    """Load the mapping that associates MNIST classes with label colors

    Returns:
        np.ndarray with dimensions (11, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
        ]
    )

def vis_segments(labels, num_classes):
    colors = get_mnist_labels()
    height = labels.shape[0]
    width = labels.shape[1]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    xv, yv = np.meshgrid(np.arange(0, width), np.arange(0, height))

    img[yv, xv, :] = colors[labels]

    return img
