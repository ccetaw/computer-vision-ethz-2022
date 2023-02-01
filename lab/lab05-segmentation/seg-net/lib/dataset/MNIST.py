import os
import glob
import logging

import numpy as np
from scipy.io import loadmat, savemat

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class MNIST(Dataset):
    def __init__(self, root='.data/', is_train=True):
        """Initialization of the dataset class

        Args:
            root (str): root location of the multi-digit MNIST dataset
            is_train (bool): indicate whether the dataset is in training mode or testing mode
        """
        self.is_train = is_train

        self.root = root

        # We work on 64x64 images
        self.patch_width = 64
        self.patch_height = 64

        # Default mean and std for image pre-processing
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.mean = np.expand_dims(np.expand_dims(np.expand_dims(mean, axis=0), axis=-1), axis=-1)
        self.std = np.expand_dims(np.expand_dims(np.expand_dims(std, axis=0), axis=-1), axis=-1)

        self.db = self._get_db()
        self.db_length = len(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def __getitem__(self, idx):
        """Interface for PyTorch to get data samples by idx

        Args:
            idx (int): index of the data sample to get
        Returns:
            image (B x 3 x H x W numpy array): images
            semantic_mask (B x H x W numpy array): semantic labels for each pixel
        """
        file_name = self.db[idx]

        data = loadmat(file_name)

        image = data['imgMat'].transpose([3, 2, 0, 1]).astype(np.float32)  # image is already normalized to [0, 1]
        semantic_mask = data['semanticMaskMat'].transpose([2, 0, 1]).astype(np.int64)  # mask is in [0, 10], 0 indicates background while 1-10 indicate digits 0-9, respectively

        # Normalize the image
        image = (image - self.mean) / self.std

        return image, semantic_mask

    def _get_db(self):
        """Get data for multi-digit MNIST

        Returns:
            gt_db: a list of .mat files that contains pixel values and ground-truth mask for the dataset
        """
        gt_db = []

        prefix = 'batch' if self.is_train else 'testset'

        dataset_path = os.path.join('data/multi-digit-mnist', '{}*.mat'.format(prefix))

        gt_db = sorted(glob.glob(dataset_path))

        return gt_db

    def __len__(self):
        return self.db_length
