import pickle
from tqdm import tqdm
import os
from torch.utils import data
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class DataloaderCifar10(data.Dataset):
    def __init__(self, img_size=32, is_transform=False, split='train'):
        self.split = split
        self.img_size = img_size
        self.is_transform = is_transform
        self.transform_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # HWC --> CHW, 0~255->[0.0,1.0]
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        self.data_list = []

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_data(self, data_root):
        all_labels = []
        all_data = []
        if self.split in ['train', 'val']:
            file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        elif self.split == 'test':
            file_list = ['test_batch']
        else:
            raise ValueError('wrong split! the split should be chosen from train/val/test!')
        for i, file_name in enumerate(file_list):
            cur_batch = self.unpickle(os.path.join(data_root, file_name))
            data = cur_batch[b'data']  # [10000, 3072(32*32*3)] array
            labels = cur_batch[b'labels']  # [10000] list
            all_data.append(data)
            all_labels = all_labels + labels
        all_data = np.concatenate(all_data, axis=0)
        all_data = np.vstack(all_data).reshape(-1, 3, 32, 32)  # [num_img, 3, 32, 32], RGB
        all_data = all_data.transpose((0, 2, 3, 1))  # CHW --> HWC
        all_data = list(all_data)

        if self.split == 'train':
            self.data_list = all_data[0:45000]
            self.label_list = all_labels[0:45000]
        elif self.split == 'val':
            self.data_list = all_data[45000:]
            self.label_list = all_labels[45000:]
        elif self.split == 'test':
            self.data_list = all_data
            self.label_list = all_labels

        print('[INFO] {} set loaded, {} samples in total.'.format(self.split, len(self.data_list)))
        return self.data_list, self.label_list



    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        img = self.data_list[index]  # HWC, RGB, array
        label = self.label_list[index]
        img = Image.fromarray(img)
        if self.is_transform and self.split == 'train':
            img = self.transform_train(img)  # CHW, RGB,[0,1]
        else:
            img = self.transform_test(img)

        img = ((img - 0.5) / 0.5)  # normalize to [-1, 1]

        return img, label









