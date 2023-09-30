import os
import cv2
import torch
from torch.utils import data
import numpy as np
import random
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
class ImageDataTestNOMOS(data.Dataset):
    def __init__(self, data_root, data_list,image_size):
        self.data_root = data_root
        self.data_list = data_list
        self.image_size = image_size
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        depth = load_image(os.path.join(self.data_root, self.image_list[item].split()[1]), self.image_size)
        depth = depth.transpose((2, 0, 1))
        depth = torch.Tensor(depth)
        name  = str(self.image_list[item % self.image_num].split()[0].split('/')[1])
        #print('name',name)
        return depth,name

    def __len__(self):
        return self.image_num

def load_image(path,size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = np.array(im , dtype=np.float32)
    img = np.resize(im,(size,size))
    img = img / 255.0
    img = img[..., np.newaxis]
    return img
