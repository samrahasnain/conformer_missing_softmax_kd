import os

import cv2
import torch
from torch.utils import data
import numpy as np
import random

random.seed(10)


class ImageDataTrain(data.Dataset):
    def __init__(self, qp_root,rgb,depth,gt,quality,image_size):
        self.sal_root = qp_root
        self.sal_rgb = rgb
        self.sal_depth = depth
        self.sal_gt = gt
        self.quality_label = quality
        self.image_size = image_size
       
        self.sal_num = len(self.sal_rgb)

    def __getitem__(self, item):
        # sal data loading
        im_name = self.sal_rgb[item]
        de_name = self.sal_depth[item]
        gt_name = self.sal_gt[item]
        q_score = self.quality_label[item]
        #print(quality_label)
        sal_image = load_image(os.path.join(self.sal_root, im_name), self.image_size)
        sal_depth = load_image(os.path.join(self.sal_root, de_name), self.image_size)
        sal_label = load_sal_label(os.path.join(self.sal_root, gt_name), self.image_size)
        name = self.sal_rgb[item].split('/')[1]
        q_score = load_score(q_score)
        if q_score==0.0:
            # Replace missing image with a default value 
            # For this example, we'll use a black image with all zeros.
            sal_depth = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
            #print(sal_depth.shape,sal_depth)

        sal_image, sal_depth, sal_label = cv_random_crop(sal_image, sal_depth, sal_label, self.image_size)
        
        sal_image = sal_image.transpose((2, 0, 1))
        sal_depth = sal_depth.transpose((2, 0, 1))
        sal_label = sal_label.transpose((2, 0, 1))

        sal_image = torch.Tensor(sal_image)
        sal_depth = torch.Tensor(sal_depth)
        sal_label = torch.Tensor(sal_label)
        q_score = torch.Tensor(q_score)
        
        #print('dataset',q_score)
        
        sample = {'sal_image': sal_image, 'sal_depth': sal_depth, 'sal_label': sal_label,'quality_label':q_score,'name':name}
        return sample

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, qp_root,rgb,depth,gt,quality,image_size):
        self.data_root = qp_root
        self.sal_rgb = rgb
        self.sal_depth = depth
        self.sal_gt = gt
        self.quality_label = quality
        self.image_size = image_size
 
        self.image_num = len(self.sal_rgb)

    def __getitem__(self, item):
        im_name = self.sal_rgb[item]
        de_name = self.sal_depth[item]
        gt_name = self.sal_gt[item]
        q_score = self.quality_label[item]
        
        image, im_size = load_image_test(os.path.join(self.data_root, im_name), self.image_size)
        depth, de_size = load_image_test(os.path.join(self.data_root,de_name), self.image_size)
        gt = load_sal_label(os.path.join(self.data_root,gt_name), self.image_size)
        name = self.sal_rgb[item].split('/')[1]
        q_score = load_score(q_score)
        '''if q_score==0.0:
            # Replace missing image with a default value 
            # For this example, we'll use a black image with all zeros.
            depth = np.zeros((3,self.image_size, self.image_size), dtype=np.float32)'''
        image = torch.Tensor(image)
        depth = torch.Tensor(depth)
        q_score = torch.Tensor(q_score)
        return {'image': image,  'depth': depth,'label':gt ,'quality_label':q_score,'name':name, 'size': im_size}

    def __len__(self):
        return self.image_num


def get_loader(qp_root,rgb,depth,gt,quality,config, mode='train', pin=True):
    shuffle = False
    if mode == 'train':
        shuffle = True
        #print('quality in get loader',quality)
        dataset = ImageDataTrain(qp_root,rgb,depth,gt,quality, config.image_size)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(qp_root,rgb,depth,gt,quality, config.image_size)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    return data_loader


def load_image(path,image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    return in_



def load_image_test(path,image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_sal_label(path,image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    label = np.array(im, dtype=np.float32)
    label = cv2.resize(label, (image_size, image_size))
    label = label / 255.0
    label = label[..., np.newaxis]
    return label

def load_score(score):
    label = np.array(score, dtype=np.float32)
    label = label[..., np.newaxis]
    #print('load_score',label)
    return label


def cv_random_crop(image, depth, label,image_size):
    crop_size = int(0.0625*image_size)
    croped = image_size - crop_size
    top = random.randint(0, crop_size)  #crop rate 0.0625
    left = random.randint(0, crop_size)

    image = image[top: top + croped, left: left + croped, :]
    depth = depth[top: top + croped, left: left + croped, :]
    label = label[top: top + croped, left: left + croped, :]
    image = cv2.resize(image, (image_size, image_size))
    depth = cv2.resize(depth, (image_size, image_size))
    label = cv2.resize(label, (image_size, image_size))
    label = label[..., np.newaxis]
    return image, depth, label

def Normalization(image):
    in_ = image[:, :, ::-1]
    in_ = in_ / 255.0
    in_ -= np.array((0.485, 0.456, 0.406))
    in_ /= np.array((0.229, 0.224, 0.225))
    return in_

