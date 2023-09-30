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

# Define the CNN regressor model
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 *192*192, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        #print(x.shape)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        #print(x.shape)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
