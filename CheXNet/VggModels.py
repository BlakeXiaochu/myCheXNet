import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class VGG16(nn.Module):

    def __init__(self, classCount, isTrained):

        super(VGG16, self).__init__()

        self.vgg16 = torchvision.models.vgg16(pretrained=True)

        kernelCount = self.vgg16.fc.in_features

        self.vgg16.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.vgg16(x)
        return x

