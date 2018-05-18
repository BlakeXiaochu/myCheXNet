import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201

from models import MyModels
from senet import *


# --------------------------------------------------------------------------------
# ---- Class to generate heatmaps (CAM)

class HeatmapGenerator(object):

    # ---- Initialize heatmap generator
    # ---- pathModel - path to the trained densenet model
    # ---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    # ---- nnClassCount - class count, 14 for chxray-14

    def __init__(self, pathModel, nnArchitecture, nnClassCount, transCrop):
        # ---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121':
            model = DenseNet121(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-169':
            model = DenseNet169(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-201':
            model = DenseNet201(nnClassCount, True).cuda()
        elif nnArchitecture == 'multi_model':
            model1 = se_resnet50(1000, pretrained=None).cuda()
            kernelCount = model1.last_linear.in_features
            model1.last_linear = nn.Sequential(nn.Linear(kernelCount, nnClassCount), nn.Sigmoid())
            model1.avg_pool = nn.AvgPool2d(8, stride=1)
            model2 = MyModels.senet50_fpn(nnClassCount, False).cuda()

        if nnArchitecture != 'multi_model':
            model = torch.nn.DataParallel(model).cuda()
            # modelCheckpoint = torch.load(pathModel)
            # model.load_state_dict(modelCheckpoint['state_dict'])
        else:
            model1 = torch.nn.DataParallel(model1).cuda()
            model2 = torch.nn.DataParallel(model2).cuda()

            modelCheckpoint1 = torch.load(pathModel[0])
            model1.load_state_dict(modelCheckpoint1['state_dict'])

            modelCheckpoint2 = torch.load(pathModel[1])
            model2.load_state_dict(modelCheckpoint2['state_dict'])

            model = MyModels.multi_model(model1, model2)

        # self.model = model.module.densenet121.features
        self.model = model
        self.model.eval()

        # ---- Initialize the weights
        # self.weights = list(self.model.parameters())[-2]

        # ---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        # transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.Scale(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)

        self.transformSequence = transforms.Compose(transformList)

    # --------------------------------------------------------------------------------

    def generate(self, pathImageFile, pathOutputFile, transCrop):
        # ---- Load image, transform, convert
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)

        x = torch.autograd.Variable(imageData, volatile=True)
        x = x.cuda()
        self.model.cuda()
        ind = 0

        res = self.model(x)
        heatmap = self.model.heatmap_generate(x, ind)
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))

        # cam = npHeatmap / np.max(npHeatmap)
        cam = npHeatmap
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        img = heatmap * 0.5 + imgOriginal

        cv2.imwrite(pathOutputFile, img)


# --------------------------------------------------------------------------------
if __name__ == '__main__':
    pathInputImage = 'test/00000118_001.png'
    pathOutputImage = 'test/heatmap118.png'
    # pathModel = 'models/m-25012018-123527.pth.tar'
    pathModel = ['./senet50.pth.tar', './senet50_fpn.pth.tar']

    # nnArchitecture = 'DENSE-NET-121'
    nnArchitecture = 'multi_model'
    nnClassCount = 14

    transCrop = 256

    h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
    h.generate(pathInputImage, pathOutputImage, transCrop)
