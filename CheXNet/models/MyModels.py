import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from senet import *

import torchvision


class ResNet18_lh(nn.Module):
    def __init__(self, classCount, isTrained):
        super(ResNet18_lh, self).__init__()

        self.ConvNet = torchvision.models.resnet18(pretrained=True)

        # lh
        mid_chn = 128
        out_chn = 256
        lh_size = [(7, 1), (1, 7)]
        lh_padding = [(3, 0), (0, 3)]
        self.lh1_1 = nn.Conv2d(in_channels=512, out_channels=mid_chn, kernel_size=lh_size[0], stride=1,
                               padding=lh_padding[0])
        self.relu_lh1_1 = nn.ReLU()
        self.lh1_2 = nn.Conv2d(in_channels=512, out_channels=mid_chn, kernel_size=lh_size[1], stride=1,
                               padding=lh_padding[1])
        self.relu_lh1_2 = nn.ReLU()
        self.lh2_1 = nn.Conv2d(in_channels=mid_chn, out_channels=out_chn, kernel_size=lh_size[1], stride=1,
                               padding=lh_padding[1])
        self.relu_lh2_1 = nn.ReLU()
        self.lh2_2 = nn.Conv2d(in_channels=mid_chn, out_channels=out_chn, kernel_size=lh_size[0], stride=1,
                               padding=lh_padding[0])
        self.relu_lh2_2 = nn.ReLU()

        self.pool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        # self.pool = nn.AdaptiveAvgPool2d((3, 3))

        self.fc = nn.Sequential(nn.Linear(out_chn, classCount), nn.Sigmoid())

    def forward(self, x):
        for name, module in self.ConvNet._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        lh1_1 = self.relu_lh1_1(self.lh1_1(x))
        lh1_2 = self.relu_lh1_2(self.lh1_2(x))
        lh2_1 = self.relu_lh2_1(self.lh2_1(lh1_1))
        lh2_2 = self.relu_lh2_2(self.lh2_2(lh1_2))

        x = self.pool(lh2_1 + lh2_2)
        x = self.fc(x.view(x.size(0), -1))

        return x


class ResNet18_fpn(nn.Module):
    def __init__(self, classCount, isTrained):
        super(ResNet18_fpn, self).__init__()

        self.sepa_inds1 = Variable(torch.LongTensor([4, 5, 6, 8, 10]).cuda())
        self.sepa_inds2 = Variable(torch.LongTensor([0, 1, 2, 3, 7, 9, 11, 12, 13]).cuda())
        self.ConvNet = torchvision.models.resnet18(pretrained=True)

        smooth4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        relu_smooth4 = nn.ReLU()
        a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        relu_a = nn.ReLU()

        self.final_conv = nn.Sequential(smooth4, relu_smooth4, a, relu_a)
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_do = nn.Dropout(p=0.5)
        self.final_fc = nn.Sequential(nn.Linear(512, 9), nn.Sigmoid())

        self.toplayer = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.lateral3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.smooth3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.branches = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for i in range(5):
            self.branches.append(self._branch(512))
            self.dropouts.append(nn.Dropout(p=0.5))
            self.fcs.append(nn.Sequential(nn.Linear(512, 1), nn.Sigmoid()))

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _branch(self, in_chn):
        conv = nn.Conv2d(in_channels=in_chn, out_channels=in_chn, kernel_size=1, stride=1, padding=0)
        # bn = nn.BatchNorm2d(in_chn)
        relu = nn.ReLU()
        fc = nn.Linear(in_chn, 1)
        mp = nn.MaxPool2d(2, 2)
        pool = nn.AdaptiveAvgPool2d((1, 1))
        # return nn.Sequential(conv, bn, relu, mp, pool)
        return nn.Sequential(conv, relu, mp, pool)

    def forward(self, x):
        for name, module in self.ConvNet._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
            if name == 'layer3':
                c3 = x
        c4 = x

        p4 = self.relu1(self.toplayer(c4))
        p3 = self._upsample_add(p4, self.relu2(self.lateral3(c3)))
        p3 = self.relu3(self.smooth3(p3))

        res1 = []
        for branch, do, fc in zip(self.branches, self.dropouts, self.fcs):
            feat3 = branch(p3).view(p3.size(0), -1)
            res1.append(fc(do(feat3)))

        res1 = torch.cat(res1, dim=1)

        feat4 = self.final_conv(c4)
        res2 = self.final_pool(feat4).view(p3.size(0), -1)
        res2 = self.final_fc(self.final_do(res2))

        final_res = Variable(torch.Tensor(c4.size(0), 14).cuda())
        final_res[:, self.sepa_inds1] = res1
        final_res[:, self.sepa_inds2] = res2

        return final_res


class ResNet50_fpn(nn.Module):
    def __init__(self, classCount, isTrained):
        super(ResNet50_fpn, self).__init__()

        self.ConvNet = torchvision.models.resnet50(pretrained=True)

        self.toplayer = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0)
        # self.relu_toplayer = nn.ReLU()

        # self.smooth4 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)
        # self.relu_smooth4 = nn.ReLU()

        self.lateral3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        # self.relu_lateral3 = nn.ReLU()

        self.smooth3 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_smooth3 = nn.BatchNorm2d(512)
        self.relu_smooth3 = nn.ReLU()

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0), \
            nn.BatchNorm2d(1024), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(1024, classCount), nn.Sigmoid())

        # self.branches = nn.ModuleList()
        # self.dropouts = nn.ModuleList()
        # self.fcs = nn.ModuleList()
        # for i in range(classCount):
        # self.branches.append(self._branch(512))
        # self.dropouts.append(nn.Dropout(p=0.5))
        # self.fcs.append(nn.Sequential(nn.Linear(512, 1), nn.Sigmoid()))

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _branch(self, in_chn):
        conv = nn.Conv2d(in_channels=in_chn, out_channels=in_chn, kernel_size=1, stride=1, padding=0)
        relu = nn.ReLU()
        fc = nn.Linear(in_chn, 1)
        # mp = nn.MaxPool2d(2, 2)
        pool = nn.AdaptiveAvgPool2d((1, 1))
        # return nn.Sequential(conv, relu, mp, pool)
        return nn.Sequential(conv, relu, pool)

    def forward(self, x):
        for name, module in self.ConvNet._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
            if name == 'layer3':
                c3 = x
        c4 = x

        p4 = self.toplayer(c4)
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p3 = self.relu_smooth3(self.bn_smooth3(self.smooth3(p3)))

        feat = self.final_conv(p3)
        feat = self.pool(feat).view(feat.size(0), -1)
        final_res = self.fc(feat)

        # res = []
        # for branch, do, fc in zip(self.branches, self.dropouts, self.fcs):
        # feat3 = branch(p3)
        # feat3 = feat3.view(feat3.size(0), -1)
        # res.append(fc(feat3))

        # final_res = torch.cat(res, dim=1)
        return final_res


class senet50_fpn(nn.Module):
    def __init__(self, classCount, isTrained):
        super(senet50_fpn, self).__init__()

        pretrained = 'imagenet' if isTrained else None

        self.ConvNet = se_resnet50(num_classes=1000, pretrained=pretrained)

        self.toplayer = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
        # self.relu_toplayer = nn.ReLU()

        # self.smooth4 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)
        # self.relu_smooth4 = nn.ReLU()

        self.lateral3 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        # self.relu_lateral3 = nn.ReLU()

        self.smooth3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn_smooth3 = nn.BatchNorm2d(512)
        self.relu_smooth3 = nn.ReLU()

        self.final_conv = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0), \
                                        nn.BatchNorm2d(1024), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(1024, classCount), nn.Sigmoid())

        # self.branches = nn.ModuleList()
        # self.dropouts = nn.ModuleList()
        # self.fcs = nn.ModuleList()
        # for i in range(classCount):
        # self.branches.append(self._branch(512))
        # self.dropouts.append(nn.Dropout(p=0.5))
        # self.fcs.append(nn.Sequential(nn.Linear(512, 1), nn.Sigmoid()))

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _branch(self, in_chn):
        conv = nn.Conv2d(in_channels=in_chn, out_channels=in_chn, kernel_size=1, stride=1, padding=0)
        relu = nn.ReLU()
        fc = nn.Linear(in_chn, 1)
        # mp = nn.MaxPool2d(2, 2)
        pool = nn.AdaptiveAvgPool2d((1, 1))
        # return nn.Sequential(conv, relu, mp, pool)
        return nn.Sequential(conv, relu, pool)

    def forward(self, x):
        for name, module in self.ConvNet._modules.items():
            if name == 'avg_pool':
                break
            x = module(x)
            if name == 'layer3':
                c3 = x
        c4 = x

        p4 = self.toplayer(c4)
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p3 = self.relu_smooth3(self.bn_smooth3(self.smooth3(p3)))

        feat = self.final_conv(p3)
        feat = self.pool(feat).view(feat.size(0), -1)
        final_res = self.fc(feat)

        return final_res

    def features(self, x):
        for name, module in self.ConvNet._modules.items():
            if name == 'avg_pool':
                break
            x = module(x)
            if name == 'layer3':
                c3 = x
        c4 = x

        p4 = self.toplayer(c4)
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p3 = self.relu_smooth3(self.bn_smooth3(self.smooth3(p3)))

        feat = self.final_conv(p3)

        return feat


class senet50_sm(nn.Module):
    def __init__(self, classCount, isTrained):
        super(senet50_sm, self).__init__()

        pretrained = 'imagenet' if isTrained else None

        self.ConvNet = se_resnet50(num_classes=1000, pretrained=pretrained)

        self.sm1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.relu_sm1 = nn.ReLU()
        self.sm2_b1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU()
        )
        self.sm2_b2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU()
        )
        self.sm_sign = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.sm_amp = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(1024 * 2
                                          , classCount), nn.Sigmoid())

    def forward(self, x):
        for name, module in self.ConvNet._modules.items():
            if name == 'avg_pool':
                break
            x = module(x)
        c4 = x
        x = self.relu_sm1(self.sm1(x))
        x = self.sm2_b1(x) + self.sm2_b2(x)

        # sm_sign = self.sm_sign(x)
        sm_amp = torch.exp(self.sm_amp(x))

        # feat = c4 * sm_sign * sm_amp
        feat = c4 * sm_amp
        feat = self.pool(feat).view(feat.size(0), -1)
        final_res = self.fc(feat)

        return final_res

    def features(self, x):
        for name, module in self.ConvNet._modules.items():
            if name == 'avg_pool':
                break
            x = module(x)
        c4 = x
        x = self.relu_sm1(self.sm1(x))
        x = self.sm2_b1(x) + self.sm2_b2(x)

        # sm_sign = self.sm_sign(x)
        sm_amp = torch.exp(self.sm_amp(x))

        # feat = c4 * sm_sign * sm_amp
        feat = c4 * sm_amp

        return feat


class multi_model(nn.Module):
    def __init__(self, model1, model2):
        super(multi_model, self).__init__()

        self.model1 = model1
        self.model2 = model2

        # self.cls_inds = Variable(torch.LongTensor([0, 3, 4, 5, 6, 7, 10])).cuda()
        # self.cls_inds = torch.LongTensor([0, 3, 4, 5, 6, 7, 10]).cuda()
        self.cls_inds = torch.LongTensor([0, 1, 3, 4, 5, 6, 10, 11]).cuda()
        self.weights1 = list(self.model1.module.parameters())[-2].clone()
        self.weights2 = list(self.model2.module.parameters())[-2].clone()
        self.bias1 = list(self.model1.module.parameters())[-1].clone()
        self.bias2 = list(self.model2.module.parameters())[-1].clone()
        self.heatSigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)

        # x = x1.copy()
        # x1[:, self.cls_inds] = x2[:, self.cls_inds]
        x = (x1 + x2) / 2
        return x

    def features(self, x):
        feat1 = self.model1.module.features(x)
        feat2 = self.model2.module.features(x)
        return feat1, feat2

    def heatmap_generate(self, x, ind):
        assert x.size(0) == 1

        feat1, feat2 = self.features(x)

        inds = self.cls_inds.cpu().numpy()
        feat = feat2 if ind in inds else feat1
        weights = self.weights2 if ind in inds else self.weights1
        weights = weights[ind, :].view(1, feat.size(1), 1, 1)
        bias = self.bias2 if ind in inds else self.bias1
        bias = bias[ind].view(1, 1, 1, 1)

        heatmap = torch.sum(feat * weights, dim=1) + bias
        heatmap = heatmap.view(heatmap.size()[-2:])         # size 8 x 8

        heatmap = self.heatSigmoid(heatmap)

        return heatmap

