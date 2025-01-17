import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from ResnetModels import *
from senet import *
from DatasetGenerator import DatasetGenerator
from models import MyModels

#-------------------------------------------------------------------------------- 

class ChexnetTrainer ():

    #---- Train the densenet network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def train (self, pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):

        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet-18': model = ResNet18(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet-50': model = ResNet50(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet18_lh': model = MyModels.ResNet18_lh(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet18_fpn': model = MyModels.ResNet18_fpn(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet50_fpn': model = MyModels.ResNet50_fpn(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'senet50_fpn': model = MyModels.senet50_fpn(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'senet50_sm': model = MyModels.senet50_sm(nnClassCount, nnIsTrained).cuda()
        else: raise Exception('Wrong Name!')

        model = torch.nn.DataParallel(model).cuda()
                
        #-------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=15, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=5, pin_memory=True)

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss(size_average = True)
        
        #---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
                         
            self.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            lossVal, losstensor = self.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(losstensor.data[0])
            
            # loss decreases, then save model
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
            if epochID == trMaxEpoch-1:
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-final-checkpoint'  + '.pth.tar')
                     
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        
        for batchID, (input, target) in enumerate (dataLoader):
                        
            target = target.cuda(async = True)
                 
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)
            
            lossvalue = loss(varOutput, varTarget)

            if batchID % 1000 == 0:
                print('training loss: {}'.format(lossvalue[0]))
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
    #-------------------------------------------------------------------------------- 
        
    def epochVal (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        
        losstensorMean = 0
        
        for i, (input, target) in enumerate (dataLoader):
            
            target = target.cuda(async=True)
                 
            varInput = torch.autograd.Variable(input, volatile=True)
            varTarget = torch.autograd.Variable(target, volatile=True)    
            varOutput = model(varInput)
            
            losstensor = loss(varOutput, varTarget)
            losstensorMean += losstensor
            
            lossVal += losstensor.data[0]
            lossValNorm += 1
            
        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm
        
        return outLoss, losstensorMean
               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (self, dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC
        
        
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (self, pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet-18': model = ResNet18(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet-50': model = ResNet50(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet18_lh': model = MyModels.ResNet18_lh(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet18_fpn': model = MyModels.ResNet18_fpn(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet50_fpn': model = MyModels.ResNet50_fpn(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'senet50_fpn': model = MyModels.senet50_fpn(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'senet50_sm': model = MyModels.senet50_sm(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'multi_model': 
            # model1 = se_resnet50(1000, pretrained='imagenet').cuda()
            # kernelCount = model1.last_linear.in_features
            # model1.last_linear = nn.Sequential(nn.Linear(kernelCount, nnClassCount), nn.Sigmoid())
            # model1.avg_pool = nn.AvgPool2d(8, stride=1)

            model1 = MyModels.senet50_sm(nnClassCount, nnIsTrained).cuda()
            model2 = MyModels.senet50_fpn(nnClassCount, nnIsTrained).cuda()
        
        if nnArchitecture != 'multi_model':
            model = torch.nn.DataParallel(model).cuda()
            modelCheckpoint = torch.load(pathModel)
            model.load_state_dict(modelCheckpoint['state_dict'])
        else:
            model1 = torch.nn.DataParallel(model1).cuda()
            model2 = torch.nn.DataParallel(model2).cuda()
            modelCheckpoint1 = torch.load(pathModel[0])
            model1.load_state_dict(modelCheckpoint1['state_dict'])

            modelCheckpoint2 = torch.load(pathModel[1])
            model2.load_state_dict(modelCheckpoint2['state_dict'])

            model = MyModels.multi_model(model1, model2)

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        for i, (input, target) in enumerate(dataLoaderTest):
            
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)
            
            bs, n_crops, c, h, w = input.size()
            
            varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)
            
            out = model(varInput)
            outMean = out.view(bs, n_crops, -1).mean(1)
            
            outPRED = torch.cat((outPRED, outMean.data), 0)

#        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocIndividual = self.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        
     
        return
#-------------------------------------------------------------------------------- 





