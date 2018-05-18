import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

#-------------------------------------------------------------------------------- 

def main ():
    
    runTest()
    # runTrain()
  
#--------------------------------------------------------------------------------   

def runTrain():
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = '/home/dataset/ChextXRay/'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_label.txt'
    pathFileVal = './dataset/val_label.txt'
    pathFileTest = './dataset/test_label.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    # nnArchitecture = DENSENET121
    nnArchitecture = 'senet50_sm'
    nnIsTrained = True
    nnClassCount = 14
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 64
    trMaxEpoch = 100
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 320
    imgtransCrop = 256
        
    pathModel = 'm-' + timestampLaunch + '.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    Trainer = ChexnetTrainer()
    Trainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    
    print ('Testing the trained model')
    Trainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = '/home/dataset/ChextXRay/'
    pathFileTest = './dataset/test_label.txt'
    
    # nnArchitecture = 'senet50_sm'
    nnArchitecture = 'multi_model'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 16
    imgtransResize = 320
    imgtransCrop = 256
    
    # pathModel = ['./senet50.pth.tar', './senet50_fpn.pth.tar']
    # pathModel = 'm-12052018-145036.pth.tar'
    pathModel = ['./senet50_sm.pth.tar', './senet50_fpn.pth.tar']
    
    timestampLaunch = ''
    Trainer = ChexnetTrainer()
    Trainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()




