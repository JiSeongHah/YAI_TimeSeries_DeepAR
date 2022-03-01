import os
import torch
import torch.nn as nn
from MY_MODELS import simpleDNN
from torch.optim import AdamW, Adam,SGD
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from DataLoading import MyEelDataset
from save_funcs import createDirectory,mk_name
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from torch.nn import L1Loss
from sklearn.metrics import f1_score
from MY_MODELS import MyDeepAR
from DEEPARPREDICTOR import DeepARPredictor

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    baseDir = '/home/a286/hjs_dir1/g-research-crypto-forecasting/dataset/'

    data_folder_dir_trn = baseDir + 'train/'
    data_folder_dir_val  = baseDir + 'val/'
    data_folder_dir_test = baseDir + 'test/'

    coin = 'Bitcoin'

    iter_to_accumul = 1
    hiddenSizeLst = [128,256,512]
    numLayerLst = [4,8]
    linNumLst = [32,64,128]

    windowRangeTstLst = [16,32,64,128,256]
    XrangeNumLst = [13,14,15,16]



    for numLayer in numLayerLst:
        for linNum in linNumLst:
            for hiddenSize in hiddenSizeLst:
                for windowRangeTst in windowRangeTstLst:
                    for XrangeNum in  XrangeNumLst:

                        muLin1 = linNum
                        muLin2 = linNum
                        sigLin1 = linNum
                        sigLin2 = linNum

                        seqLen = windowRangeTst + 15
                        windowRangeTrn = 1
                        windowRangeVal = 1
                        sigmaNum = 1
                        bSizeTrn = 2048
                        bSizeVal = 2048
                        bSizeTst = 100
                        MaxStepTrn = seqLen
                        MaxStepVal = seqLen
                        sampleNum = 1024
                        XtMethod = 'real'

                        modelLoadNum = 600
                        save_range = 10
                        MaxEpoch = 100000
                        gpuUse = True
                        inputSize = XrangeNum*5 + 5

                        savingDir = mk_name(dir4='/',model='divVer',XrangeNum=XrangeNum,hidden=hiddenSize,nlayer=numLayer,linNum=linNum,wdw=windowRangeTst,signum=sigmaNum,bs=bSizeTrn)
                        modelPlotSaveDir = baseDir +'Results/'+savingDir + '/'
                        createDirectory(modelPlotSaveDir)
                        createDirectory(modelPlotSaveDir+'models')

                        try:
                            print(f'Loading {modelPlotSaveDir + str(modelLoadNum)}.pth')
                            MODEL_START = torch.load(modelPlotSaveDir +'models/'+ str(modelLoadNum) + '.pth')
                        except:
                            MODEL_START  = DeepARPredictor(
                                data_folder_dir_trn= data_folder_dir_trn,
                                data_folder_dir_val = data_folder_dir_val,
                                data_folder_dir_test = data_folder_dir_test,
                                modelPlotSaveDir = modelPlotSaveDir,
                                coin = coin,
                                iter_to_accumul = iter_to_accumul,
                                inputSize = inputSize,
                                hiddenSize= hiddenSize,
                                numLayer = numLayer,
                                muLin1 = muLin1,
                                muLin2 = muLin2,
                                sigLin1 = sigLin1,
                                sigLin2 = sigLin2,
                                seqLen =seqLen,
                                windowRangeTrn = windowRangeTrn,
                                windowRangeVal = windowRangeVal,
                                windowRangeTst = windowRangeTst,
                                MaxStepTrn=MaxStepTrn,
                                MaxStepVal=MaxStepVal,
                                sigmaNum= sigmaNum,
                                bSizeTrn= bSizeTrn,
                                bSizeVal= bSizeVal,
                                bSizeTst= bSizeTst,
                                gpuUse= gpuUse,
                                sampleNum=sampleNum,
                                XtMethod=XtMethod
                            )

                        #MODEL_START.TestStep(timeStamp=1624846200+60*90000)

                        for i in range(100):
                            MODEL_START.START_TRN_VAL(epoch=i,MaxEpoch=MaxEpoch)

                            if i%save_range ==0:
                                if i > MaxEpoch:
                                    break

                                try:
                                    torch.save(MODEL_START, modelPlotSaveDir+'models/' + str(i) + '.pth')
                                    print('saving model complete')
                                    time.sleep(5)
                                except:
                                    print('saving model failed')
                                    time.sleep(5)

