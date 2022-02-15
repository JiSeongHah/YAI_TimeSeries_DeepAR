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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    baseDir = '/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/dataset/'

    data_folder_dir_trn = baseDir + 'train/'
    data_folder_dir_val  = baseDir + 'val/'
    data_folder_dir_test = baseDir + 'test/'



    coin = 'Bitcoin'
    iter_to_accumul = 1
    inputSize = 5
    hiddenSize = 128
    numLayer = 3
    muLin1 = 128
    muLin2= 128
    sigLin1 = 128
    sigLin2 = 128
    seqLen = 128+15
    windowRangeTrn = 1
    windowRangeVal = 1
    windowRangeTst = 128
    sigmaNum = 2
    bSizeTrn = 3
    bSizeVal = 3
    bSizeTst = 4
    MaxStepTrn = 128
    MaxStepVal =128

    modelLoadNum = 1
    save_range= 100
    MaxEpoch = 10000
    gpuUse= True

    savingDir = mk_name(mu1=muLin1,mu2=muLin2,sig1=sigLin1,sig2=sigLin2,wdw=windowRangeTst,signum=2,bs=64)
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
        )



    # MODEL_START.TestStep()

    for i in range(10000):
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

























