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
from MY_MODELS import EelPredCNNModel
from DEEPARPREDICTOR import EelPredictor

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()

    baseDir = '/home/a286/hjs_dir1/EelPred/datasetVer0/'
    #baseDir = '/home/a286/hjs_dir1/Dacon1/'

    data_folder_dir_trn = baseDir + 'train/'
    data_folder_dir_val  = baseDir + 'val/'
    labelDir = baseDir + 'train.csv'
    data_folder_dir_test = baseDir + 'test/'

    backboneOutFeature = 1000
    LinNum = 25

    MaxEpoch= 10000
    iter_to_accumul = 4
    MaxStep = 25
    MaxStepVal = 10000
    bSizeTrn =4
    save_range= 10
    modelLoadNum = 50
    CROP = [0,1000,300,1500]
    gpuUse = True
    whichModel= 'effnet-b7'
    lossFuc = 'Huber'

    savingDir = mk_name(model=whichModel,backNum=backboneOutFeature,LinNum=LinNum,bS=bSizeTrn,iter=iter_to_accumul,loss=lossFuc)
    modelPlotSaveDir = baseDir +'Results/'+savingDir + '/'
    createDirectory(modelPlotSaveDir)


    try:
        print(f'Loading {modelPlotSaveDir + str(modelLoadNum)}.pth')
        MODEL_START = torch.load(modelPlotSaveDir + str(modelLoadNum) + '.pth')
        MODEL_START.iter_to_accmul = 4
    except:
        MODEL_START  = EelPredictor(
            data_folder_dir_trn=data_folder_dir_trn,
            data_folder_dir_val=data_folder_dir_val,
            MaxEpoch=MaxEpoch,
            backboneOutFeature=backboneOutFeature,
            LinNum=LinNum,
            lossFuc=lossFuc,
            labelDir=labelDir,
            modelPlotSaveDir=modelPlotSaveDir,
            iter_to_accumul=iter_to_accumul,
            MaxStep=MaxStep,
            MaxStepVal=MaxStepVal,
            bSizeTrn=bSizeTrn,
            gpuUse=gpuUse,
            CROP=CROP,
            data_folder_dir_test=data_folder_dir_test,
            whichModel=whichModel,
            bSizeVal=10, lr=3e-4, eps=1e-9)


    # MODEL_START.TestStep()

    for i in range(10000):
        MODEL_START.START_TRN_VAL(epoch=i)

        if i%save_range ==0:
            if i > MaxEpoch:
                break

            try:
                torch.save(MODEL_START, modelPlotSaveDir + str(i) + '.pth')
                print('saving model complete')
                print('saving model complete')
                print('saving model complete')
                print('saving model complete')
                print('saving model complete')
                time.sleep(5)
            except:
                print('saving model failed')
                print('saving model failed')
                print('saving model failed')
                print('saving model failed')
                print('saving model failed')
                time.sleep(5)

























