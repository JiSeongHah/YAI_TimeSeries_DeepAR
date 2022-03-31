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
import pickle
from vanillaLSTM4ConfidenceCheck import vanillaLSTMPredictor

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    baseDir = '/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/dataset/'

    data_folder_dir_trn = baseDir + 'train/'
    data_folder_dir_val = baseDir + 'val/'
    data_folder_dir_test = baseDir + 'test/'

    coin = 'Bitcoin'

    iter_to_accumul = 1
    hiddenSizeLst = [64,128,256,512,1024]
    numLayerLst = [4,8]
    linNumLst = [64,12,256]
    afterRangeNumLst= [1,2,4,8,16,32,64]
    seqLenLst= [1,2,4,8,16,32,64,128,512]

    MaxStepTrn = 1000000000000000000
    MaxStepVal = 10000000000000000000
    FeeRate= 0.1 /100
    commitThresholdLst = [0,0.05,0.1]
    confidenThresholdLst = [0.7,0.8,0.9]
    labelMethod = 'UpDown'


    for numLayer in numLayerLst:
        for hiddenSize in hiddenSizeLst:
            for linNum in linNumLst:
                for seqLen in seqLenLst:
                    for afterRangeNum in afterRangeNumLst:
                        for commitThreshold in commitThresholdLst:
                            for confidenThreshold in confidenThresholdLst:



                                    linNum1 = linNum
                                    linNum2 = linNum
                                    bSizeTrn = 2048
                                    bSizeVal = 2048
                                    bSizeTst = 100

                                    modelLoadNum = 0
                                    save_range = 100
                                    MaxEpoch = 1000000
                                    gpuUse = True
                                    inputSize = 5

                                    savingDir = mk_name(dirVanillaLSTM='/',
                                                        hidden=hiddenSize,
                                                        nlayer=numLayer,
                                                        linNum=linNum,
                                                        seqLen=seqLen,
                                                        after=afterRangeNum)
                                    modelPlotSaveDir = baseDir +'Results/'+savingDir + '/'
                                    createDirectory(modelPlotSaveDir)
                                    createDirectory(modelPlotSaveDir+'models')

                                    try:
                                        print(f'Loading {modelPlotSaveDir + str(modelLoadNum)}.pth')
                                        MODEL_START = torch.load(modelPlotSaveDir +'models/'+ str(modelLoadNum) + '.pth')
                                    except:
                                        MODEL_START  = vanillaLSTMPredictor(
                                            data_folder_dir_trn= data_folder_dir_trn,
                                            data_folder_dir_val = data_folder_dir_val,
                                            data_folder_dir_test = data_folder_dir_test,
                                            modelPlotSaveDir = modelPlotSaveDir,
                                            coin = coin,
                                            iter_to_accumul = iter_to_accumul,
                                            inputSize = inputSize,
                                            hiddenSize= hiddenSize,
                                            numLayer = numLayer,
                                            seqLen =seqLen,
                                            MaxStepTrn=MaxStepTrn,
                                            MaxStepVal=MaxStepVal,
                                            bSizeTrn= bSizeTrn,
                                            bSizeVal= bSizeVal,
                                            bSizeTst= bSizeTst,
                                            linNum1=linNum1,
                                            linNum2=linNum2,
                                            commitThreshold=commitThreshold,
                                            confidenceThreshold=confidenThreshold,
                                            FeeRate=FeeRate,
                                            labelMethod=labelMethod,
                                            afterRangeNum=afterRangeNum,

                                            gpuUse= gpuUse

                                        )



                                    for i in range(10000000000):
                                        MODEL_START.START_TRN_VAL(epoch=i)

                                        if i%save_range ==0:
                                            try:
                                                torch.save(MODEL_START, modelPlotSaveDir+'models/' + str(i) + '.pth')
                                                print('saving model complete')
                                                time.sleep(5)
                                            except:
                                                print('saving model failed')
                                                time.sleep(5)

                                        if np.mean(MODEL_START.loss_lst_val[-10:]) < 0.0001:
                                            ResultDict= dict()
                                            ResultDict['loss'] = MODEL_START.loss_lst_val
                                            ResultDict['acc'] = MODEL_START.acc_lst_val
                                            ResultDict['confi'] = MODEL_START.confAndCorr_lst_val

                                            with open('valdationResult.pkl','wb') as F:
                                                pickle.dump(ResultDict,F)


                                            break


