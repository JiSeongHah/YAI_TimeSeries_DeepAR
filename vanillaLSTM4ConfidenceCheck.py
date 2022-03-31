import os
import torch
import torch.nn as nn
from MY_MODELS import MyvanillaLSTM
from torch.optim import AdamW, Adam,SGD
from torch.nn import MSELoss,L1Loss,HuberLoss
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from DataLoading import MyEelDataset
from save_funcs import createDirectory,mk_name
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import pickle
from sklearn.metrics import f1_score
from efficientnet_pytorch import EfficientNet
from torch.nn import DataParallel
from torchvision import models
from vanillaLSTMDataset import vanillaLSTMDataset
import matplotlib.pyplot as plt


class vanillaLSTMPredictor(nn.Module):
    def __init__(self,
                 data_folder_dir_trn,
                 data_folder_dir_val,
                 data_folder_dir_test,
                 modelPlotSaveDir,
                 coin,
                 iter_to_accumul,
                 inputSize,
                 hiddenSize,
                 numLayer,
                 linNum1,
                 linNum2,
                 afterRangeNum,
                 seqLen,
                 FeeRate,
                 commitThreshold,
                 confidenceThreshold,
                 labelMethod,
                 MaxStepTrn,
                 MaxStepVal,
                 gpuUse,
                 bSizeTrn= 8,
                 bSizeVal=1,
                 bSizeTst=1,
                 lr=3e-4,
                 eps=1e-9):


        super(vanillaLSTMPredictor,self).__init__()

        self.data_folder_dir_trn = data_folder_dir_trn
        self.data_folder_dir_val = data_folder_dir_val
        self.data_folder_dir_test = data_folder_dir_test

        self.modelPlotSaveDir = modelPlotSaveDir

        self.iter_to_accumul = iter_to_accumul
        self.coin = coin
        self.gpuUse = gpuUse

        self.afterRangeNum =afterRangeNum
        self.seqLen =seqLen
        self.FeeRate = FeeRate
        self.commitThreshold = commitThreshold
        self.confidenceThreshold = confidenceThreshold
        self.labelMethod = labelMethod

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayer = numLayer
        self.linNum1 = linNum1
        self.linNum2 = linNum2

        self.MaxStepTrn = MaxStepTrn
        self.MaxStepVal = MaxStepVal

        self.lr = lr
        self.eps = eps
        self.bSizeTrn = bSizeTrn
        self.bSizeVal = bSizeVal
        self.bSizeTst = bSizeTst

        ###################MODEL SETTING###########################
        print('failed loading model, loaded fresh model')

        self.MyLSTMModel = MyvanillaLSTM(
            inputSize = self.inputSize,
            hiddenSize = self.hiddenSize,
            numLayer = self.numLayer,
            linNum1 = self.linNum1,
            linNum2 = self.linNum2,
            seqLen= self.seqLen,
            dropRatio=0.2,
            rnnBase='lstm',
            lastOrAll = 'last'
        )

        self.loss_lst_trn = []
        self.loss_lst_trn_tmp = []
        self.loss_lst_val = []
        self.loss_lst_val_tmp = []

        self.acc_lst_trn = []
        self.acc_lst_trn_tmp = []
        self.acc_lst_val = []
        self.acc_lst_val_tmp = []

        self.confAndCorr_lst_trn = []
        self.confAndCorr_lst_trn_tmp = []
        self.confAndCorr_lst_val = []
        self.confAndCorr_lst_val_tmp = []


        self.num4epoch = 0

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)

        self.optimizer = Adam(self.MyLSTMModel.parameters(),
                             lr=self.lr,  # 학습률
                             eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                             )

        MyTrnDataset = vanillaLSTMDataset(
            afterRangeNum=self.afterRangeNum,
            seqLen=self.seqLen,
            FeeRate=self.FeeRate,
            commitThreshold=self.commitThreshold,
            baseDir=self.data_folder_dir_trn,
            coin=self.coin,
            labelMethod=self.labelMethod
        )
        self.trainDataloader = DataLoader(MyTrnDataset,batch_size=self.bSizeTrn,shuffle=True)

        MyValDataset = vanillaLSTMDataset(
            afterRangeNum=self.afterRangeNum,
            seqLen=self.seqLen,
            FeeRate=self.FeeRate,
            commitThreshold=self.commitThreshold,
            baseDir=self.data_folder_dir_trn,
            coin=self.coin,
            labelMethod=self.labelMethod
        )
        self.valDataloader = DataLoader(MyValDataset, batch_size=self.bSizeVal, shuffle=False)

        self.MyTestDataset = vanillaLSTMDataset(
            afterRangeNum=self.afterRangeNum,
            seqLen=self.seqLen,
            FeeRate=self.FeeRate,
            commitThreshold=self.commitThreshold,
            baseDir=self.data_folder_dir_trn,
            coin=self.coin,
            labelMethod=self.labelMethod
        )


        self.MyLSTMModel.to(device=self.device)

    def forward(self,input):

        input = input.to(self.device)

        out = self.MyLSTMModel(input)

        return out


    def calLoss(self,logit,label):


        loss = nn.CrossEntropyLoss()

        lossResult = loss(logit,label)

        correct= torch.argmax(logit,dim=1) == label

        confidence = torch.argmax(logit,dim=1) > self.confidenceThreshold

        confAndCorr = confidence * correct

        correctRatio = correct/len(label)

        confAndCorrRatio = confAndCorr/len(label)


        return lossResult,correctRatio, confAndCorrRatio

    def init_hidden_cell(self,batchSize):

        return torch.zeros(self.numLayer, batchSize, self.hiddenSize)


    def trainingStep(self,trainingNum):

        self.MyLSTMModel.train()
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            globalTime= time.time()

            for idx,(bInput,bLabel) in enumerate(self.trainDataloader):

                bInput = bInput.to(self.device)

                localTime = time.time()

                bLogit = self.forward(bInput)

                bLogit = bLogit.cpu()

                Loss,correctRatio,confAndCorrRatio = self.calLoss(logit=bLogit,label=bLabel)

                Loss.backward()

                self.loss_lst_trn_tmp.append(Loss.item())
                self.acc_lst_trn_tmp.append(correctRatio)
                self.confAndCorr_lst_trn_tmp.append(confAndCorrRatio)

                print(f' {idx}/{len(self.trainDataloader)} th training for {trainingNum} complete !!!')

                if idx >= self.MaxStepTrn:
                    break

        torch.set_grad_enabled(False)
        self.MyLSTMModel.eval()

    def valdatingStep(self,validatingNum):

        self.MyLSTMModel.eval()
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            for idx,(bInputVal,bLabelVal) in enumerate(self.valDataloader):

                bInputVal = bInputVal.to(self.device)

                bLogitVal = self.forward(bInputVal)

                bLogitVal = bLogitVal.cpu()

                Loss, correctRatio, confAndCorrRatio = self.calLoss(logit=bLogitVal, label=bLabelVal)

                self.loss_lst_val_tmp.append(Loss.item())
                self.acc_lst_val_tmp.append(correctRatio)
                self.confAndCorr_lst_val_tmp.append(confAndCorrRatio)

                print(f' {idx}/{len(self.valDataloader)} th validation for {validatingNum} complete !!!')

                if idx >= self.MaxStepVal:
                    break


        torch.set_grad_enabled(True)
        self.MyLSTMModel.train()

    def valdatinStepEnd(self):

        self.loss_lst_trn.append(np.mean(self.loss_lst_trn_tmp))
        self.loss_lst_val.append(np.mean(self.loss_lst_val_tmp))

        self.acc_lst_trn.append(np.mean(self.acc_lst_trn_tmp))
        self.acc_lst_val.append(np.mean(self.acc_lst_val_tmp))

        self.confAndCorr_lst_trn.append(np.mean(self.confAndCorr_lst_trn_tmp))
        self.confAndCorr_lst_val.append(np.mean(self.confAndCorr_lst_val_tmp))

        self.flushLst()
        print('flushing lst complete!!!')


    def flushLst(self):

        self.loss_lst_trn_tmp.clear()
        self.loss_lst_val_tmp.clear()

        self.acc_lst_trn_tmp.clear()
        self.acc_lst_val_tmp.clear()

        self.confAndCorr_lst_trn_tmp.clear()
        self.confAndCorr_lst_val_tmp.clear()


    def TestStep(self,timeStamp=1514768940):
        pass




    def START_TRN_VAL(self,epoch):


        print('training step start....')
        self.trainingStep(trainingNum=epoch)
        print('training step complete!')

        print('Validation start.....')
        self.valdatingStep(validatingNum=epoch)
        print('Validation complete!')

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2 , 1)
        ax1.plot(range(len(self.loss_lst_trn)), self.loss_lst_trn)
        ax1.set_title('train loss')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(len(self.loss_lst_val)), self.loss_lst_val)
        ax2.set_title('val loss')
        plt.savefig(self.modelPlotSaveDir +  'LossResult.png', dpi=200)
        print('saving loss plot complete!')
        plt.cla()
        plt.clf()
        plt.close()




        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(len(self.acc_lst_trn)), self.acc_lst_trn)
        ax1.set_title('train acc')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(len(self.acc_lst_val)), self.acc_lst_val)
        ax2.set_title('val acc')
        plt.savefig(self.modelPlotSaveDir + 'AccResult.png', dpi=200)
        print('saving acc plot complete!')
        plt.cla()
        plt.clf()
        plt.close()





        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(len(self.confAndCorr_lst_trn)), self.confAndCorr_lst_trn)
        ax1.set_title('train conf')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(len(self.confAndCorr_lst_val)), self.confAndCorr_lst_val)
        ax2.set_title('val conf')
        plt.savefig(self.modelPlotSaveDir + 'ConfResult.png', dpi=200)
        print('saving confidence plot complete!')
        plt.cla()
        plt.clf()
        plt.close()



    def START_TEST(self):

        self.TestStep()