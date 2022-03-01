import os
import torch
import torch.nn as nn
from MY_MODELS import MyDeepAR
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
from MyDeepARDataset import DeepARDataset,DeepARTestDataset
import matplotlib.pyplot as plt


class DeepARPredictor(nn.Module):
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
                 muLin1,
                 muLin2,
                 sigLin1,
                 sigLin2,
                 seqLen,
                 windowRangeTrn,
                 windowRangeVal,
                 windowRangeTst,
                 MaxStepTrn,
                 MaxStepVal,
                 gpuUse,
                 XrangeNum,
                 sampleNum,
                 XtMethod,
                 sigmaNum=3,
                 bSizeTrn= 8,
                 bSizeVal=1,
                 bSizeTst=1,
                 lr=3e-4,
                 eps=1e-9):


        super(DeepARPredictor,self).__init__()

        self.data_folder_dir_trn = data_folder_dir_trn
        self.data_folder_dir_val = data_folder_dir_val
        self.data_folder_dir_test = data_folder_dir_test

        self.modelPlotSaveDir = modelPlotSaveDir

        self.iter_to_accumul = iter_to_accumul
        self.coin = coin
        self.gpuUse = gpuUse

        self.XtMethod = XtMethod

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayer=  numLayer
        self.muLin1 = muLin1
        self.muLin2 = muLin2
        self.sigLin1= sigLin1
        self.sigLin2 = sigLin2

        self.sampleNum = sampleNum
        self.seqLen = seqLen
        self.windowRangeTrn = windowRangeTrn
        self.windowRangeVal = windowRangeVal
        self.windowRangeTst = windowRangeTst
        self.XrangeNum = XrangeNum

        self.MaxStepTrn = MaxStepTrn
        self.MaxStepVal = MaxStepVal

        self.lr = lr
        self.eps = eps
        self.bSizeTrn = bSizeTrn
        self.bSizeVal = bSizeVal
        self.bSizeTst = bSizeTst

        self.sigmaNum =sigmaNum

        ###################MODEL SETTING###########################
        print('failed loading model, loaded fresh model')

        self.DeepArModel = MyDeepAR(
            inputSize=self.inputSize,
            hiddenSize=self.hiddenSize,
            numLayer=self.numLayer,
            muLin1=self.muLin1,
            muLin2 = self.muLin2,
            sigLin1=self.sigLin1,
            sigLin2 = self.sigLin2,
            dropRatio=0,
            rnnBase='lstm'
        )


        self.loss_lst_trn = []
        self.loss_lst_trn_tmp = []
        self.loss_lst_val = []
        self.loss_lst_val_tmp = []


        self.num4epoch = 0

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)

        self.optimizer = Adam(self.DeepArModel.parameters(),
                             lr=self.lr,  # 학습률
                             eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                             )

        MyTrnDataset = DeepARDataset(
            seqLen=self.seqLen,
            baseDir=self.data_folder_dir_trn,
            coin=self.coin,
            XtMethod=XtMethod,
            XrangeNum=self.XrangeNum
        )
        self.trainDataloader = DataLoader(MyTrnDataset,batch_size=self.bSizeTrn,shuffle=True)

        MyValDataset = DeepARDataset(
            seqLen=self.seqLen,
            baseDir=self.data_folder_dir_val,
            coin=self.coin,
            XtMethod=XtMethod,
            XrangeNum=self.XrangeNum
        )
        self.valDataloader = DataLoader(MyValDataset, batch_size=self.bSizeVal, shuffle=False)

        self.MyTestDataset = DeepARTestDataset(
            seqLen=self.seqLen,
            baseDir=self.data_folder_dir_test,
            coin=self.coin,
            XtMethod=XtMethod,
            windowRangeTst=windowRangeTst,
            XrangeNum=self.XrangeNum
        )

        self.DeepArModel.to(device=self.device)


    def forward(self,input,hidden,cell):

        input = input.to(self.device)
        hidden = hidden.to(self.device)
        cell = cell.to(self.device)

        mu,sig,hidden,cell = self.DeepArModel(input,hidden,cell)

        return mu,sig,hidden,cell


    def calLoss(self,mu,sig,label):
        #print(mu,sig,label)
        distribution = torch.distributions.normal.Normal(mu, sig)
        likelihood = (distribution.log_prob(label))

        #likelihood = torch.exp(likelihood)

        #print(-torch.mean(likelihood))

        return -torch.mean(likelihood)

    def init_hidden_cell(self,batchSize):

        return torch.zeros(self.numLayer, batchSize, self.hiddenSize)


    def trainingStep(self,trainingNum):

        self.DeepArModel.train()
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            globalTime= time.time()


            for idx,(bX,bZ) in enumerate(self.trainDataloader):

                localTime = time.time()

                TotalLoss = torch.zeros(1)
                hidden = self.init_hidden_cell(batchSize=bX.size(0))
                cell = self.init_hidden_cell(batchSize=bX.size(0))

                # bZ = bZ.float()
                # bInput = bInput.to(self.device)

                for t in range(self.windowRangeTrn):
                    if t == 0:
                        Z0 = torch.zeros(bX.size(0),5)

                        bInput = torch.unsqueeze(torch.cat((bX[:,t,:],Z0),dim=1),dim=1)
                    else:
                        bInput = torch.unsqueeze(torch.cat((bX[:, t, :], bZ[:, t - 1, :]), dim=1),dim=1)

                    mu,sig,hidden,cell = self.forward(bInput,hidden=hidden,cell=cell)

                    mu = mu.to('cpu')
                    sig = sig.to('cpu')
                    hidden = hidden.to('cpu')
                    cell = cell.to('cpu')
                    ResultLoss = self.calLoss(mu=mu,sig=sig,label=bZ[:,t,:])
                    print(f'mu : {mu[0,3].item()} , label : {bZ[0, t, :][3].item()},'
                          f' diff : {abs(mu[0,3].item() - bZ[0, t, :][3].item())}')

                    TotalLoss += ResultLoss

                self.loss_lst_trn_tmp.append(TotalLoss.item())
                TotalLoss.backward()
                if (idx + 1) % self.iter_to_accumul == 0:
                    localTimeElaps = round(time.time() - localTime, 2)
                    globalTimeElaps = round(time.time() - globalTime, 2)
                    print(
                        f'globaly {globalTimeElaps} elapsed and locally {localTimeElaps} '
                        f'elapsed for {idx} / {self.MaxStepTrn}'
                        f' of epoch : {trainingNum}/{len(self.trainDataloader)}'
                        f' with loss : {TotalLoss.item()}')
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.loss_lst_trn.append(np.sum(self.loss_lst_trn_tmp))
                    self.loss_lst_trn_tmp = []

                if idx >= self.MaxStepTrn:
                    break

        torch.set_grad_enabled(False)
        self.DeepArModel.eval()

    def valdatingStep(self,validatingNum):

        self.DeepArModel.eval()
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            for idx,(bXVal,bZVal) in enumerate(self.valDataloader):

                TotalLoss = torch.zeros(1)
                hidden = self.init_hidden_cell(batchSize=bXVal.size(0))
                cell = self.init_hidden_cell(batchSize=bXVal.size(0))

                # bZVal = bZVal.float()

                for t in range(self.windowRangeVal):
                    if t == 0:
                        Z0Val = torch.zeros(bXVal.size(0),5)
                        bInputVal = torch.unsqueeze(torch.cat((bXVal[:, t, :], Z0Val), dim=1),dim=1)
                    else:
                        bInputVal = torch.unsqueeze(torch.cat((bXVal[:, t, :], bZVal[:, t - 1, :]), dim=1),dim=1)

                    mu, sig, hidden, cell = self.forward(bInputVal, hidden=hidden, cell=cell)
                    mu = mu.to('cpu')
                    sig = sig.to('cpu')
                    hidden = hidden.to('cpu')
                    cell = cell.to('cpu')

                    print(f'mu : {mu[0,3].item()} , label : {bZVal[0, t, :][3].item()},'
                          f' diff : {abs(mu[0,3].item() - bZVal[0, t, :][3].item())}')

                    ResultLoss = self.calLoss(mu=mu,sig=sig,label=bZVal[:,t,:])
                    TotalLoss += ResultLoss

                self.loss_lst_val_tmp.append(TotalLoss.item())
                if (idx + 1) % self.iter_to_accumul == 0:
                    self.loss_lst_val.append(np.sum(self.loss_lst_val_tmp))
                    self.loss_lst_val_tmp = []

                if idx >= self.MaxStepVal:
                    break

            print(f'validation complete with mean loss : {self.loss_lst_val[-1]}')


        torch.set_grad_enabled(True)
        self.DeepArModel.train()

    def SupEndValue(self,zPred):

        if zPred[3] > zPred[1]:
            zPred[3] = zPred[1]

        elif zPred[3] < zPred[2]:
            zPred[3] = zPred[2]

        return zPred

    def TestStep(self,timeStamp=1514768940):

        self.DeepArModel.eval()
        self.optimizer.zero_grad()

        ResultLst = []
        predLst = []
        labelLst = []
        lowerBoundLst = []
        upperBoundLst = []

        with torch.set_grad_enabled(False):

            bXTst,bZTst = self.MyTestDataset.getItem(timeStamp=timeStamp)

            samples = torch.zeros(1,self.sampleNum,self.seqLen - self.windowRangeTst)

            TotalLoss = torch.zeros(1)
            hidden = self.init_hidden_cell(batchSize=bXTst.size(0))
            cell = self.init_hidden_cell(batchSize=bXTst.size(0))

            bZTst = bZTst.float()
            bXTst = bXTst.float()

            for t in range(self.windowRangeTst):
                if t == 0:
                    Z0Tst = torch.zeros(bXTst.size(0),5)
                    bInputTst = torch.unsqueeze(torch.cat((bXTst[:, t, :], Z0Tst), dim=1),dim=1)
                else:
                    bInputTst = torch.unsqueeze(torch.cat((bXTst[:, t, :], bZTst[:, t - 1, :]), dim=1),dim=1)

                mu, sig, hidden, cell = self.forward(bInputTst, hidden=hidden, cell=cell)

                mu = mu.to('cpu')
                sig = sig.to('cpu')
                hidden = hidden.to('cpu')
                cell = cell.to('cpu')

                bzTst_copy = bZTst[:, t, :].clone().detach().to('cpu')
                mu_copy = mu.clone().detach().to('cpu')
                sig_copy = sig.clone().detach().to('cpu')
                # print(f'mu is : {mu} and bzTst is {bzTst_copy}')
                print(f'mu : {mu[3].item()} and label : {bzTst_copy[0,3].item()}'
                      f' diff : {abs(mu[3].item() - bzTst_copy[0,3].item())}')


                predLst.append(bzTst_copy[0,3])
                labelLst.append(bzTst_copy[0,3])
                lowerBoundLst.append(mu_copy[3]-self.sigmaNum*sig_copy[3])
                upperBoundLst.append(mu_copy[3]+self.sigmaNum*sig_copy[3])


            for samplestep in range(self.sampleNum):

                hidden4sample = hidden
                cell4sample = cell

                for t in range(self.windowRangeTst,self.seqLen):
                    if t == self.windowRangeTst:
                        bInputTst = torch.unsqueeze(torch.cat((bXTst[:, t, :], bZTst[:, t - 1, :]), dim=1),dim=1)

                        mu, sig, hidden4sample, cell4sample = self.forward(bInputTst,
                                                                           hidden=hidden4sample,
                                                                           cell=cell4sample)

                        gaussian = torch.distributions.normal.Normal(mu,sig)
                        z_pred = gaussian.sample()
                        z_pred = z_pred.to('cpu')
                        z_pred = self.SupEndValue(z_pred)

                        z_pred_copy = z_pred.clone().detach().to('cpu')

                        samples[0,samplestep,t-self.windowRangeTst] = z_pred_copy[3]

                        bzTst_copy = bZTst[:,t,:].clone().detach().to('cpu')

                        if samplestep == 0:
                            labelLst.append(bzTst_copy[0,3])

                        print(f'z_pred : {z_pred[3].item()} and label : {bzTst_copy[0,3].item()}'
                              f' diff : {abs(z_pred[3].item()-bzTst_copy[0,3].item())}')
                        mu_copy = mu.clone().detach().to('cpu')
                        sig_copy = sig.clone().detach().to('cpu')

                    else:

                        bInputTst = torch.unsqueeze(torch.cat((bXTst[:, t, :],
                                                               torch.unsqueeze(z_pred,dim=0))
                                                              , dim=1),dim=1)
                        mu, sig, hidden4sample, cell4sample = self.forward(bInputTst,
                                                                           hidden=hidden4sample,
                                                                           cell=cell4sample)
                        gaussian = torch.distributions.normal.Normal(mu, sig)
                        z_pred = gaussian.sample()
                        z_pred = z_pred.to('cpu')
                        z_pred = self.SupEndValue(z_pred)

                        z_pred_copy = z_pred.clone().detach().to('cpu')
                        samples[0, samplestep, t - self.windowRangeTst] = z_pred_copy[3]

                        bzTst_copy = bZTst[:, t, :].clone().detach().to('cpu')
                        if samplestep == 0:
                            labelLst.append(bzTst_copy[0,3])
                        print(f'z_pred : {z_pred[3].item()} and label : {bzTst_copy[0,3].item()}'
                              f' diff : {abs(z_pred[3].item()-bzTst_copy[0,3].item())}')
                        mu_copy = mu.clone().detach().to('cpu')
                        sig_copy = sig.clone().detach().to('cpu')

            sample_mu = torch.mean(samples,dim=1).squeeze()
            sample_sig = samples.std(dim=1).squeeze()

            for i in range(len(sample_mu)):
                predLst.append(sample_mu[i])
                lowerBoundLst.append(sample_mu[i] - self.sigmaNum * sample_sig[i])
                upperBoundLst.append(sample_mu[i] + self.sigmaNum * sample_sig[i])


            plt.plot(range(len(predLst)),predLst,'r')
            plt.plot(range(len(labelLst)), labelLst,'b')
            plt.fill_between(range(len(predLst)),lowerBoundLst,upperBoundLst,color='g',alpha=.1)
            plt.savefig(self.modelPlotSaveDir+'TestResult_'+str(timeStamp)+'.png',dpi=200)
            plt.show()
            plt.close()

        torch.set_grad_enabled(True)
        self.DeepArModel.train()


    def START_TRN_VAL(self,epoch,MaxEpoch):


        print('training step start....')
        self.trainingStep(trainingNum=epoch)
        print('training step complete!')

        print('Validation start.....')
        self.valdatingStep(validatingNum=epoch)
        print('Validation complete!')

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.plot(range(len(self.loss_lst_trn)), self.loss_lst_trn)
        ax1.set_title('train loss')

        ax3 = fig.add_subplot(1, 4, 3)
        ax3.plot(range(len(self.loss_lst_val)), self.loss_lst_val)
        ax3.set_title('val loss')


        plt.savefig(self.modelPlotSaveDir +  'LossResult.png', dpi=200)
        print('saving plot complete!')
        plt.close()

        print(f'num4epoch is : {epoch} and self.max_epoch : {MaxEpoch}')

    def START_TEST(self):

        self.TestStep()