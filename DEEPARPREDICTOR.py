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

class DeepARPredictor(nn.Module):
    def __init__(self,
                 data_folder_dir_trn,
                 data_folder_dir_val,
                 data_folder_dir_test,
                 modelPlotSaveDir,
                 iter_to_accumul,
                 inputSize,
                 hiddenSize,
                 numLayer,
                 muLin1,
                 muLin2,
                 sigLin1,
                 sigLin2,
                 seqLen,
                 windowRange,

                 bSizeTrn= 8,
                 bSizeVal=1,
                 lr=3e-4,
                 eps=1e-9):


        super(DeepARPredictor,self).__init__()

        self.data_folder_dir_trn = data_folder_dir_trn
        self.data_folder_dir_val = data_folder_dir_val
        self.data_folder_dir_test = data_folder_dir_test

        self.iter_to_accumul = iter_to_accumul

        self.inputSize = inputSize
        self.hiddenSzie = hiddenSize
        self.numLayer=  numLayer
        self.muLin1 = muLin1
        self.muLin2 = muLin2
        self.sigLin1= sigLin1
        self.sigLin2 = sigLin2

        self.seqLen = seqLen
        self.windowRange = windowRange

        self.lr = lr
        self.eps = eps
        self.bSizeTrn = bSizeTrn
        self.bSizeVal = bSizeVal

        self.modelPlotSaveDir = modelPlotSaveDir

        ###################MODEL SETTING###########################
        print('failed loading model, loaded fresh model')

        self.DeepArModel = MyDeepAR(
            inputSize=self.inputSize,
            hiddenSize=self.hiddenSzie,
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

        MyTrnDataset = MyEelDataset(data_folder_dir=self.data_folder_dir_trn,tLabelDir=self.labelDir,TEST=False,CROP=self.CROP)

        self.trainDataloader = DataLoader(MyTrnDataset,batch_size=self.bSizeTrn,shuffle=True)

        MyValDataset = MyEelDataset(data_folder_dir=self.data_folder_dir_val, tLabelDir=self.labelDir,TEST=False,CROP=self.CROP)
        self.valLen = int(len(MyValDataset)/self.bSizeVal)
        if self.valLen < self.MaxStepVal:
            self.MaxStepVal = self.valLen
        self.valDataloader = DataLoader(MyValDataset, batch_size=self.bSizeVal, shuffle=False)

        MyTestDataset = MyEelDataset(data_folder_dir=self.data_folder_dir_test,tLabelDir=self.labelDir,TEST=True,CROP=self.CROP)
        self.testLen = len(MyTestDataset)
        self.TestDataloader = DataLoader(MyTestDataset,batch_size=1,shuffle=False)

        self.DeepArModel.to(device=self.device)


    def forward(self,input,hidden,cell):

        input = input.to(self.device)
        hidden = hidden.to(self.device)
        cell = cell.to(self.device)

        mu,sig,hidden,cell = self.EelModel(input,hidden,cell)

        return mu,sig,hidden,cell


    def calLoss(self,mu,sig,label):

        distribution = torch.distributions.normal.Normal(mu, sig)
        likelihood = distribution.log_prob(label)
        return -torch.mean(likelihood)

    def init_hidden_cell(self, input_size):
        return torch.zeros(self.numLayer, self.inputSize, self.hiddenSize)


    def trainingStep(self,trainingNum):

        self.DeepArModel.train()
        countNum = 0

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            globalTime= time.time()

            for idx,_,bInput, bLabel  in enumerate(self.trainDataloader):

                localTime = time.time()

                TotalLoss = torch.zeros(1)
                hidden = self.init_hidden_cell(self.bSizeTrn)
                cell = self.init_hidden_cell(self.bSizeTrn)

                bLabel = bLabel.float()
                # bInput = bInput.to(self.device)

                for t in range(self.windowRange):
                    mu,sig,hidden,cell = self.forward(bInput[:,t,:])
                    mu = mu.to('cpu')
                    sig = sig.to('cpu')
                    hidden = hidden.to('cpu')
                    cell = cell.to('cpu')
                    ResultLoss = self.calLoss(mu=mu,sig=sig,label=bLabel)

                    TotalLoss += ResultLoss


                ResultLoss.backward()
                self.loss_lst_trn_tmp.append(10000*float(ResultLoss.item()))

                if (countNum + 1) % self.iter_to_accumul == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if countNum == self.MaxStep:
                    break
                else:
                    countNum += 1

                localTimeElaps = round(time.time() - localTime,2)
                globalTimeElaps = round(time.time() - globalTime,2)

                print(f'globaly {globalTimeElaps} elapsed and locally {localTimeElaps} elapsed for {countNum} / {self.MaxStep}'
                      f' of epoch : {trainingNum}/{self.MaxEpoch}'
                      f' with loss : {10000*float(ResultLoss.item())}')

        self.loss_lst_trn.append(self.iter_to_accumul * np.mean(self.loss_lst_trn_tmp))
        print(f'training complete with mean loss : {self.iter_to_accumul * np.mean(self.loss_lst_trn_tmp)}')
        self.loss_lst_trn_tmp = []

        torch.set_grad_enabled(False)
        self.EelModel.eval()

    def valdatingStep(self,validatingNum):

        self.EelModel.eval()
        countNum = 0
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            for _,valBInput, valBLabel in self.valDataloader:

                valBInput = valBInput.to(self.device)

                valBLogit = self.forward(valBInput)
                valBLogit = valBLogit.cpu()

                ResultLoss = self.calLoss(valBLogit,valBLabel)

                ResultLoss = ResultLoss / self.iter_to_accumul
                self.loss_lst_val_tmp.append(10000*float(ResultLoss.item()))

                print(f'{countNum}/ {self.MaxStepVal} th val of epoch : {validatingNum} complete with loss : {10000 * float(ResultLoss.item())}')

                if countNum == self.MaxStepVal:
                    break
                else:
                    countNum += 1

            self.loss_lst_val.append(self.iter_to_accumul*np.mean(self.loss_lst_val_tmp))
            print(f'validation complete with mean loss : {self.iter_to_accumul*np.mean(self.loss_lst_val_tmp)}')
            self.loss_lst_val_tmp = []

        torch.set_grad_enabled(True)
        self.EelModel.train()

    def TestStep(self):

        self.EelModel.eval()
        countNum = 0
        self.optimizer.zero_grad()

        ResultDict = dict()

        with torch.set_grad_enabled(False):
            for ImageName,TestBInput in self.TestDataloader:

                ImageName = ImageName[0]

                TestBInput = (TestBInput.float()).to(self.device)

                TestBLogit = self.forward(TestBInput)
                TestBLogit = TestBLogit.cpu()

                if ImageName not in ResultDict:
                    ResultDict[str(ImageName)] = [100*TestBLogit.item()]
                if ImageName in ResultDict:
                    ResultDict[str(ImageName)].append(100*TestBLogit.item())
                print(f'{countNum} / {self.testLen} Pred done  data : {[str(ImageName),100*TestBLogit]}')
                countNum +=1

        print('Start saving Result.....')

        with open(self.modelPlotSaveDir+'resultDict.pkl','wb') as f:
            pickle.dump(ResultDict,f)

        header = ['ImageDir','AvgWeight']
        with open(self.modelPlotSaveDir+'sample_submission.csv','w') as f:
            wr = csv.writer(f)
            wr.writerow(header)
            for ImageKey in ResultDict.keys():

                wr.writerow([str(ImageKey),np.mean(ResultDict[ImageKey])])
                print(f'appending {ImageKey} with {ResultDict[ImageKey]} complete')


        torch.set_grad_enabled(True)
        self.EelModel.train()


    def START_TRN_VAL(self,epoch):


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


        plt.savefig(self.modelPlotSaveDir +  'Result.png', dpi=300)
        print('saving plot complete!')
        plt.close()

        print(f'num4epoch is : {epoch} and self.max_epoch : {self.MaxEpoch}')