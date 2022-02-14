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
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression

class EelPredictor(nn.Module):
    def __init__(self,
                 data_folder_dir_trn,
                 data_folder_dir_val,
                 labelDir,
                 data_folder_dir_test,
                 modelPlotSaveDir,
                 ratioLst,
                 newDataNum,
                 DataAugflg= True

                 ):


        super(EelPredictor,self).__init__()

        self.data_folder_dir_trn = data_folder_dir_trn
        self.data_folder_dir_val = data_folder_dir_val
        self.data_folder_dir_test = data_folder_dir_test

        self.labelDir = labelDir

        self.DataAugflg = DataAugflg

        self.modelPlotSaveDir = modelPlotSaveDir

        self.loss_lst_trn = []
        self.loss_lst_trn_tmp = []
        self.loss_lst_val = []
        self.loss_lst_val_tmp = []

        self.acc_lst_trn = []
        self.acc_lst_trn_tmp = []

        self.acc_lst_val = []
        self.acc_lst_val_tmp = []

        self.num4epoch = 0


        self.ratioLst=  ratioLst
        self.newDataNum = newDataNum

        self.TrnX = []
        self.TrnY = []

        self.TesX = []
        self.TesName = []


        MyTrnDataset = MyEelDataset(data_folder_dir=self.data_folder_dir_trn,tLabelDir=self.labelDir,TRAIN=True)
        MyValDataset = MyEelDataset(data_folder_dir=self.data_folder_dir_val, tLabelDir=self.labelDir,TRAIN=True)


        MyTestDataset = MyEelDataset(data_folder_dir=self.data_folder_dir_test,tLabelDir=self.labelDir,TRAIN=False)
        self.testLen = len(MyTestDataset)

        for i in MyTrnDataset:
            self.TrnX.append(i[1].numpy())
            self.TrnY.append(i[2].numpy())
            print(self.TrnX[-1],self.TrnY[-1])

        for i in MyValDataset:
            self.TrnX.append(i[1].numpy())
            self.TrnY.append(i[2].numpy())

        for i in MyTestDataset:
            self.TesX.append(i[1].numpy())
            self.TesName.append(i[0])

        print('before : ',len(self.TrnX),len(self.TrnY))

        if self.DataAugflg == True:
            self.TrnX, self.TrnY = self.DataAug(DataXLst=self.TrnX,DataYLst=self.TrnY,
                                                ratioLst=self.ratioLst,newDataNum=self.newDataNum)

        print('after : ', len(self.TrnX), len(self.TrnY))

        self.TrnX = np.array(self.TrnX)
        self.TrnY = np.array(self.TrnY)
        self.TesX = np.array(self.TesX)

        self.model = LinearRegression()

    def DataAug(self,DataXLst,DataYLst,ratioLst,newDataNum):


        for i in range(newDataNum):
            for ratio in ratioLst:
                randNum = np.random.choice(len(DataXLst),2)

                newDataX = DataXLst[randNum[0]]*ratio + DataXLst[randNum[1]]*(1-ratio)
                newDataY = DataYLst[randNum[0]]*ratio + DataYLst[randNum[1]]*(1-ratio)

                DataXLst.append(newDataX)
                DataYLst.append(newDataY)

        return DataXLst , DataYLst

    def trainingStep(self,):

        self.model.fit(self.TrnX,self.TrnY)



    def TestStep(self):

        TestResult = self.model.predict(self.TesX)

        header = ['ImageDir','AvgWeight']
        with open(self.modelPlotSaveDir+'sample_submission.csv','w') as f:
            wr = csv.writer(f)
            wr.writerow(header)
            for idx,i in enumerate(TestResult):
                wr.writerow([self.TesName[idx],100*i])
                print(f'appending {i} complete')


if __name__ == '__main__':



    baseDir = '/home/a286winteriscoming/Downloads/EelPred/dataset/dataset/'
    #baseDir = '/home/a286/hjs_dir1/Dacon1/'

    data_folder_dir_trn = baseDir + 'train/'
    data_folder_dir_val  = baseDir + 'val/'
    labelDir = baseDir + 'train.csv'
    data_folder_dir_test = baseDir + 'test/'
    save_range= 5
    ratioLst= [(2*i+1)/10 for i in range(5)]
    print(ratioLst)

    newDataNum = 300

    savingDir = mk_name(model='linReg')
    modelPlotSaveDir = baseDir +savingDir + '/'
    createDirectory(modelPlotSaveDir)

    MODEL_START  = EelPredictor(
        data_folder_dir_trn=data_folder_dir_trn,
        data_folder_dir_val=data_folder_dir_val,
        labelDir=labelDir,
        modelPlotSaveDir=modelPlotSaveDir,
        data_folder_dir_test=data_folder_dir_test,
        ratioLst=ratioLst,
        newDataNum=newDataNum
    )


    MODEL_START.trainingStep()
    print('Training Complete!')
    print('Test Start....')
    MODEL_START.TestStep()

    # for i in range(10000):
    #     MODEL_START.START_TRN_VAL()
    #
    #     if i%save_range ==0:
    #         if i > 15000:
    #             break
    #
    #         try:
    #             torch.save(MODEL_START, modelPlotSaveDir + str(i) + '.pth')
    #             print('saving model complete')
    #             print('saving model complete')
    #             print('saving model complete')
    #             print('saving model complete')
    #             print('saving model complete')
    #             time.sleep(5)
    #         except:
    #             print('saving model failed')
    #             print('saving model failed')
    #             print('saving model failed')
    #             print('saving model failed')
    #             print('saving model failed')
    #             time.sleep(5)

























