import torch
from torch.utils.data import Dataset
import csv
import numpy as np

class DeepARDataset(Dataset):
    def __init__(self,seqLen,baseDir,coin='Bitcoin'):
        super(DeepARDataset).__init__()

        self.baseDir = baseDir
        self.coin = coin
        self.loadDataDir = self.baseDir+self.coin+'.csv'

        self.seqLen = seqLen

        with open(self.loadDataDir,'r') as f:
            rdr = csv.reader(f)
            self.totalDataLst = list(rdr)

        self.totalDataLst= self.totalDataLst[1:] # remove header

        self.totalDataArr = np.array(self.totalDataLst,dtype=np.float32)

        self.ZtTensor = torch.as_tensor(np.reshape(self.totalDataArr[:,6],(-1,1)))

        self.XtTensor = torch.as_tensor(np.concatenate((self.totalDataArr[:,3:6],np.reshape(self.totalDataArr[:,7],(-1,1))),axis=1))

        del self.totalDataArr
        del self.totalDataLst

    def __len__(self):
        return len(self.XtTensor)- self.seqLen-1

    def __getitem__(self, idx):

        Xt = self.XtTensor[idx:idx+self.seqLen,:]
        firstDayOpen = Xt[0,0].clone().detach()

        Zt = self.ZtTensor[idx:idx+self.seqLen,:]

        Xt = Xt / firstDayOpen
        Zt = Zt / firstDayOpen

        # print(Xt[:,1])

        return Xt, Zt


class DeepARTestDataset():
    def __init__(self,seqLen,baseDir,coin='Bitcoin'):

        self.baseDir = baseDir
        self.coin = coin
        self.loadDataDir = self.baseDir+self.coin+'.csv'

        self.seqLen = seqLen

        with open(self.loadDataDir,'r') as f:
            rdr = csv.reader(f)
            self.totalDataLst = list(rdr)

        self.totalDataLst= self.totalDataLst[1:] # remove header

        self.totalDataArr = np.array(self.totalDataLst,dtype=np.float64)

        self.timeStampArr= self.totalDataArr[:,0]
        self.ZtTensor = torch.as_tensor(np.reshape(self.totalDataArr[:,6],(-1,1)))
        self.XtTensor = torch.as_tensor(np.concatenate((self.totalDataArr[:,3:6],np.reshape(self.totalDataArr[:,7],(-1,1))),axis=1))

        del self.totalDataArr
        del self.totalDataLst


    def getItem(self, timeStamp):

        idx = np.where(self.timeStampArr == timeStamp)[0][0]

        Xt = self.XtTensor[idx:idx+self.seqLen,:]
        firstDayOpen = Xt[0,0].clone().detach()

        Zt = self.ZtTensor[idx:idx+self.seqLen,:]

        Xt = Xt / firstDayOpen
        Zt = Zt / firstDayOpen


        return Xt.unsqueeze(0), Zt.unsqueeze(0)


# x = DeepARTestDataset(seqLen=128,baseDir='/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/dataset/test/')
#
# ITEM = x.getItem(timeStamp=float(1514768940
#
# )
#
# )
#
# print(I








