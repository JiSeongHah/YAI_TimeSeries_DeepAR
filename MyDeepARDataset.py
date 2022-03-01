import torch
from torch.utils.data import Dataset
import csv
import numpy as np

class DeepARDataset(Dataset):
    def __init__(self,XrangeNum,seqLen,baseDir,coin='Bitcoin',XtMethod='real'):
        super(DeepARDataset).__init__()

        self.baseDir = baseDir
        self.coin = coin
        self.loadDataDir = self.baseDir+'new'+self.coin+'.csv'

        self.XtMethod = XtMethod
        self.seqLen = seqLen

        self.XrangeNum = XrangeNum

        with open(self.loadDataDir,'r') as f:
            rdr = csv.reader(f)
            self.totalDataLst = list(rdr)

        self.totalDataLst = self.totalDataLst[1:] # remove header

        self.totalDataArr = np.array(self.totalDataLst,dtype=np.float32)

        self.ZtTensor = torch.as_tensor(self.totalDataArr[:,3:8])

        del self.totalDataArr
        del self.totalDataLst

    def __len__(self):
        return len(self.ZtTensor)- 1*self.seqLen -self.XrangeNum*self.seqLen-1

    def __getitem__(self, idx):

        for i in range(self.XrangeNum):
            rawXt = self.ZtTensor[idx+i*self.seqLen:idx+(i+1)*self.seqLen,:]

            if self.XtMethod == 'real':
                XtStart= rawXt[0,0]
                XtHighest = torch.max(rawXt)
                XtLowest = torch.min(rawXt)
                XtEnd = rawXt[-1,3]
                XtMeanVol = torch.mean(rawXt[:,-1])

                rawXt = torch.tensor([XtStart,XtHighest,XtLowest,XtEnd,XtMeanVol])
            if self.XtMethod == 'mean':
                rawXt = torch.mean(rawXt,dim=0)

            if i == 0:
                Xt = rawXt
            if i != 0:
                Xt = torch.cat((Xt,rawXt),dim=0)

        Zt = self.ZtTensor[idx+self.XrangeNum*self.seqLen:idx+self.XrangeNum*self.seqLen+self.seqLen,:]

        firstDayOpen = Zt[0, 0].clone().detach()

        Xt = Xt / firstDayOpen
        # print(f'size of Xt is : {Xt.size()}')
        Xt = Xt.expand(Zt.size(0), -1)
        # print(f'size of Xt is : {Xt.size()}')
        Zt = Zt / firstDayOpen
        # print(f'size of Zt is : {Zt.size()}')

        # Xt = torch.log(Xt / firstDayOpen)
        # Zt = torch.log(Zt / firstDayOpen)

        # print(Xt[:,1])

        return Xt, Zt


class DeepARTestDataset():
    def __init__(self,XrangeNum,seqLen,baseDir,windowRangeTst,coin='Bitcoin',XtMethod='real'):

        self.baseDir = baseDir
        self.coin = coin
        self.loadDataDir = self.baseDir+'new'+self.coin+'.csv'

        self.XtMethod = XtMethod

        self.seqLen = seqLen
        self.windowRangeTst = windowRangeTst

        self.XrangeNum = XrangeNum

        with open(self.loadDataDir,'r') as f:
            rdr = csv.reader(f)
            self.totalDataLst = list(rdr)

        self.totalDataLst= self.totalDataLst[1:] # remove header

        self.totalDataArr = np.array(self.totalDataLst,dtype=np.float64)

        self.timeStampArr= self.totalDataArr[:,0]
        self.ZtTensor = torch.as_tensor(self.totalDataArr[:, 3:8])

        del self.totalDataArr
        del self.totalDataLst

    def getItem(self, timeStamp):

        idx = np.where(self.timeStampArr == timeStamp)[0][0]
        print(f'dir is : {self.loadDataDir}')
        print(f'idx is : {idx} and timestamp is : {timeStamp}')

        assert idx - self.XrangeNum*self.seqLen-self.windowRangeTst >=0 ,\
            "idx got out of range, idx - self.XrangeNum x seqLen-windowRange must be >= 0"

        for i in range(self.XrangeNum):
            rawXt = self.ZtTensor[idx-self.windowRangeTst-(i+1)*self.seqLen+1:idx-self.windowRangeTst-i*self.seqLen+1,:]

            if self.XtMethod == 'real':
                XtStart= rawXt[0,0]
                XtHighest = torch.max(rawXt)
                XtLowest = torch.min(rawXt)
                XtEnd = rawXt[-1,3]
                XtMeanVol = torch.mean(rawXt[:,-1])

                rawXt = torch.tensor([XtStart,XtHighest,XtLowest,XtEnd,XtMeanVol])
            if self.XtMethod == 'mean':
                rawXt = torch.mean(rawXt,dim=0)

            if i ==0:
                Xt = rawXt
            if i != 0:
                Xt = torch.cat((Xt,rawXt),dim=0)

        Zt = self.ZtTensor[idx-self.windowRangeTst+1:idx+self.seqLen-self.windowRangeTst+1,:]


        firstDayOpen = Zt[0,0].clone().detach()

        Xt = Xt / firstDayOpen
        Xt= Xt.expand(Zt.size(0),-1)
        Zt = Zt / firstDayOpen

        return Xt.unsqueeze(0), Zt.unsqueeze(0)










