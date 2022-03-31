import torch
from torch.utils.data import Dataset
import csv
import numpy as np

class vanillaLSTMDataset(Dataset):
    def __init__(self,
                 afterRangeNum,
                 seqLen,
                 FeeRate,
                 commitThreshold,
                 baseDir,
                 coin='Bitcoin',
                 labelMethod='UpDown'
                 ):
        super(vanillaLSTMDataset).__init__()

        self.baseDir = baseDir
        self.coin = coin
        self.loadDataDir = self.baseDir+'new'+self.coin+'.csv'

        self.labelMethod = labelMethod
        self.seqLen = seqLen
        self.FeeRate = FeeRate
        self.commitThreshold = commitThreshold


        self.afterRangeNum = afterRangeNum

        with open(self.loadDataDir,'r') as f:
            rdr = csv.reader(f)
            self.totalDataLst = list(rdr)

        self.totalDataLst = self.totalDataLst[1:] # remove header

        self.totalDataArr = np.array(self.totalDataLst,dtype=np.float32)

        self.DataTensor = torch.as_tensor(self.totalDataArr[:,3:8])

        del self.totalDataArr
        del self.totalDataLst

    def __len__(self):
        return len(self.DataTensor) - self.seqLen-self.afterRangeNum

    def __getitem__(self, idx):

        Input = self.DataTensor[idx:idx+self.seqLen , :]

        output = self.DataTensor[idx+self.seqLen : idx+self.seqLen + self.afterRangeNum , : ]

        ratioLower = torch.mean(output[:,2])/Input[-1,3]
        ratioUpper = torch.mean(output[:,1])/Input[-1,3]

        if ratioLower > 0:
            if ratioLower -self.FeeRate > self.commitThreshold:
                label = torch.tensor(0)
            else:
                label = torch.tensor(2)
        elif ratioUpper < 0:
            if abs(ratioUpper) -self.FeeRate > self.commitThreshold:
                label= torch.tensor(1)
            else:
                label= torch.tensor(2)
        else:
            label= torch.tensor(2)

        return Input,label


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

#
#
# x = vanillaLSTMDataset(
#     afterRangeNum =3,
#     seqLen =3,
#     FeeRate = 3,
#     commitThreshold =0.1,
#     baseDir = '/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/dataset/train/',
#     coin='Bitcoin',
#     labelMethod='UpDown'
# )
#
# xxx= 0
# for i in x:
#     print(len(x.DataTensor))
#     xxx +=1
#     if xxx == 10:
#         break
#
#
# x = vanillaLSTMDataset(
#     afterRangeNum =3,
#     seqLen =3,
#     FeeRate = 3,
#     commitThreshold =0.1,
#     baseDir = '/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/dataset/val/',
#     coin='Bitcoin',
#     labelMethod='UpDown'
# )
#
# xxx=0
# for i in x:
#     print(len(x.DataTensor))
#     xxx +=1
#     if xxx == 10:
#         break
#
#
#
#
#
#
