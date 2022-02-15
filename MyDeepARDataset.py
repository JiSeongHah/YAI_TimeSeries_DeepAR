import torch
from torch.utils.data import Dataset
import csv
import numpy as np

class DeepARDataset(Dataset):
    def __init__(self,seqLen,baseDir,coin='Bitcoin'):
        super(DeepARDataset).__init__()

        self.baseDir = baseDir
        self.coin = coin
        self.loadDataDir = self.baseDi+self.coin

        self.seqLen = seqLen

        with open(self.loadDataDir,'r') as f:
            rdr = csv.reader(f)
            self.totalDataLst = list(rdr)

        self.totalDataLst= self.totalDataLst[1:] # remove header

        self.totalDataArr = np.array(self.totalDataLst)

        self.ZtArr = self.totalDataArr[:,6]

        self.XtArr = np.concatenate((self.totalDataArr[:,3:6],self.totalDataArr[:,7]),axis=1)

        del self.totalDataArr
        del self.totalDataLst

    def __len(self):
        return len(self.XtArr)

    def __getitem__(self, idx):

        Xt = self.XtArr[idx,:]

        Zt = self.ZtArr[idx,:]


        return Xt, Zt













