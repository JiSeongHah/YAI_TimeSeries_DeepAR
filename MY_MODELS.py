import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

class simpleDNN(nn.Module):

    def __init__(self,innerNum):
        super(simpleDNN, self).__init__()
        print(f'innerNUm is : {innerNum}')

        self.innerNum = innerNum

        self.lin1 = nn.Linear(in_features=2,out_features=self.innerNum)
        #self.lin1 = nn.Linear(2,self.innerNum)
        self.lin2 = nn.Linear(in_features=self.innerNum,out_features=self.innerNum)
        self.lin3 = nn.Linear(in_features=self.innerNum, out_features=self.innerNum)
        self.lin4 = nn.Linear(in_features=self.innerNum, out_features=1)

    def forward(self, x):
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = self.lin4(out)

        return out


class MyDeepAR(torch.nn.Module):

    def __init__(self,inputSize,
                 hiddenSize,
                 numLayer,
                 muLin1,
                 muLin2,
                 sigLin1,
                 sigLin2,
                 dropRatio=0,
                 rnnBase='lstm'):
        super(MyDeepAR, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayer =  numLayer
        self.rnnBase = rnnBase

        self.dropRatio = dropRatio

        self.muLin1 = muLin1
        self.muLin2 = muLin2
        self.sigLin1 = sigLin1
        self.sigLin2 = sigLin2

        if self.rnnBase =='lstm':
            self.lstm = nn.LSTM(
                input_size=self.inputSize,
                hidden_size=self.hiddenSize,
                num_layers=numLayer,
                bias=True,
                batch_first=True,
                dropout= self.dropRatio
            )
            for names in self.lstm._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.lstm, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.)


        self.mu1 = nn.Linear(in_features=self.hiddenSize,out_features=self.muLin1)
        self.mu2 = nn.Linear(in_features=self.muLin1, out_features=self.muLin2)
        self.mu3 = nn.Linear(in_features=self.muLin2,out_features=1)

        self.sig1 = nn.Linear(in_features=self.hiddenSize, out_features=self.sigLin1)
        self.sig2 = nn.Linear(in_features=self.sigLin1, out_features=self.sigLin2)
        self.sig3 = nn.Linear(in_features=self.sigLin2, out_features=1)
        self.distribution_sigma = nn.Softplus()


    def forward(self, x,hidden,cell):


        lstmOut, (hidden,cell) = self.lstm(x,(hidden,cell))


        muOut = self.mu1(lstmOut)
        muOut = self.mu2(muOut)
        muOut = self.mu3(muOut)


        sigOut = self.sig1(lstmOut)
        sigOut = self.sig2(sigOut)
        sigOut = self.sig3(sigOut)
        sigOut = self.distribution_sigma(sigOut)


        return torch.squeeze(muOut), torch.squeeze(sigOut), hidden, cell





