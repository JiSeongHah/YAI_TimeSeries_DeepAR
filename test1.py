import os
import torch.nn as nn
import torch
import csv
import numpy as np

# path = '/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/dataset/supplemental_train.csv'
#
# lst = []
# with open(path,'r') as f:
#     rdr =csv.reader(f)
#     lst = list(rdr)
#
#
#
# LST2 = []
# LST0 = []
# LST1 = []
# LST5 = []
# LST7 = []
# LST6 = []
# LST9 = []
# LST11 = []
# LST13 = []
# LST12 = []
# LST3 = []
# LST8 = []
# LST10 = []
# LST4 = []
#
# for idx,eachLine in enumerate(lst):
#     if idx ==0:
#         header = eachLine
#     else:
#         assetId = eachLine[1]
#         if int(assetId) == 2:
#             LST2.append(eachLine)
#         if int(assetId) == 0:
#             LST0.append(eachLine)
#         if int(assetId) == 1:
#             LST1.append(eachLine)
#         if int(assetId) == 5:
#             LST5.append(eachLine)
#         if int(assetId) == 7:
#             LST7.append(eachLine)
#         if int(assetId) == 6:
#             LST6.append(eachLine)
#         if int(assetId) == 9:
#             LST9.append(eachLine)
#         if int(assetId) == 11:
#             LST11.append(eachLine)
#         if int(assetId) == 13:
#             LST13.append(eachLine)
#         if int(assetId) == 12:
#             LST12.append(eachLine)
#         if int(assetId) == 3:
#             LST3.append(eachLine)
#         if int(assetId) == 8:
#             LST8.append(eachLine)
#         if int(assetId) == 10:
#             LST10.append(eachLine)
#         if int(assetId) == 4:
#             LST4.append(eachLine)
#
#         print(f'{idx}/{len(lst)} th append done')


def makeinterpolation(prevLst,postLst,ratio,plusNum):

    prevArr = np.array(prevLst,dtype=np.float32)[1:]
    postArr = np.array(postLst,dtype=np.float32)[1:]

    Result = prevArr * (1-ratio) +postArr * ratio

    ResultLst = Result.tolist()
    ResultLst.insert(0,int(prevLst[0])+plusNum)

    print(f'prev is : {prevLst[0]} and pose : {postLst[0]} so Result i'
          f's : {ResultLst[0]} post-prev is : {int(postLst[0])-int(prevLst[0])}'
          f'and new - prev is : {ResultLst[0]-int(prevLst[0])}'
          f'while plusNum is : {plusNum}')

    return ResultLst


coinLst = [
    'BitcoinCash.csv',
    'BinanceCoin.csv',
    'Bitcoin.csv',
    'EOS.csv',
    'ETHClassic.csv',
    'ETH.csv',
    'Litecoin.csv',
    'Monero.csv',
    'TRON.csv',
    'Stellar.csv',
    'Cardano.csv',
    'IOTA.csv',
    'Maker.csv',
    'Doge.csv'
]


def CHANGECSV2ADDINTERPOLATION(csvFile):

    saveDir = '/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/dataset/val/'

    newLst = []

    with open(saveDir+csvFile,'r') as f:
        rdr = csv.reader(f)
        lst = list(rdr)

        for idx,line in enumerate(lst):

            if idx < 1:
                newLst.append(line[:-2])
            elif idx >=1 and idx <= len(lst)-2:
                newLst.append(line[:-2])

                timeStamp = float( line[0] )
                nextStamp = float( lst[idx+1][0])

                if nextStamp - timeStamp > 60:
                    print(f'{idx} th line is missing next data because {nextStamp} and {timeStamp} diff is : {nextStamp-timeStamp}')
                    diff = int(int(nextStamp - timeStamp)/60) -1
                    #diff = int(nextStamp) - int(timeStamp)

                    #if diff% 60 != 0:
                    #    print(f'here is problem, {line[0]} with {lst[idx+1][0]} and {diff}')
                    for middle in range(diff):
                        middlePoint = makeinterpolation(prevLst=line[:-2],postLst=lst[idx+1][:-2],ratio=(middle+1)/(diff+1),plusNum=(middle+1)*60 )
                        newLst.append(middlePoint)
            else:
                newLst.append(line[:-2])

    with open(saveDir + 'new'+csvFile,'w',newline='') as f:
        wr = csv.writer(f)
        wr.writerows(newLst)

    print(f'{csvFile} complete')


for coin in coinLst:
    CHANGECSV2ADDINTERPOLATION(coin)





# saveDir = '/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/dataset/val/'
#
# with open(saveDir + 'BitcoinCash.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST2)
#
# with open(saveDir + 'BinanceCoin.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST0)
#
# with open(saveDir + 'Bitcoin.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST1)
#
# with open(saveDir + 'EOS.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST5)
#
# with open(saveDir + 'ETHClassic.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST7)
#
# with open(saveDir + 'ETH.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST6)
# with open(saveDir + 'Litecoin.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST9)
# with open(saveDir + 'Monero.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST11)
# with open(saveDir + 'TRON.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST13)
# with open(saveDir + 'Stellar.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST12)
# with open(saveDir + 'Cardano.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST3)
# with open(saveDir + 'IOTA.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST8)
# with open(saveDir + 'Maker.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST10)
# with open(saveDir + 'Doge.csv','w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(header)
#     wr.writerows(LST4)
#
#
# print('mission complete')

# import torch
# import torch.nn as nn
# import csv
# import numpy as np
#
# lst = [i for i in range(100)]
# idx = 17
#
# wdw = 5
# seqlen = wdw +7
#
# print(lst[idx-wdw+1:idx+seqlen-wdw+1],len(lst[idx-wdw:idx+seqlen-wdw]))
# Result= lst[idx-wdw+1:idx+seqlen-wdw+1]
#
# print(Result[wdw])
#
#
# mu = torch.tensor([2.0,3.0,4.0,5.0,47.412])
# sig = torch.tensor([1.0,1.0,1.0,1.0,0.000000000001])
#
# dist = torch.distributions.normal.Normal(mu,sig)
#
# samples = dist.sample()
#
#
#
# print(samples)
#
# def scalVlue(zPred):
#
#     if zPred[3] > zPred[1]:
#         zPred[3] = zPred[1]
#
#     elif zPred[3] < zPred[2]:
#         zPred[3]= zPred[2]
#
#     return zPred
#
# QQQ = torch.tensor([1,2,3,-1,5])
#
# qqq = scalVlue(QQQ)
#
# print(qqq)
#
#
# a = [1,2,3]
#
# b = a[0]
# print(b)
# a[0]= 7
#
# print(b)
import torch
import torch.nn as nn
import csv
import numpy as np
from datetime import datetime
import time


# baseDir = '/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/'
#
# with open(baseDir+'train.csv','r') as f:
#     rdr = csv.reader(f)
#     lst = list(rdr)
#
#
#
#
# with open(baseDir+'dataset/trainfile.csv','w',newline='') as ff:
#     wr =csv.writer(ff)
#     wr.writerows(lst[:20742578])
#
# with open(baseDir+'dataset/testfile.csv','w',newline='') as ff:
#     wr =csv.writer(ff)
#     wr.writerow(lst[0])
#     wr.writerows(lst[20742578:])
#
# print('spliting train and test data complete!')




# criterion = '2021-04-01 00:00:00'
# criterion =  datetime.strptime(criterion,'%Y-%m-%d %H:%M:%S')
# criterionTimestmp = datetime.timestamp(criterion)
#
# for row in range(len(lst)):
#     if row == 0:
#         continue
#
#     elif float(lst[row][0]) >= criterionTimestmp:
#         print(f'found at row : {row} and timestamp : {lst[row][0]}')
#         break
#     else:
#         print(f'{row} th line done...')




