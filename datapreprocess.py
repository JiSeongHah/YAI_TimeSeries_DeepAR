import csv
import numpy as np
import torch

saveDir = '/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/train'

finalSaveDir = '/home/a286winteriscoming/Downloads/g-research-crypto-forecasting/'
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


for coin in coinLst:

    with open(finalSaveDir+'dataset/train/'+coin,'r') as f:
        rdr = csv.reader(f)
        prevLst = list(rdr)
    with open(finalSaveDir+'dataset/val/'+coin,'r') as f:
        rdr = csv.reader(f)
        postLst = list(rdr)

    prevTimeStamp = int(prevLst[-1][0])
    posttimeStamp = int(postLst[1][0])

    if posttimeStamp-prevTimeStamp == 60:
        print(f'yes for {coin} ')
    else:
        print(f'no for {coin}')


    # with open(saveDir+coin) as f:
    #     rdr = csv.reader(f)
    #     totalLst = list(rdr)
    #     header= totalLst[0]
    #
    #     totalLstLen = len(totalLst)
    #
    #     len4val = int(totalLstLen/16)
    #
    #     lst4trn = totalLst[:-len4val]
    #     lst4val = totalLst[-len4val:]

    # with open(finalSaveDir+'dataset/train/'+coin,'w',newline='') as ff:
    #     wr = csv.writer(ff)
    #     wr.writerows(lst4trn)
    # with open(finalSaveDir+'dataset/val/'+coin,'w',newline='') as ff:
    #     wr = csv.writer(ff)
    #     wr.writerow(header)
    #     wr.writerows(lst4val)

    print(f'dividing {coin} complete')
