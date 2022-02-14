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
    with open(saveDir+coin) as f:
        rdr = csv.reader(f)
        totalLst = list(rdr)
        header= totalLst[0]

        totalLstLen = len(totalLst)

        len4val = int(totalLstLen/16)

        lst4trn = totalLst[:-len4val]
        lst4val = totalLst[-len4val:]

    with open(finalSaveDir+'dataset/train/'+coin,'w',newline='') as ff:
        wr = csv.writer(ff)
        wr.writerows(lst4trn)
    with open(finalSaveDir+'dataset/train/val'+coin,'w',newline='') as ff:
        wr = csv.writer(ff)
        wr.writerow(header)
        wr.writerows(lst4val)

    print(f'dividing {coin} complete')
