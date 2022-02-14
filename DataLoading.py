import csv
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from PIL import Image
from AreaCalc import calcArea
import os

# rootPath = '/home/a286winteriscoming/Downloads/EelPred/dataset/dataset/'
# trainFolderPath = rootPath +'train/'
# testFolderPath = rootPath + 'test'
# folderLst = os.listdir(trainFolderPath)
# print(folderLst)





# class MyEelDataset(torch.utils.data.Dataset):
#
#     def __init__(self,data_folder_dir,tLabelDir,TRAIN=True):
#
#         self.data_folder_dir = data_folder_dir
#
#         self.tLabelDir = tLabelDir
#
#         self.data_folder_lst = os.listdir(data_folder_dir)
#
#         self.TRAIN = TRAIN
#
#
#
#         self.labelDict = dict()
#
#         with open(self.tLabelDir, 'r') as f:
#             rdr = csv.reader(f)
#             for line in rdr:
#                 try:
#                     self.labelDict[str(line[0])] = float(line[1])
#                 except:
#                     self.labelDict[str(line[0])] = line[1]
#
#
#
#
#
#     def __len__(self):
#         return len(os.listdir(self.data_folder_dir))
#
#     def __getitem__(self, idx):
#
#         data_folder_name = self.data_folder_lst[idx]
#         full_data_dir = self.data_folder_dir+data_folder_name+'/'
#
#         json_lst = os.listdir(full_data_dir)
#         json_lst = [file for file in json_lst if file.endswith(".json")]
#
#         area_lst = []
#
#         area_avg = 0
#         area_std = 0
#
#         for each_json in json_lst:
#             with open(full_data_dir+each_json) as json_file:
#                 each_json_data = json.load(json_file)
#
#             for j in each_json_data['data']:
#                 lstX = list(map(int,j['x']))
#                 lstY = list(map(int,j['y']))
#
#                 area = calcArea(lstX,lstY) /10000
#
#                 area_lst.append(area)
#
#         area_avg = np.mean(area_lst)
#         area_std = np.std(area_lst)
#         area_quant = len(area_lst)/1000
#
#         input = torch.tensor([area_avg,area_std,area_quant]).float()
#
#
#
#         if self.TRAIN == True:
#
#             label = torch.tensor(float(self.labelDict[str(data_folder_name)]) /100 )
#
#             return data_folder_name, input, label
#
#         if self.TRAIN != True:
#
#             return data_folder_name, input

# class MyEelDataset(torch.utils.data.Dataset):
#
#     def __init__(self,data_folder_dir,tLabelDir,TRAIN=True):
#
#         self.data_folder_dir = data_folder_dir
#
#         self.tLabelDir = tLabelDir
#
#         self.data_folder_lst = os.listdir(data_folder_dir)
#
#         self.TRAIN = TRAIN
#
#
#
#         self.labelDict = dict()
#
#         with open(self.tLabelDir, 'r') as f:
#             rdr = csv.reader(f)
#             for line in rdr:
#                 try:
#                     self.labelDict[str(line[0])] = float(line[1])
#                 except:
#                     self.labelDict[str(line[0])] = line[1]
#
#
#
#
#
#     def __len__(self):
#         return len(os.listdir(self.data_folder_dir))
#
#     def __getitem__(self, idx):
#
#         data_folder_name = self.data_folder_lst[idx]
#         full_data_dir = self.data_folder_dir+data_folder_name+'/'
#
#         json_lst = os.listdir(full_data_dir)
#         json_lst = sorted([file for file in json_lst if file.endswith(".json")])
#         #print(json_lst)
#
#         area_lst = []
#
#         area_avg = 0
#         area_std = 0
#
#         instanceNum = -1
#
#         areaTmp = []
#
#         for each_json in json_lst:
#             with open(full_data_dir+each_json) as json_file:
#                 each_json_data = json.load(json_file)
#
#
#             if len(each_json_data['data']) != instanceNum and instanceNum == -1:
#                 #print('startstartstartstartstartstartstartstartstartstartstartstartstart')
#                 #print(f'istanceNum of now is : {instanceNum} but len is : ',len(each_json_data['data']))
#
#                 instanceNum = len(each_json_data['data'])
#                 #print(f'istanceNum of after change is : {instanceNum}')
#
#
#                 for j in each_json_data['data']:
#                     lstX = list(map(int, j['x']))
#                     lstY = list(map(int, j['y']))
#
#                     area = calcArea(lstX, lstY) / 10000
#
#                     areaTmp.append(area)
#
#                 #print(f'done 1 with NUm : {instanceNum}')
#
#             if len(each_json_data['data']) == instanceNum and instanceNum != -1:
#                 #print(f'istanceNum of now is : {instanceNum} but len is : ',len(each_json_data['data']))
#                 instanceNum = len(each_json_data['data'])
#                 #print(f'istanceNum of after change is : {instanceNum}')
#
#
#                 for j in each_json_data['data']:
#                     lstX = list(map(int, j['x']))
#                     lstY = list(map(int, j['y']))
#
#                     area = calcArea(lstX, lstY) / 10000
#
#                     areaTmp.append(area)
#
#                 #print(f'done 2 with NUm : {instanceNum}')
#
#             if len(each_json_data['data']) != instanceNum and instanceNum != -1:
#                 #print('--------------------------------------------------------------------------------')
#                 #print(f'istanceNum of now is : {instanceNum} but len is : ', len(each_json_data['data']))
#
#                 area_lst.append(np.mean(areaTmp))
#                 #print(areaTmp)
#                 areaTmp = []
#
#                 instanceNum = len(each_json_data['data'])
#                 #print(f'istanceNum of after change is : {instanceNum}')
#
#
#                 for j in each_json_data['data']:
#                     lstX = list(map(int, j['x']))
#                     lstY = list(map(int, j['y']))
#
#                     area = calcArea(lstX, lstY) / 10000
#
#                     areaTmp.append(area)
#
#                 #print(f'done 3 with NUm : {instanceNum}')
#
#         #print(f'areaTmp is : {areaTmp}')
#         area_lst.append(np.mean(areaTmp))
#
#         # area_lst = sorted(area_lst)
#         #
#         # plt.plot(range(len(area_lst)),area_lst)
#         # plt.show()
#
#         area_avg = np.mean(area_lst)
#         area_std = np.std(area_lst)
#         area_quant = len(area_lst)/1000
#
#         if np.isnan(area_avg) == True or np.isnan(area_std) ==True:
#             print(area_avg,area_std,data_folder_name,each_json,each_json_data)
#
#         input = torch.tensor([area_avg,area_std,area_quant]).float()
#
#         if self.TRAIN == True:
#
#             label = torch.tensor(float(self.labelDict[str(data_folder_name)]) /100 )
#
#             return data_folder_name, input, label
#
#         if self.TRAIN != True:
#
#             return data_folder_name, input











#
#
#
# class MyEelDataset(torch.utils.data.Dataset):
#
#     def __init__(self,data_folder_dir,tLabelDir,rangeNum,TRAIN=True):
#
#         self.data_folder_dir = data_folder_dir
#
#         self.tLabelDir = tLabelDir
#
#         self.data_folder_lst = os.listdir(data_folder_dir)
#
#         self.TRAIN = TRAIN
#
#         self.rangeNum = rangeNum
#
#         self.labelDict = dict()
#
#         with open(self.tLabelDir, 'r') as f:
#             rdr = csv.reader(f)
#             for line in rdr:
#                 try:
#                     self.labelDict[str(line[0])] = float(line[1])
#                 except:
#                     self.labelDict[str(line[0])] = line[1]
#
#
#     def __len__(self):
#         return len(os.listdir(self.data_folder_dir))
#
#     def __getitem__(self, idx):
#
#         data_folder_name = self.data_folder_lst[idx]
#         full_data_dir = self.data_folder_dir+data_folder_name+'/'
#
#         json_lst = os.listdir(full_data_dir)
#         json_lst = sorted([file for file in json_lst if file.endswith(".json")])
#         #print(json_lst)
#
#         area_lst = []
#
#         area_avg = 0
#         area_std = 0
#
#         instanceNum = -1
#
#         areaTmp = []
#
#         for each_json in json_lst:
#             with open(full_data_dir+each_json) as json_file:
#                 each_json_data = json.load(json_file)
#
#
#             if len(each_json_data['data']) != instanceNum and instanceNum == -1:
#                 #print('startstartstartstartstartstartstartstartstartstartstartstartstart')
#                 #print(f'istanceNum of now is : {instanceNum} but len is : ',len(each_json_data['data']))
#
#                 instanceNum = len(each_json_data['data'])
#                 #print(f'istanceNum of after change is : {instanceNum}')
#
#
#                 for j in each_json_data['data']:
#                     lstX = list(map(int, j['x']))
#                     lstY = list(map(int, j['y']))
#
#                     area = calcArea(lstX, lstY) / 10000
#
#                     areaTmp.append(area)
#
#                 #print(f'done 1 with NUm : {instanceNum}')
#
#             if len(each_json_data['data']) == instanceNum and instanceNum != -1:
#                 #print(f'istanceNum of now is : {instanceNum} but len is : ',len(each_json_data['data']))
#                 instanceNum = len(each_json_data['data'])
#                 #print(f'istanceNum of after change is : {instanceNum}')
#
#
#                 for j in each_json_data['data']:
#                     lstX = list(map(int, j['x']))
#                     lstY = list(map(int, j['y']))
#
#                     area = calcArea(lstX, lstY) / 10000
#
#                     areaTmp.append(area)
#
#                 #print(f'done 2 with NUm : {instanceNum}')
#
#             if len(each_json_data['data']) != instanceNum and instanceNum != -1:
#                 #print('--------------------------------------------------------------------------------')
#                 #print(f'istanceNum of now is : {instanceNum} but len is : ', len(each_json_data['data']))
#
#                 area_lst.append(np.mean(areaTmp))
#                 #print(areaTmp)
#                 areaTmp = []
#
#                 instanceNum = len(each_json_data['data'])
#                 #print(f'istanceNum of after change is : {instanceNum}')
#
#
#                 for j in each_json_data['data']:
#                     lstX = list(map(int, j['x']))
#                     lstY = list(map(int, j['y']))
#
#                     area = calcArea(lstX, lstY) / 10000
#
#                     areaTmp.append(area)
#
#                 #print(f'done 3 with NUm : {instanceNum}')
#
#         #print(f'areaTmp is : {areaTmp}')
#         area_quant = len(area_lst)
#
#         #area_lst = area_lst[int(len(area_lst)/2)-self.rangeNum:int(len(area_lst)/2)+self.rangeNum]
#
#         area_lst.append(np.mean(areaTmp))
#
#         # area_lst = sorted(area_lst)
#         #
#         # plt.plot(range(len(area_lst)),area_lst)
#         # plt.show()
#
#         area_avg = np.mean(area_lst)
#         area_std = np.std(area_lst)
#
#
#         if np.isnan(area_avg) == True or np.isnan(area_std) ==True:
#             print(area_avg,area_std,data_folder_name,each_json,each_json_data)
#
#         input = torch.tensor([area_avg,area_std]).float()
#
#         if self.TRAIN == True:
#
#             label = torch.tensor(float(self.labelDict[str(data_folder_name)]) /100  )
#
#             return data_folder_name, input, label
#
#         if self.TRAIN != True:
#
#             return data_folder_name, input






















#
# class MyEelDataset(torch.utils.data.Dataset):
#     def __init__(self,data_folder_dir,tLabelDir,rangeNum,TRAIN=True):
#
#         self.data_folder_dir = data_folder_dir
#
#         self.tLabelDir = tLabelDir
#
#         self.data_folder_lst = os.listdir(data_folder_dir)
#
#         self.TRAIN = TRAIN
#
#         self.avgfilterNum = 0
#
#         self.rangeNum = rangeNum
#
#         self.labelDict = dict()
#
#         with open(self.tLabelDir, 'r') as f:
#             rdr = csv.reader(f)
#             for line in rdr:
#                 try:
#                     self.labelDict[str(line[0])] = float(line[1])
#                 except:
#                     self.labelDict[str(line[0])] = line[1]
#
#
#     def __len__(self):
#         return len(os.listdir(self.data_folder_dir))
#
#     def __getitem__(self, idx):
#
#         data_folder_name = self.data_folder_lst[idx]
#         full_data_dir = self.data_folder_dir+data_folder_name+'/'
#
#         json_lst = os.listdir(full_data_dir)
#         jpg_lst = os.listdir(full_data_dir)
#
#         json_lst = sorted([file for file in json_lst if file.endswith(".json")])
#         jpg_lst = sorted([file for file in jpg_lst if file.endswith(".jpg")])
#
#
#         area_lst = []
#
#         area_avg = 0
#         area_std = 0
#
#         instanceNum = 0
#
#         areaTmp = []
#
#         for each_jpg in jpg_lst:
#
#             loadedImg = Image.open(full_data_dir+each_jpg)
#             Img2blwh = np.asarray(loadedImg.convert('L'))
#
#             maskedImg =  Img2blwh > self.rangeNum
#
#
#             if instanceNum == 0:
#                 tmpArr = maskedImg
#
#             else:
#                 tmpArr = maskedImg * tmpArr
#
#                 instanceNum +=1
#
#         for each_jpg,each_json in zip(jpg_lst,json_lst):
#
#             with open(full_data_dir+each_json) as json_file:
#                 loadedJson = json.load(json_file)
#
#             for j in loadedJson['data']:
#                 lstX = list(map(int, j['x']))
#                 lstY = list(map(int, j['y']))
#
#                 partArea = calcArea(lstX, lstY) / 10000
#
#                 areaTmp.append(partArea)
#
#             givenArea = np.mean(areaTmp)
#
#             loadedImg = Image.open(full_data_dir+each_jpg)
#             Img2blwh = np.asarray(loadedImg.convert('L'))
#             maskedImg =  Img2blwh > self.rangeNum
#
#             filtedImg = (maskedImg - tmpArr*1) >0
#
#             meanArea = np.mean(filtedImg) * givenArea
#
#             area_lst.append(meanArea)
#
#
#
#
#
#
#         #print(f'areaTmp is : {areaTmp}')
#         area_quant = len(area_lst)
#
#         #area_lst = area_lst[int(len(area_lst)/2)-self.rangeNum:int(len(area_lst)/2)+self.rangeNum]
#
#
#         # area_lst = sorted(area_lst)
#         #
#         # plt.plot(range(len(area_lst)),area_lst)
#         # plt.show()
#
#         area_avg = np.mean(area_lst)
#         area_std = np.std(area_lst)
#
#         input = torch.tensor([area_avg,area_std]).float()
#
#         if self.TRAIN == True:
#
#             label = torch.tensor(float(self.labelDict[str(data_folder_name)]) )
#
#             return data_folder_name, input, label
#
#         if self.TRAIN != True:
#
#             return data_folder_name, input


class MyEelDataset(torch.utils.data.Dataset):

    def __init__(self,data_folder_dir,tLabelDir,TEST=False,CROP=None):

        self.data_folder_dir = data_folder_dir

        self.tLabelDir = tLabelDir

        self.data_folder_lst = os.listdir(data_folder_dir)
        self.data_folder_lst = [fle for fle in self.data_folder_lst if fle.endswith('.jpg')]

        self.labelDict = dict()

        self.TEST = TEST
        self.CROP = CROP

        with open(self.tLabelDir, 'r') as f:
            rdr = csv.reader(f)
            for line in rdr:
                try:
                    self.labelDict[str(line[0])] = float(line[1])
                except:
                    self.labelDict[str(line[0])] = line[1]


    def __len__(self):
        return len(self.data_folder_lst)

    def __getitem__(self, idx):

        dataFileName = self.data_folder_lst[idx]
        full_data_dir = self.data_folder_dir+dataFileName

        img = Image.open(full_data_dir)
        imgArr = np.asarray(img)


        if self.CROP != None:
            w1 = self.CROP[0]
            w2 = self.CROP[1]
            h1 = self.CROP[2]
            h2 = self.CROP[3]

            imgArr = imgArr[w1:w2,h1:h2,:]



        input = torch.tensor(imgArr).float()
        input = input.permute(2,0,1)

        dataLabelName = str(dataFileName).split('_')[0]



        if self.TEST == False:

            label = torch.tensor([float(self.labelDict[dataLabelName]) /100 ])

            return dataLabelName, input, label
        if self.TEST == True:

            return dataLabelName, input


rootPath = '/home/a286winteriscoming/Downloads/EelPred/datasetVer1/dataset/'
trainFolderPath = rootPath +'train/'
valFolderPath = rootPath + 'val/'
testFolderPath = rootPath + 'test/'
labelPath = rootPath+'train.csv'

#
# # #
# dt = MyEelDataset(data_folder_dir=valFolderPath,tLabelDir=labelPath,TEST=False)
#
# for idx,i in enumerate(dt):
#     if idx > 3:
#         break
#     print(i[2])




