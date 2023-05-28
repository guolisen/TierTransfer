from datasets import load_dataset
#from datasets import load_from_disk
import zlib
import time
import threading
import math
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from threading import Thread, Lock

plt.ion()

class Drive:
    def __init__(self):
        self.internaldata = []

    def writeData(self, data, tier, readHeat) :
        self.internaldata.append({'tier': tier, 
                            'data' : data, 
                            'readHeat': readHeat, 
                            'readTick': time.time()})
        return self.internaldata

    def updateData(self, lba, data, tier, readHeat) :
        self.internaldata[lba] = {'tier': tier, 
                                'data' : data, 
                                'readHeat': readHeat, 
                                'readTick': time.time()}
        return self.internaldata
    
    def updateData(self, lba, plb) :
        self.internaldata[lba]['tier'] = plb['tier']
        self.internaldata[lba]['data'] = plb['data']
        self.internaldata[lba]['readHeat'] = plb['readHeat']
        self.internaldata[lba]['readTick'] = time.time()
        return self.internaldata[lba]

    def readData(self, lba) :
        if lba >= len(self.internaldata) or lba < 0:
            return None, "error"
        #self.internaldata[lba]['readHeat'] += 1.0

        return self.internaldata[lba], ""
    
    def useSize(self):
        return len(self.internaldata)
    
    def getData(self):
        return self.internaldata

class Volume: 
    def __init__(self):
        self.drive = Drive()
        self.TransferThreshold = 50
        self.UpTierCount = 0
        self.DownTierCount = 0

    def write(self, dataset):
        for data in tqdm(dataset):
            dataByte = data['sentence'].encode()
            compressData = zlib.compress(dataByte, zlib.Z_BEST_COMPRESSION)
            self.drive.writeData(data = compressData, tier='HDD', readHeat=1.0)

    def read(self, lba):
        if lba >= self.drive.useSize() or lba < 0 :
            return
        currentTick = time.time()
        plb, _ = self.drive.readData(lba)
        newHeat = self.readHeatDecay(plb['readHeat'], plb['readTick'], currentTick)
        plb['readHeat'] = newHeat + 30
        #print ("!!!new heat: {} update heat: {}".format(newHeat, newHeat + 30))
        self.drive.updateData(lba, plb)

    def getHeat(self):
        data = self.drive.getData()
        heat = []
        for i in range(len(data)):
            heat.append(data[i]['readHeat'])
        return heat

    #T_now = T_last * Exp(-(tx) * coefficient)
    def newtonDecay(self, T_last, time_diff, coefficient):
        T_now = T_last * math.exp(-(time_diff) * coefficient)
        return T_now

    def readHeatDecay(self, readHeat, readTick, currentTick):
        return self.newtonDecay(readHeat, currentTick - readTick, 0.07)

    def readHeatIncDelta(T_last, delta, coefficient):
        T_now = T_last * math.log((delta) * coefficient)
        return T_now

    def refreshMonitorProcess(self):
        for lba in range(len(self.drive.internaldata)):
            if self.drive.internaldata[lba]['readHeat'] >= self.TransferThreshold \
                and self.drive.internaldata[lba]['tier'] == 'HDD':
                self.drive.internaldata[lba]['tier'] = 'SSD'
                self.UpTierCount += 1
                print("Refresh lba: {} UpTier: {}".format(lba, self.UpTierCount))
            if self.drive.internaldata[lba]['readHeat'] < self.TransferThreshold \
                and self.drive.internaldata[lba]['tier'] == 'SSD':
                self.drive.internaldata[lba]['tier'] = 'HDD'
                self.DownTierCount += 1
                print("Refresh lba: {} DownTier: {}".format(lba, self.DownTierCount))
        threading.Timer(1, self.refreshMonitorProcess).start()

    def refreshMonitorStart(self):
        self.refreshMonitorProcess()
        return
        


if __name__ == '__main__':
    vol = Volume()
    vol.refreshMonitorStart()
    dataset = load_dataset(path='glue', name='sst2', split='train')

    vol.write(dataset)

    # 设置均值和标准差
    l = len(dataset)
    mu1 = 30000  # 第一个正态分布的均值
    sigma1 = 5000  # 第一个正态分布的标准差

    mu2 = 50000  # 第二个正态分布的均值，左移100个单位
    sigma2 = 5000  # 第二个正态分布的标准差

    # 生成随机样本
    samples1 = np.random.normal(mu1, sigma1, 90000)  # 生成第一个正态分布的随机样本
    samples2 = np.random.normal(mu2, sigma2, 90000)  # 生成第二个正态分布的随机样本

    # 叠加两个正态分布的抽样
    combined_samples = []
    combined_samples.append(samples1)
    combined_samples.append(samples2)

    #plt.hist(samples1, bins=100, density=False, alpha=0.7)
    #plt.hist(samples2, bins=100, density=False, alpha=0.7)

    '''
    for i in tqdm(range(1000)):
        vol.read(0)
        print (vol.drive.internaldata[0])
        time.sleep(0.5)
    '''
    
    for sample in tqdm(samples1):
        vol.read(int(sample))
        time.sleep(0.001)
    for sample in tqdm(samples2):
        vol.read(int(sample))
        time.sleep(0.001)

    #heat = vol.getHeat()
    #x1=list(heat.keys())
    #x1=list(map(str,x1))
    #y1=list(heat.values())
    
    #plt.scatter(x1,y1, alpha=0.5)
    
    # 绘制直方图
    heat = vol.getHeat()
    #plt.hist(heat, bins=100, density=False, alpha=0.7)  # 使用30个柱形，设置密度为True，alpha为0.7
    
    dataframe = pd.DataFrame(heat)
    dataframe.hist()


    # 设置图形标题和坐标轴标签
    plt.title('Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # 显示图形
    plt.show()
    

    print (vol.UpTierCount)
    print (vol.DownTierCount)
    print (vol.volSize())