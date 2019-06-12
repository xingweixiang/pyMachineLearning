import random

#分层采样
from example.code_3.random_sampling import RandomSampling
def StratiFiedSampling(dataMat1,dataMat2,dataMat3,number):
    sample = []
    num = int(number/3)
    sample.append(RandomSampling(dataMat1,num))
    sample.append(RandomSampling(dataMat2, num))
    sample.append(RandomSampling(dataMat3, num))
    return sample

if __name__ == '__main__':
    dataMat = list(range(10))
    print(StratiFiedSampling(dataMat,dataMat,dataMat,6))