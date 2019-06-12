import random
#随机采样dataMat数据集，number采样数
def RandomSampling(dataMat,number): #无放回采样
    try:  
         slice = random.sample(dataMat, number if len(dataMat) > number else len(dataMat))
         return slice  
    except:  
         print ('sample larger than population')
  
def RepetitionRandomSampling(dataMat,number):#放回采样
    sample=[]  
    for i in range(number):  
         sample.append(dataMat[random.randint(0,len(dataMat)-1)])  
    return sample

if __name__ == '__main__':
    dataMat = list(range(10))
    print(RandomSampling(dataMat,3))#无放回采样
    print(RepetitionRandomSampling(dataMat, 3))  # 无放回采样
