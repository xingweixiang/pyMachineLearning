import random

#系统采样
from example.code_3.random_sampling import RandomSampling
def SystematicSampling(dataMat,number):      
      
       length=len(dataMat)  
       k=int(length/number)
       sample=[]       
       i=0  
       if k>0 :         
         while len(sample)!=number:  
            sample.append(dataMat[0+i*k])  
            i+=1              
         return sample  
       else :  
         return RandomSampling(dataMat,number)

if __name__ == '__main__':
    dataMat = list(range(10))
    print(SystematicSampling(dataMat,4))