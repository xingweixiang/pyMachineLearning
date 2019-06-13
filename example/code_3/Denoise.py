#去除噪声，正态分布
def GetAverage(mat): #返回均值
      
    n=len(mat)  
    m= width(mat)   
    num = [0]*m  
    for j in range(0,m):   
           for i in mat:  
              num[j]=num[j]+i[j]             
           num[j]=num[j]/n     
    return num  
  
def width(lst):#获取矩阵的宽度
    i=0  
    for j in lst[0]:  
       i=i+1  
    return i  
  
def GetVar(average,mat):  #获取方差
    ListMat=[]  
    for i in mat:      
        ListMat.append(list(map(lambda x: x[0]-x[1], zip(average, i))))  
     
    n=len(ListMat)  
    m= width(ListMat)   
    num = [0]*m  
    for j in range(0,m):   
        for i in ListMat:  
                  num[j]=num[j]+(i[j]*i[j])         
        num[j]=num[j]/n     
    return num   
  
def DenoisMat(mat):#用于去噪
    average=GetAverage(mat)  
    variance=GetVar(average,mat)  
    section=list(map(lambda x: x[0]+x[1], zip(average, variance)))      
      
    n=len(mat)  
    m= width(mat)   
    num = [0]*m  
    denoisMat=[]      
    for i in mat:  
        for j in range(0,m):  
               if i[j]>section[j]:  
                     i[j]=section[j]  
        denoisMat.append(i)    
    return denoisMat
