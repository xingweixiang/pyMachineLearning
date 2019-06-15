#K近邻（KNN）
import math
import operator

def euclideanDistance(inst1, inst2, length):#求两个向量的欧式距离，参数向量1、向量2和向量的维度
    distance = 0
    for x in range(length):
        distance += pow((inst1[x] - inst2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):#在数据集中找到所需要的预测数据的K个最临近点并返回。参数是数据集、预测集以及K值
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):#返回对目标值的预测结果
    classVotes = {}
    for x in range(len(neighbors)):
        
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
            
        else:
            classVotes[response] = 1
            
    sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1),reverse=True)#Python3中：iteritems变为items
    return sortedVotes[0][0]

if __name__=='__main__':
    trainSet = [[1,1,1,'a'],[2,2,2,'a'],[1,1,3,'a'],[4,4,4,'b'],[0,0,0,'a'],[4,4.5,4,'b']]
    testInstance = [5,5,5]
    k = 3  # 预测是b
    #k=5#预测是a
    neighbors = getNeighbors(trainSet,testInstance,k)
    response = getResponse(neighbors)
    print(neighbors)
    print(repr(response))

