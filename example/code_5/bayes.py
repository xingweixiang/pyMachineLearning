# -*- coding: UTF-8 -*-
#朴素贝叶斯分类
import numpy as np
from functools import reduce

"""
函数1：创建实验样本
功能说明：首先要将文本切分成词条，这个函数就是干这个用的不过，现在已经切好了
返回值说明：postingList就是词条，classVec则是词条对应的分类标签
"""


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec  # 返回实验样本切分的词条和类别标签向量


"""
函数2: 制作词汇表
函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
参数说明：dataSet就是上面的postingList，也就是重复的词条样本集，而vocabSet则是无重复的词汇表
"""


def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


"""
函数3：词汇向量化
函数说明: 根据vocabList词汇表（也就是上面函数制作的词汇表），将inputSet（你输入的词汇）向量化，
          向量的每个元素为1或0，如果词汇表中有这个单词，就置1；没有，就置0
参数说明：最后返回的是文档向量（不是0就是1）
"""


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条
        if word in inputSet:  # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("词汇：%s 并没有在词汇表中" % word)  # 词汇表中没有这个单词，表示出现了问
    return returnVec  # 返回文档向量


"""
函数4：朴素贝叶斯分类器训练函数
函数说明: 利用朴树贝叶斯求出分类概率，也可以说是求出先验概率
参数说明：
输入参数trainMatrix：是所有样本数据矩阵，每行是一个样本，一列代表一个词条
输入参数trainCategory：是所有样本对应的分类标签，是一个向量，维数等于矩阵的行数
输出参数p0Vect：是一个向量，维数与上面相同，每个元素表示对应样本属于侮辱类的概率
输出参数p1Vect：是一个向量，和上面那个向量互补（因为是二分类问题），每个元素对应样本属于非侮辱类的概率
输出参数pAbusive：是一个概率值，表示这篇文档（所有样本的综合）属于侮辱类的概率
"""


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 训练集中样本数量
    numWords = len(trainMatrix[0])  # 每条样本中的词条数量
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)  # 词条初始化次数为1，避免出现0的情况，拉普拉斯平滑第一步
    p0Denom = 2.0;
    p1Denom = 2.0  # 分母初始化为2.0，拉普拉斯平滑第二步
    for i in range(numTrainDocs):  # 对每个标签进行判断
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  # 相除,然后取对数，防止下溢出
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive  # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率


"""
函数5：朴素贝叶斯分类器分类函数
函数说明: 利用几个函数得到的结果，直接对vec2Classify进行分类,说白了就是利用贝叶斯定理直接求了
但是这里并没有求分母，因为要比较的概率中分母都一样，可以不用求分母
参数说明：
输入参数vec2Classify——要分类的向量
输入参数后面三个——就是函数3得到的三个输出向量
输入参数：就是分类结果了（因为分类标签就只有2个，如果更改数据的话，这里要改一下）
"""


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # 对应元素相乘,log(A*B)=logA + logB
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0


"""
函数6：测试朴素贝叶斯分类器
函数说明: 这个就是一个测试函数了
"""


def testingNB():
    listOPosts, listClasses = loadDataSet()  # 创建实验样本
    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 将实验样本向量化

    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果


if __name__ == '__main__':
    testingNB()