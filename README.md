机器学习
=========
## 运行环境
     python 3.7.*
## 目录
* [机器学习](#机器学习)
	* [一、机器学习概述](#一机器学习概述)
		* [1、机器学习常用场景](#1机器学习常用场景)
		* [2、机器学习流程](#2机器学习流程)
		* [3、数据源结构](#3数据源结构)
		* [4、算法分类](#4算法分类)
		* [5、结果评估](#5结果评估)
	* [二、场景解析](#二场景解析)
		* [1、场景抽象](#1场景抽象)
		* [2、算法选择](#2算法选择)
	* [三、数据预处理](#三数据预处理)
		* [1、采样](#1采样)
		* [2、归一化](#2归一化)
		* [3、去除噪声](#3去除噪声)
		* [4、数据过滤](#4数据过滤)
	* [四、特征工程](#四特征工程)
		* [1、特征抽象](#1特征抽象)
		* [2、特征降维](#2特征降维)
	* [五、常规算法](#五常规算法)
		* [1、分类算法](#1分类算法 )
		* [2、聚类算法](#2聚类算法)
		* [3、回归算法](#3回归算法 )
		* [4、文本分析算法](#4文本分析算法)
		* [5、推荐算法](#5推荐算法 )
		* [6、关系图算法](#6关系图算法)
### 一、机器学习概述
### 1、机器学习常用场景
- 聚类场景：人群划分和产品种类划分。
- 分类场景：广告投放预测和网站用户点击预测。
- 回归场景：降雨量预测、商品购买量预测和股票成交额预测。
- 文本分析场景：新闻的标签提取、文本自动分类和文本关键信息抽取。
- 关系图算法场景：社交网络关系挖掘和金融风险控制。
- 模式识别：语音识别、图像识别和手写字识别。
### 2、机器学习流程
- 场景解析：理解业务，把业务场景进行一个抽象。把业务逻辑和算法进行匹配。<br>
如广告点击预测，其实是判断用户是否点击，可抽象为二分类问题，就可用监督学习以及二分类进行算法选择。
- 数据预处理：主要是进行数据的清洗工作，针对矩阵中的空值和乱码进行处理，目示是减少噪音数据对训练数据集的影响。
- 特征工程：特征工程的效果从某种意义上决定了最终模型的优劣。在算法固定的情况下，特征决定了好结果。
- 模型训练：在数据预处理、特征工程后进入算法训练模块，并且生成模型。读取模型的预测集数据进行计算，生成预测结果。
- 模型评估：机器学习算法计算的结果一般是一个模型，模型质量影响接下来的数据业务。对模型的成熟度评估，就是对整上机器学习流程的评估。
- 离线/在线服务：在实际的业务运用中，机器学习通常需要配合调度系统来使用。<br>
如每天用户将当日的增量数据流入数据库表里，通过调度系统启动机器学习的离线训练服务，生成最新的离线模型，<br>
然后通过在线预测服务（通常通过api，发送数据到服务器的算法模型进行计算，然后返回结果）进行实时的预测。
### 3、数据源结构
- 结构化数据：是指以矩阵结构存储在数据库中的数据，可以通过二维表结构来显示。
- 半结构化数据：是指按照一定的结构存储，但不是二维的数据库行存储形态。像XML扩展存储的数据。
- 非结构化数据：非结构化数据的数据挖掘是机器学习领域的热点话题，典型的就是图像、文本或者是语音文件。<br>
这些数据不能以矩阵的结构存储，通常转为二进制存储格式，然后通过算法来挖掘其中的信息。
### 4、算法分类
机器学习常分为4种：监督学习、无监督学习、半监督学习和增强学习
- 监督学习：是指进入算法的训练数据样本都有对应的期望值，通过过往的数据特征以及最终结果来进行训练。<br>
依赖于每个样本的打标，可以得到每个特征序列映射到的目标值是什么，常用于回归及分类场景。常见的监督学习是<br>
1）分类算法：K近邻、朴素贝叶斯、决策权、随机森林、GBDT和支持向量机<br>
2）回归算法：逻辑回归、线性回归
- 无监督学习：是指训练样本不依赖于打标数据的机器学习算法，常用来解决一些聚类的场景问题。常用的算法是<br>
1）聚类算法：K-Means、DBSCAN<br>
2）推荐算法：协同过滤
- 半监督学习：对样本的部分打标来进行训练数据的机器学习算法，很多是监督学习算法的变形，如标签传播算法。
- 强化学习：是比较复杂的机器学习种类，如隐马尔科夫就是一种强化学习的思想。<br>典型的案例如无人汽车驾驶和阿尔法狗下围棋。
### 5、结果评估
- 机器学习算法关于结果评估常用到的概念包括：精确率、召回率、F1值、ROC和AUC几种
### 二、场景解析
### 1、场景抽象
- 场景抽象：就是通过已有的数据，挖掘出可以应用的业务场景。目前机器学习主要用来解决的场景包括二分类、多分类、聚类和回归。<br>
如：商品推荐（二分类）、疾病预测（多分类）和人物关系挖掘（聚类）
### 2、算法选择
- 算法选择：用户需对自身的业务有一定的判断，到底是二分类、多分类、聚类还是回归等，把业务逻辑抽象成算法场景。
- 多算法尝试：多尝试几种算法，选择效果比较好的一种应用到实际业务中去。
- 多视角分析：当确定了特定业务场景下的算法后，除了考虑算法的效果外，还要从其他维度考察算法。如算法需要周期性调度使用，算法的调参和优化成本，还有运维成本等。
### 三、数据预处理
### 1、采样
采样就是按照某种规则从数据集中抽取样本数据，通常应用场景数据样本过大，抽取少部分样本来训练或验证，这样节约计算资源，也提升实验效果。
- 随机采样：是采样中最常用的一种，有放回采样和无放回采样。
- 系统采样：又称为等距采样，一般情况下是无放回式抽样。
- 分层采样：是先将数据分成若干个类别，再从每一层内随机抽取一定数量的观察样本，然后将抽取出来的样本组合起来。
### 2、归一化
- 归一化是一种简化计算的方式，将数据经过处理之后限定到一定的范围之内，一般将数据限定在[0,1]。数据归一化可以加快算法的收敛速度，也后续处理也会方便，是一种去量纲的行为。<br>
计算方法用数学公式表示：y=(x-MinValue)/(MaxValue-MinValue)。MaxValue,MinValue是矩阵的每一字段的最大最小值，x是字段中的值，y为归一化结果。
### 3、去除噪声
- 是指去除数据集中有干扰的数据（对场景描述不准确的数据）
### 4、数据过滤
- 对同一份数据，如果想做不同目的挖掘，需要做不同方式的处理，数据过滤是数据前期处理的重要一环。
### 四、特征工程
### 1、特征抽象
- 是指将源数据抽象成算法可以理解的数据。
### 2、特征降维
- 特征降维是挖掘出数据中的关键字段，减少输入矩阵的维度。机器学习中经常被用到，特别是图像识别或者是文本分析领域。
- 主成分分析（PCA）是最常用的一种线性降维方法。
- 线性判别式分析（LDA）是一种经典的特征降维方法。
- 主成分分析算法实现：
去除平均值<br>
计算协方差矩阵<br>
计算协方差矩阵的特征值和特征向量<br>
将特征值排序<br>
保留前N个最大的特征值对应的特征向量<br>
将数据转换到上面得到的N个特征向量构建的新空间中（实现了特征压缩）<br>
### 五、常规算法
### 1、分类算法
- K近邻（KNN）： 主要用来解决分类问题，是监督学习分类算法，在分类前需要有打标数据。<br>
算法在二分类、多分类、文本分类和图像分类等场景有许多的应用。
- 朴素贝叶斯（NBM）：以条件概率为基础的分类器，是一种监督学习算法，常被用于文本分类和垃圾邮件过滤等场景。<br>
因为输出结果是概率值，可以容易地从二分类扩展到多分类场景。
- 逻辑回归（LR）：是广义的线性回归分析监督学习算法。可以用在回归、二分类和多分类等问题上，但最常用的还是二分类。
- 支持向量机（SVM）：是监督学习分类算法，通过探求风险最小来提高学习机的泛化能力，实现经验风险和置信度范围的最小化。
- 随机森林（RF）：由多个决策树组成的分类器，是一种监督学习算法。通过对特征和训练样本的随机采样训练，生成多个决策树模型从而实现预测。<br>
如训练数据，预测用户是否有还贷能力。
### 2、聚类算法
- K-means：是根据距离聚类，在计算之前确定聚类簇心数量，是一种无监督算法。<br>
应用如在用户画像，通过人群的多维属性对人群的类别进行划分，在文本分析中对生成的词向量进行聚类可以挖掘出相似语义的词语。
- DBSCAN：是一种基于密度的聚类算法。
### 3、回归算法
- 线性回归：是一种监督算法。如果预测变量是离散的，称为分类。如果预测变量是连续的，则叫作回归。线性回归可用来预测如股票的涨跌、天气的变化。
### 4、文本分析算法
- 分词算法就是将句子按照每个词的意义进行分割。机器学习分词比较常见的方法是隐马尔科夫模型（HMM）和条件随机场算法。<br>
HMM作为统计模型，被广泛应用到文本分析，特别是分词领域。
- 中文分词常见的是jieba分词
- TF-IDF：是一种用于信息检索和数据挖掘的加权技术，文本打标的算法。如对文章进行分类，也可作为文本内容推荐的依据。
- 隐含狄利克雷分布（LDA）：是文本挖掘领域的主题模型，是一种无监督的机器学习算法。在实际业务中可通过标签进行相关性的推荐。
### 5、推荐算法
- 协同算法（CF）：是一种基于类别的推荐算法，最核心的理念是找出爱好相同的人或者属性相似的物。分为两种：<br>
基于人的推荐（UCF）和基于物品的推荐（ICF）（啤酒和尿布）
### 6、关系图算法
- 
