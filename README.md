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
