# 2018年天池"之江杯零样本识别比赛" #
比赛最终排名29
由于在这个比赛前完全不了解zero-shot这个领域，因此从一开始就决定先大量阅读相关论文，复现算法，再看有什么针对比赛数据集提高的方法。

## 初赛阶段 ##
1. 在完全没看过zeroshot论文的时候自己想的做法：用resnet18作为分类网络，因为发现数据中有30%左右是nonzero-shot的数据，因此专门训练了一个SVM来分辨zero和nonzero的数据。对nonzero的数据，直接用神经网络的分类层输出的结果作为结果，对zero的数据，用神经网络的分类结果的top5类别的属性的加权求和（加权方式是$alpha^k , k = rank  , alpha = 0.7$） , 用这种方式LB到 5%准确率左右。。
2. 根据Semantically Consistent Regularization for Zero-Shot Recognition这篇文章中介绍的方法，实现了其中的RULE和RIS，LB达到7%的样子。
4. 对模型加上人脸识别中的ArcFace loss ( additve angular margin ) ， LB最终到11%，进了复赛 , 其中使用ArcFace Loss明显提高了nonzero类别的分类准确率，从50%到60%左右(在比赛论坛中似乎没有看到比我更高的准确率)。

## 复赛阶段 ##
1. 发现复赛阶段没有nonzero的数据了，因此去掉了novelty_detector的部分。
2. 复现了Zero-shot Recognition via Semantic Embeddings and Knowledge Graphs，2018CVPR。该方法用GCN对类间关系进行建模，邻接矩阵A用wordnet相似度进行构建,图节点为类别的属性，GCN的输出为每个类别的向量表示$/phai_k$,用这个向量表示作为最后分类层的权值。LB达到11%
3. 对GCN的层数和A矩阵使用的相似度度量方法（wordnet API提供了不同的相似度度量）进行调参，对类别的“属性”concat上了300维的word2vec词向量，LB达到12%左右。
4. 因为感觉数据明显的分为几大类，因此找了一篇基于attention的细粒度分类算法来复现使用了2018年CVPR上 Fine-Grained Representation Learning and Recognition by Exploiting Hierarchical Semantic Embedding , 作为backbone ，然而发现效果很差,在不考虑zeroshot的问题的时候效果就明显弱于additive angular margin loss 的 resnet。
5. 尝试了叫QFSL(论文名字想不起来了)的2018 CVPR上的一篇zeroshot的算法，效果很差。
6. 在Arxiv上看到Towards Effective Deep Embedding for Zero-Shot Learning这篇文章，抛弃了GCN的结构，LB提高到了16% , 根据文中的提到的Hubness Problem ， 我仔细地设置了intermediate space dimension ， LB提高到20%。
7. 我对属性和词向量做了归一化到[-1,1],对网络的收敛有所帮助，LB最终提高到21%。

## 总结 ##
1. 像CVPR这种顶会上面的文章质量也是良莠不齐，特别是zeroshot这种相对比较小众的领域，感觉都在提fancy的方法，但其中的数据可信度不是很高,大部分文章都没提供代码,难以复现。对于文章中闭口不谈效果的文章基本上可以不用尝试了。
2. 在LB上总是看到排名前十的队伍score特别高，当时就很纳闷，后来了解到原来他们要么是在50维属性上做了自动特征生成等特征工程，要么是自己给类别标记了一些属性orz，而这些属性对zero-shot的效果影响特别大。听说我们学校排名第一的大佬自己手动标了十几个属性，而很多队伍都使用了根据规则生成属性的方式。由于之前没意识到属性的问题，而且也对机器学习中特征工程了解甚少，完全没有往这方面想,属实是一大败笔。不过最终在没做特征工程的情况下到29名，也算是比较高的名次了。

## 超参数 ##
- SGD , 学习率阶梯形衰减
- resize 到 (69,69) , 然后random crop (64,64)
- random horizontal flip
- ArcFace : s = 16 , m = 0.2
- MSE loss weight : 1.0


## Reference ##
1. Semantically Consistent Regularization for Zero-Shot Recognition
2. Zero-shot Recognition via Semantic Embeddings and Knowledge Graphs
3. Fine-Grained Representation Learning and Recognition by Exploiting Hierarchical Semantic Embedding
4. ArcFace: Additive Angular Margin Loss for Deep Face Recognition
5. Learning a Deep Embedding Model for Zero-Shot Learning
