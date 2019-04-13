depencies:
    OS : Ubuntu14.04
    pytorch : 0.4.0
    cudnn : 7.1.2
    cuda : 9.0

python package:
    wordnet相关依赖(nltk.corpus) , 需要能够调用 nltk.corpus.wordnet.synset()
    tensorboardX 1.1
    scikit-image 0.13.0
    scikit-learn 0.19.2
    numpy 1.15.1

数据：
    data/ 为空,没有使用外部词向量,训练集只使用DatasetB
随机性：
    bagging所使用的数据的随机性由代码控制，但pytorch dataloader仍有一定随机性（很小）
模型：
    训练得到的模型将会保存在 data/save/ 中

复现结果: cd code && python main.py
