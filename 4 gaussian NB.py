
import numpy as np

class NaiveBayes():
    '''高斯朴素贝叶斯分类器'''

    def __init__(self):

        self._X_train = None
        self._y_train = None
        self._classes = None
        self._priorlist = None
        self._meanmat = None
        self._varmat = None

    def fit(self, X_train, y_train):

        self._X_train = X_train
        self._y_train = y_train
        self._classes = np.unique(self._y_train)                           # 得到各个类别
        priorlist = []
        meanmat0 = np.array([[0, 0, 0, 0]])
        varmat0 = np.array([[0, 0, 0, 0]])
        for i, c in enumerate(self._classes):
            # 计算每个种类的平均值，方差，先验概率
            X_Index_c = self._X_train[np.where(self._y_train == c)]        # 属于某个类别的样本组成的“矩阵”
            priorlist.append(X_Index_c.shape[0] / self._X_train.shape[0])  # 计算类别的先验概率
            X_index_c_mean = np.mean(X_Index_c, axis=0, keepdims=True)     # 计算该类别下每个特征的均值，结果保持二维状态[[3 4 6 2 1]]
            X_index_c_var = np.var(X_Index_c, axis=0, keepdims=True)       # 方差
            meanmat0 = np.append(meanmat0, X_index_c_mean, axis=0)         # 各个类别下的特征均值矩阵罗成新的矩阵，每行代表一个类别。
            varmat0 = np.append(varmat0, X_index_c_var, axis=0)
        self._priorlist = priorlist
        self._meanmat = meanmat0[1:, :]                                    #除去开始多余的第一行
        self._varmat = varmat0[1:, :]

    def predict(self,X_test):

        eps = 1e-10                                                        # 防止分母为0
        classof_X_test = []                                                #用于存放测试集中各个实例的所属类别
        for x_sample in X_test:
            matx_sample = np.tile(x_sample,(len(self._classes),1))         #将每个实例沿列拉长，行数为样本的类别数
            mat_numerator = np.exp(-(matx_sample - self._meanmat) ** 2 / (2 * self._varmat + eps))
            mat_denominator = np.sqrt(2 * np.pi * self._varmat + eps)
            list_log = np.sum(np.log(mat_numerator/mat_denominator),axis=1)# 每个类别下的类条件概率相乘后取对数
            prior_class_x = list_log + np.log(self._priorlist)             # 加上类先验概率的对数
            prior_class_x_index = np.argmax(prior_class_x)                 # 取对数概率最大的索引
            classof_x = self._classes[prior_class_x_index]                 # 返回一个实例对应的类别
            classof_X_test.append(classof_x)
        return classof_X_test

    def score(self, X_test, y_test):

        j = 0
        for i in range(len(self.predict(X_test))):
            if self.predict(X_test)[i] == y_test[i]:
                j += 1
        return ('accuracy: {:.10%}'.format(j / len(y_test)))