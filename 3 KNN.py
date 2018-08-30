import numpy as np
import operator


class KNNClassify():

    def __init__(self,k=5, p=2):

        self.k = k
        self.p = p
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict_y(self, X_test):

        m = self._X_train.shape[0]
        y_pre = []
        for intX in X_test:
            minus_mat = np.fabs(np.tile(intX, (m, 1)) - self._X_train)       # 将新的实例复制成m行1列，并进行相减
            sq_minus_mat = minus_mat ** self.p
            sq_distance = sq_minus_mat.sum(axis=1)
            diff_sq_distance = sq_distance ** float(1/self.p)

            sorted_distance_index = diff_sq_distance.argsort()               # 记录距离最近的k个点的索引
            class_count = {}
            vola = []
            for i in range(self.k):
                vola = self._y_train[sorted_distance_index[i]]
                class_count[vola] = class_count.get(vola, 0) + 1             # 统计k个点中所属各个类别的实例数目

            sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True) # 返回列表，元素为元组。每个类别以及对应的实例数目
            y_pre.append((sorted_class_count[0][0]))
        return (np.array(y_pre))

    def score(self, X_test, y_test):

        j = 0
        for i in range(len(self.predict_y(X_test))):
            if self.predict_y(X_test)[i] == y_test[i]:
                j += 1
        return ('accuracy: {:.10%}'.format(j / len(y_test)))