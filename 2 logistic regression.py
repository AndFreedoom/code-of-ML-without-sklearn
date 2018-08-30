import numpy as np


class Logisticregression():


    def __init__(self, learn_rate = 0.001, max_iteration=10000):

        self.learn_rate = learn_rate
        self.max_iteration = max_iteration
        self._X_train = None
        self._y_train = None
        self._w = None

    def fit(self, X_train, y_train):

        m_samples, n_features = X_train.shape
        self._X_train = np.insert(X_train, 0, 1, axis=1)  # 二维数组，每行加一。相当于前面添加了一列。
        self._y_train = np.reshape(y_train, (m_samples, 1))  # 二维数组，m 行一列。为了便于直接用向量计算
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)
        iteration = 0
        while iteration < self.max_iteration:
            h_x = self._X_train.dot(self.w)
            y_pred = 1/(1+np.exp(- h_x))
            w_grad = self._X_train.T.dot(y_pred - self._y_train) # X.T(sigmod(Xw)-y)梯度,这里是批量梯度下降，使用随机梯度下降时，不能再用矩阵，需要循环。
            self.w = self.w - self.learn_rate * w_grad        # 迭代
            iteration = iteration + 1

        '''随机梯度下降时每次随机用循环，最后计算w 可以用向量，减少计算，关于迭代次数与样本数，这里的取舍，没有考虑。
        while iteration < self.max_iteration:
            for i in range(m_samples):
                randindex = np.random.randint(0,m_samples)
                y_pred = 1/(1+np.exp(self._X_train[randindex].dot(self.w)))    ##实质是w.Txi,不考虑严谨性，只是思路。
                error = y_pred - self._y_train[randindex]
                w_grad = error *self._X_train[randindex].T                     ## 注意此处，error是数 w_是向量
                self.w = self.w - self.learn_rate * w_grad
                iteration = iteration + 1
        '''
    def predict(self, X_test):

        X_test = np.insert(X_test, 0, 1, axis=1)
        h_x = X_test.dot(self.w)
        y_pripr_1 = (1/(1+np.exp(-h_x)))
        y_pripr_0 = 1 - y_pripr_1
        y_cal = y_pripr_1 - y_pripr_0             #这里实质是与0.5比较,实际中可以调节
        y_class = np.where(y_cal > 0, 1, 0)
        return y_class

    def score(self, X_test, y_test):

        j = 0
        y_test = np.reshape(y_test,(len(y_test),1))
        for i in range(y_test.shape[0]):

            if self.predict(X_test)[i,0] == y_test[i,0]:
                j += 1
        return ('accuracy: {:.10%}'.format(j / len(y_test)))