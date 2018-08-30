import numpy as np
class MyLinearRegression():


    def __init__(self, n_iterations=10000, learning_rate=0.0005, regularization=None, gradient=True):
        '''初始化。是否正则化及L1L2的选择；选用梯度下降法还是正规方程法。梯度下降学习率以及迭代次数'''
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient = gradient
        if regularization == None:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = regularization

    def initialize_weights(self, n_features):
        '''初始化权重.初始化模型参数,参数矩阵w里的大小范围在（-limit，limit）之间，矩阵大小为（n_features，1）。w加入b的值相当于把偏置值加进去'''
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))              #二维数组，n行一列。
        b = 0
        self.w = np.insert(w, 0, b, axis=0)                                #对w,每列的0号加上b,只有一列，相当于加上w0

    def fit(self,X,y,):

        m_samples, n_features = X.shape                                      # !!!
        self.initialize_weights(n_features)
        X = np.insert(X, 0, 1, axis=1)                                      #二维数组，每行加一。相当于前面添加了一列。
        y = np.reshape(y, (m_samples, 1))                                    #二维数组，m 行一列。为了便于直接用向量计算
        self.training_errors = []
        if self.gradient == True:
            # 梯度下降
            for i in range(self.n_iterations):
                y_pred = X.dot(self.w)
                loss = np.mean(0.5 * (y_pred - y) ** 2) + self.regularization(self.w)  # 计算loss   !!!理解这里运算
                '''mean()函数功能：求取均值
                经常操作的参数为axis，以m * n矩阵举例：
                axis 不设置值，对 m*n 个数求均值，返回一个实数
                axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
                axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
                np.mean(X,axis=0或者1,keepdims=True)
                '''
                # print(loss)
                self.training_errors.append(loss)
                w_grad = X.T.dot(y_pred - y)/m_samples + self.regularization.grad(self.w)  # (y_pred - y).T.dot(X)，计算梯度
                self.w = self.w - self.learning_rate * w_grad  # 更新权值w
        else:
            # 正规方程
            X = np.matrix(X)
            y = np.matrix(y)
            X_T_X = X.T.dot(X)
            X_T_X_I_X_T = X_T_X.I.dot(X.T)
            X_T_X_I_X_T_X_T_y = X_T_X_I_X_T.dot(y)
            self.w = X_T_X_I_X_T_X_T_y

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred