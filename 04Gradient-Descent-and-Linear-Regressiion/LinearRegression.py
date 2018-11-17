import numpy as np

from metrics import r2_score


class LinearRegression:
    
    def __init__(self):
        """初始化线性回归模型"""
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练数据集使用正规方程解训练线性回归模型"""
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集使用梯度下降法训练线性回归模型"""
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2 /len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=n_iters, epsilon=1e-8):
            theta = initial_theta
            i_ters = 0

            while i_ters < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta

                theta = theta - eta * gradient
                
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                
                i_ters += 1
                
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])

        self._theta = gradient_descent(X_b, y_train, initial_theta, eta)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        """根据训练数据集使用随机梯度下降法训练线性回归模型"""
        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters, t0, t1):
            
            def learning_rate(t):
                return t0 / (t + t1)
            
            theta = initial_theta
            m = len(X_b)

            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]

                for i in range(m):
                    grandient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * grandient
                
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])

        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """根据给定的待测数据集，返回预测的结果向量"""
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
