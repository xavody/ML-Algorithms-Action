import numpy as np

from metrics import r2_score


class SimpleLinearRegression1:
    """
    模型训练中使用 for 循环
    """
    
    def __init__(self):
        """初始化简单线性回归模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集训练模型"""
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2
        
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """根据给定的待测数据集，返回预测的结果向量"""
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """预测给定的单个待测数据"""
        return self.a_* x_single + self.b_

    def score(self, x_test, y_test):
        """根据测试数据集确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:
    """
    模型训练中使用向量化运算
    """    

    def __init__(self):
        """初始化简单线性回归模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集训练模型"""
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
        
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """根据给定的待测数据集，返回预测的结果向量"""
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """预测给定的单个待测数据"""
        return self.a_* x_single + self.b_

    def score(self, x_test, y_test):
        """根据测试数据集确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression2()"