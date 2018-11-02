import numpy as np
import math


def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的MSE"""
    return np.sum((y_predict - y_true) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE"""
    return math.sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict之间的MAE"""
    return np.sum(np.absolute(y_predict - y_true)) / len(y_true)


def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的 R Square"""
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
