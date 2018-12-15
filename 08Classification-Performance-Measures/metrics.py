import numpy as np


def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确度"""
    return sum(y_predict == y_true) / len(y_true)


def TN(y_true, y_predict):
    return np.sum((y_true == 0) & (y_predict == 0))


def FP(y_true, y_predict):
    return np.sum((y_true == 0) & (y_predict == 1))


def FN(y_true, y_predict):
    return np.sum((y_true == 1) & (y_predict == 0))


def TP(y_true, y_predict):
    return np.sum((y_true == 1) & (y_predict == 1))


def confusion_matrix(y_test, y_predict):
    return np.array([
        [TN(y_test, y_predict), FP(y_test, y_predict)],
        [FN(y_test, y_predict), TP(y_test, y_predict)]
    ])


def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0


def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0


def f1_score(y_true, y_predict):
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.0


def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0


def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.0
