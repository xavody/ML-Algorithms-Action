

def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确度"""
    return sum(y_predict == y_true) / len(y_true)
