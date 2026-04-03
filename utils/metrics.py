import numpy as np

# Mean Squared Error 均方误差
def MSE(true, pred):
    return np.mean(np.square(true - pred))

# Root Mean Squared Error 均方根误差
def RMSE(true, pred):
    return np.sqrt(MSE(true, pred))

# Mean Absolute Error 平均绝对误差
def MAE(true, pred):
    return np.mean(np.abs(true - pred))

# Mean Absolute Percentage Error 平均百分比误差
def MAPE(true, pred):
    return np.mean(np.abs((true - pred) / true)) * 100

# Mean Absolute Scaled Error 平均绝对缩放误差
def MASE(true, pred):
    n = len(true)
    d = np.abs(true[1:] - true[:-1]).mean()
    errors = np.abs(true - pred)
    return errors.mean() / d
# Symmetric Mean Absolute Percentage Error 对称平均百分比误差
def SMAPE(true, pred):
    return np.mean(np.abs((true - pred) / true)) * 200 / (np.abs(true) + np.abs(pred)) * 100

# Pearson Correlation Coefficient 皮尔逊相关系数
def CORR(true, pred):
    return np.corrcoef(true, pred)[0, 1]

# Spearman Correlation Coefficient 斯皮尔曼相关系数
def SPEARMAN(true, pred):
    u = np.array([np.argsort(x) for x in true.T])
    v = np.array([np.argsort(x) for x in pred.T])
    return np.mean(np.sum(np.abs(u - v), 0))
# Normalized Discounted Cumulative Gain 归一化折扣累积增益
def ND(pred, true):
    return np.mean(np.abs(true - pred)) / np.mean(np.abs(true))