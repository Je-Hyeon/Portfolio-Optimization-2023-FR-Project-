import numpy as np

def obj_sharpe(weights, cov_matrix, mean_rtn):
    mean_return = mean_rtn * 250
    portfolio_return = np.dot(weights, mean_return)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot((cov_matrix), weights))) * np.sqrt(250)
    sharpe_ratio = portfolio_return  / portfolio_volatility
    return -sharpe_ratio

def obj_variance(weight, cov_matrix):
    '''목적 함수 정의(최소 분산)'''
    return np.dot(weight.T, np.dot((cov_matrix), weight)) * np.sqrt(250)

