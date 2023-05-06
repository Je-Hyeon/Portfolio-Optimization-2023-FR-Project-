import numpy as np

def obj_sharpe(weights, cov_matrix, rtn_df):
    mean_return = rtn_df.mean().values * 250
    portfolio_return = np.dot(weights, mean_return)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot((cov_matrix*250), weights)))
    sharpe_ratio = portfolio_return  / portfolio_volatility
    return -sharpe_ratio

def __obj_variance(weight, cov_matrix):
    '''목적 함수 정의(최소 분산)'''
    return np.dot(weight.T, np.dot((cov_matrix*250), weight))

