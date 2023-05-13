import warnings
import numpy as np
import pandas as pd 

warnings.simplefilter(action='ignore', category=RuntimeWarning)
pd.options.mode.chained_assignment = None 


def no_shrinkage(corr_matrix, arg):
    return corr_matrix
    
    
def linear_shrinkage(corr_matrix, arg):
    alpha = arg
    '''
    alpha: float
    corr matrix를 리턴합니다
    '''
    return  alpha * np.identity(corr_matrix.shape[0]) + (1-alpha) * corr_matrix
    
    
def constant_correlation_model(corr_matrix, arg): 
    n = len(corr_matrix)
    sum_r = np.sum(corr_matrix).sum() - np.sum(np.diag(corr_matrix)).sum()
    r = sum_r / (n*(n-1))
    return np.full(corr_matrix.shape, fill_value=r) - ((r-1) * np.identity(n))
    
    
def eigenvalue_clipping(corr_matrix, arg):
    '''
    k:int
    '''
    k = arg
    eigen_value, eigen_vector = np.linalg.eigh(corr_matrix)
    eigen_value_bigger = np.where(eigen_value >= k, eigen_value, 0)
    eigen_value_smaller = eigen_value[eigen_value_bigger == 0]
    eigen_value_otherwise = np.nanmean(eigen_value_smaller)
    eigen_value_clipped = np.where(eigen_value >= k, eigen_value_bigger, eigen_value_otherwise)
    return eigen_vector @ np.diag(eigen_value_clipped) @ eigen_vector.T

