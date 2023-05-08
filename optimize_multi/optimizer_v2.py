import numpy as np 
import pandas as pd 
from scipy.optimize import minimize
import ray
from tqdm import tqdm
from optimize_multi.object_function import * 
from optimize_multi.shrinkage_method import *


#################### v2 개선사항 #######################
#1. S&P500에 리벨런싱날 상장되어 있다고 투자 대상으로 삼는게 아니라... 
# 리벨런싱날 이전 1년간 S&P500 주식 중 리턴 데이터가 전부 존재하는 애들만 투자 대상으로 삼아야겠다. 

#2. look_back_size 설정가능 / 마지막 하루 포함하는 걸로 수정

#3.  
######################################################

ray.init(num_cpus=16)

@ray.remote
def run_optimizer(obj_function, rtn_df:pd.DataFrame, spx_mask:pd.DataFrame,start_year:str, end_year:str, rebalancing:str, look_back_size:int, max_ratio:float, shrinkage_method="None"):
    '''
    obj_function: 목적함수
    rtn_df: 수익률 데이터프레임
    spx_mask: S&P500 mask 데이터프레임
    look_back_size : int (Default: 365 days)
    max_ratio : 개별 주식당 최대 얼마나 담을지 지정
    shrinkage_method : str -> [None, linear, constant, clipping]
    '''
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    shrink = {"None": no_shrinkage,
              "linear":linear_shrinkage,
              "constant":constant_correlation_model,
              "clipping":eigenvalue_clipping
              }
    
    weight_df = pd.DataFrame(columns=rtn_df.columns) # weight를 담을 dataframe

    end_idx = pd.date_range(start_year,end_year, freq=f"{rebalancing}")

    for rebalancing_date in tqdm(end_idx):
            # start~end의 주가를 보고 포폴 구성(Look Back Window는 1년이 된다)
            start = (rebalancing_date - pd.Timedelta(days=look_back_size)).strftime("%Y-%m")         
            
            mask_sample = spx_mask.loc[:rebalancing_date].iloc[-1]
            universe = mask_sample.loc[~mask_sample.isna()].index # S&P500 구성종목을 가져옵니다
            rtn_lookback = rtn_df.loc[start:rebalancing_date, universe].dropna(axis=1) # v2개선
            universe = rtn_lookback.columns
           
            rtn_vol = np.diag(rtn_lookback.std())
            corr_matrix = rtn_lookback.corr() # corr_matrix를 추정하고, optimizer에 넣기 전에 cov_matrix로 변환해야함
            mean_return = rtn_lookback.mean()
            
            shrinked_corr_matrix = shrink[shrinkage_method](corr_matrix = corr_matrix)
            cov_matrix = rtn_vol.dot(shrinked_corr_matrix).dot(rtn_vol) # corr matrix를 cov matrix로 변경
                        
            bounds = tuple((0,max_ratio) for _ in range(len(rtn_lookback.columns)))
            initial_weights = np.ones(len(rtn_lookback.columns)) / len(rtn_lookback.columns)
            
            # 최적화 수행
            result = minimize(obj_function, 
                              initial_weights, 
                              args=(cov_matrix, mean_return,),
                              method='SLSQP', 
                              constraints=constraints, 
                              bounds=bounds
                              )
            min_variance_weights = result.x
            weight_df.loc[rebalancing_date, universe] = min_variance_weights

    weight_df = weight_df.astype("float64") # v2추가 result의 리턴에 object 였다고??
    
    print("Jobs Done...")
    return weight_df