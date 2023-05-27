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

#3. ETF도 최적화 할 수 있음 (spx_mask를 "None"으로 주면 된다)
########################################################

ray.init(num_cpus=16)

@ray.remote
def run_optimizer(obj_function, rtn_df:pd.DataFrame, spx_mask, start_year:str, end_year:str, rebalancing:str, 
                  look_back_size:int, max_ratio:float, min_ratio:float, shrinkage_method="None", arg="None"):
    '''
    obj_function: 목적함수 [obj_sharpe, obj_variance]
    rtn_df: 수익률 데이터프레임
    spx_mask: S&P500 mask 데이터프레임 (ETF 최적화 하는 경우에는 "None" 주면 된다)
    look_back_size : int (과거 얼마 동안의 주가를 보고 포트폴리오를 구성할 지, "days")
    max_ratio : float (개별 주식당 포트폴리오 최대 비율 제한) 
    shrinkage_method : str -> ["None", "linear", "constant", "clipping", "kmeans]
    arg: float -> shrinkage method에 맞는 arg 를 주면 된다. (linear, clipping, kmeans만 해당!)
    '''
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    shrink = {"None": no_shrinkage,
              "linear":linear_shrinkage,
              "constant":constant_correlation_model,
              "clipping":eigenvalue_clipping,
              "kmeans":k_means_clustering
              }
    
    weight_df = pd.DataFrame(columns=rtn_df.columns) # weight를 담을 dataframe

    end_idx = pd.date_range(start_year,end_year, freq=f"{rebalancing}")

    for rebalancing_date in tqdm(end_idx):
            # start~rebalancing_date의 리턴을 보고 포폴 구성 (Look Back Window는 look_back_size로 조절)
            start = (rebalancing_date - pd.Timedelta(days=look_back_size))#.strftime("%Y-%m")
             
            if type(spx_mask) == pd.DataFrame: # S&P500을 최적화 하는 경우
                mask_sample = spx_mask.loc[:rebalancing_date].iloc[-1]
                universe = mask_sample.loc[~mask_sample.isna()].index # S&P500 구성종목을 가져옵니다
                rtn_lookback = rtn_df.loc[start:rebalancing_date, universe].dropna(axis=1) # v2개선 (이전 Lookback Window 동안 하나라도 없는게 있으면 Drop 해줍니다)
            else: # ETF 최적화 하는 경우
                rtn_lookback = rtn_df.loc[start:rebalancing_date] 
                
            universe = rtn_lookback.columns
            rtn_vol = np.diag(rtn_lookback.std())
            mean_return = rtn_lookback.mean()
            corr_matrix = rtn_lookback.corr() # corr_matrix를 추정하고 축소한 후에, optimizer에 넣기 전에 cov_matrix로 변환해야함
            
            if shrinkage_method == "kmeans":
                shrinked_corr_matrix = shrink[shrinkage_method](corr_matrix = corr_matrix, arg=[arg, rtn_lookback])
            else:
                shrinked_corr_matrix = shrink[shrinkage_method](corr_matrix = corr_matrix, arg=arg) # Corr_matrix를 축소
                
            cov_matrix = rtn_vol.dot(shrinked_corr_matrix).dot(rtn_vol) # corr matrix를 cov matrix로 변경
                        
            bounds = tuple((min_ratio, max_ratio) for _ in range(len(rtn_lookback.columns)))
            initial_weights = np.ones(len(rtn_lookback.columns)) / len(rtn_lookback.columns)
            
            # 최적화 수행
            if obj_function == obj_sharpe:
                args = (cov_matrix, mean_return,)
            elif obj_function == obj_variance:
                args = (cov_matrix, )
            
            result = minimize(obj_function, 
                              initial_weights, 
                              args=args,
                              method='SLSQP', 
                              constraints=constraints, 
                              bounds=bounds
                              )
            min_variance_weights = result.x
            weight_df.loc[rebalancing_date, universe] = min_variance_weights

    weight_df = weight_df.astype("float64") # v2추가: result의 리턴이 object였음
    
    return weight_df