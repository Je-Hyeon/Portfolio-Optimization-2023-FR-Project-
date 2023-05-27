import numpy as np 
import pandas as pd

# 그냥 Pandas 사용해서 쉽게 만들어 봤어요,,,
def split_time_series(df:pd.DataFrame, look_back_size:int, freq:str):
    '''
    look_back_size = (365 -> 1년 / 90 -> 1분기 / 30 -> 1달)
    freq = ["M","Q","Y"]
    Note) 맨 첫번쨰 Split에서 Sample 개수가 look_back_size보다 작은 경우가 있지만 무시합시다...
    '''
    idx_list = pd.date_range("2008","2023", freq=freq)
    
    for end_date in idx_list:
        start_date = end_date - pd.Timedelta(days=look_back_size)
        yield df.loc[start_date:end_date]
        
        
        
# Backtest editted by Hwang
def simulate_strategy(group_weight_df:pd.DataFrame, daily_rtn_df:pd.DataFrame, fee_rate:float):
  '''
  전략의 수익을 평가합니다(Long-Only Portfolio)
  '''
  pf_value = 1
  pf_dict = {}
  weight = group_weight_df.iloc[0] # 시작 weight를 지정해준다(첫 weight에서 투자 시작, 장마감 직전에 포트폴리오 구성)
  rebalancing_idx = group_weight_df.index
  start_idx = rebalancing_idx[0]

  for idx, row in daily_rtn_df.loc[start_idx:].iloc[1:].iterrows(): #Daily로 반복, 첫 weight 구성 다음 날부터 성과를 평가
      # 수익률 평가가 리밸런싱보다 선행해야함
      dollar_value = weight * pf_value
      dollar_value = dollar_value * (1+np.nan_to_num(row)) # update the dollar value
      pf_value = np.nansum(dollar_value) # update the pf value
      weight = dollar_value / pf_value   # update the weight 

      if idx in rebalancing_idx: # Rebalancing Date (장마감 직전에 리벨런싱 실시)
          weight = group_weight_df.loc[idx]
          target_dollar_value = np.nan_to_num(pf_value * weight)
          dollar_fee = np.nansum(np.abs(target_dollar_value - np.nan_to_num(dollar_value)) * fee_rate)
          pf_value = pf_value - dollar_fee # fee 차감
          
      pf_dict[idx] = pf_value
      
  # 결과를 pct로 정렬
  pf_result = pd.Series(pf_dict)
  idx = pf_result.index[0] - pd.Timedelta(days=1)
  pf_result[idx] = 1
  pf_result.sort_index(inplace=True)
  pf_result = pf_result.pct_change().fillna(0)

  #sharpe ratio 계산
  sharpe_ratio = (pf_result.mean()*252) / (pf_result.std()*np.sqrt(252))

  return pf_result, weight, sharpe_ratio