import numpy as np 
import pandas as pd 


def simulate_strategy(group_weight_df:pd.DataFrame, daily_rtn_df:pd.DataFrame, fee_rate:float):
        '''
        전략의 수익을 평가합니다(Long-Only Portfolio)
        '''
        pf_value = 1
        pf_dict = {}

        weight = group_weight_df.iloc[0] # 시작 weight를 지정해준다 (첫 weight에서 투자 시작, 장마감 직전에 포트폴리오 구성)
        dollar_value = weight * pf_value # Start Dollar Value를 지정
        
        rebalancing_idx = group_weight_df.index # 리벨런싱 할 날들
        start_idx = rebalancing_idx[0]          # 투자 시작일
        
        idx = daily_rtn_df.loc[start_idx:].index
        weight_df = pd.DataFrame(index=idx, columns=daily_rtn_df.columns) # Weight 변화를 기록할 빈 데이터프레임 생성
        weight_df.loc[start_idx] = weight # 시작 weight를 기록

        for idx, row in daily_rtn_df.loc[start_idx:].iloc[1:].iterrows(): # Daily로 반복 / 시작 weight 구성 다음 날부터 성과를 평가
            # 수익률 평가가 리밸런싱보다 선행해야함
            dollar_value = dollar_value * (1+np.nan_to_num(row)) # update the dollar value
            pf_value = np.nansum(dollar_value) # update the pf value
            
            weight = dollar_value / pf_value # update the weight

            if idx in rebalancing_idx: # Rebalancing Date (장마감 직전에 리벨런싱 실시)
                weight = group_weight_df.loc[idx] # Weight Rebalancing
                target_dollar_value = np.nan_to_num(pf_value * weight) * (1 - fee_rate)
                dollar_fee = np.nansum(np.abs(target_dollar_value - np.nan_to_num(dollar_value)) * fee_rate) # fee계산
                pf_value = pf_value - dollar_fee # fee 차감

                dollar_value = weight * pf_value  # dollar value를 Rebalancing 이후로 update
            
            weight_df.loc[idx] = weight # weight 변화를 기록
            pf_dict[idx] = pf_value
            
        # 결과를 pct로 정렬
        pf_result = pd.Series(pf_dict)
        idx = pf_result.index[0] - pd.Timedelta(days=1)
        pf_result[idx] = 1
        pf_result.sort_index(inplace=True)
        pf_result = pf_result.pct_change().fillna(0)
        
        return pf_result, weight_df