import warnings
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from tqdm import tqdm
 
warnings.simplefilter(action='ignore', category=RuntimeWarning)
pd.options.mode.chained_assignment = None 

class PortfolioOptimzer:
    def __init__(self, price, spx_mask):
        '''
        Initialize the data (price: Price DataFrame 
                             spx_mask : S&P500 mask DataFrame)
        '''
        self.spx_mask = spx_mask
        self.rtn = price.pct_change(fill_method=None)

    def __obj_sharpe(self, weights, cov_matrix):
        portfolio_return = np.dot(weights, self.mean_return.values) * 250
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot((cov_matrix), weights))) * np.sqrt(250)
        sharpe_ratio = portfolio_return  / portfolio_volatility
        return -sharpe_ratio

    def __obj_variance(self, weight, cov_matrix):
        '''목적 함수 정의(최소 분산)'''
        return np.dot(weight.T, np.dot(cov_matrix, weight))
    
    def __no_shrinkage(self, corr_matrix, args):
        return corr_matrix
    
    def __linear_shrinkage(self, corr_matrix, alpha:int): # cov_matrix를 corr_matrix로 변환하는 과정이 필요함!
        '''Method of A'''
        return  alpha * np.identity(corr_matrix.shape[0]) + (1-alpha) * corr_matrix
    
    def __constant_correlation_model(self, corr_matrix, args): # cov_matrix를 corr_matrix로 변환하는 과정이 필요함!
        '''Method B'''
        n = len(corr_matrix)
        sum_r = np.sum(corr_matrix).sum() - np.sum(np.diag(corr_matrix)).sum()
        r = sum_r / (n*(n-1))
        return np.full(corr_matrix.shape, fill_value=r) - ((r-1) * np.identity(n))
    
    def __eigenvalue_clipping(self, corr_matrix, k:int): # cov_matrix를 corr_matrix로 변환
        '''Method C'''
        eigen_value, eigen_vector = np.linalg.eigh(corr_matrix)
        eigen_value_bigger = np.where(eigen_value >= k, eigen_value, 0)
        eigen_value_smaller = eigen_value[eigen_value_bigger == 0]
        eigen_value_otherwise = np.nanmean(eigen_value_smaller)
        # Result
        eigen_value_clipped = np.where(eigen_value >= k, eigen_value_bigger, eigen_value_otherwise)
        return eigen_vector @ np.diag(eigen_value_clipped) @ eigen_vector.T
    
    def __kmeans_clustering(self, corr_matrix, alpha:float):
        rtn_use = self.rtn_sample.copy()
        
        t,n = len(rtn_use.index), len(rtn_use.columns)
        q = n/t
        lambda_plus = 1 + 2*(np.sqrt(q)) + q
        # Cluster의 개수를 구하기(RMT 이론에 의해)
        eigen_values = np.linalg.eigvalsh(corr_matrix)
        k = (eigen_values > lambda_plus).sum() #  k가 클러스터의 수
        self.k = k
        
        # NaN 값 처리를 위해
        mean = rtn_use.mean(1)
        data = rtn_use.dropna(thresh=1).T.fillna(mean)

        kmean = KMeans(n_clusters=k, n_init=200,max_iter=1000)
        kmean.fit(data.values)
        label = kmean.labels_ #라벨의 순서는 cov_matrix의 (idx,col)순서와 동일하다
        
        within_corr_dict = {}
        between_corr_dict = {}
        
        # within corr 구하기
        for i in range(k): # i는 클러스터를 의미
            mask = (label == i)
            card_cluster = mask.sum()
            cluster_corr = rtn_use.loc[:,mask].corr().values

            with_in_cluster = (cluster_corr - np.diag(np.diag(cluster_corr))).sum() / (card_cluster * (card_cluster-1))
            if np.isnan(with_in_cluster) == True:
                with_in_cluster = 0
            within_corr_dict[i] = with_in_cluster 

        # Between corr 구하기
        for i in range(k): # i는 클러스터를 의미
            mask_i = (label == i)
            card_i = mask_i.sum()

            for j in range(k): # 클러스터 j를 뽑고
                if i == j:
                    continue
                mask_j = (label == j)
                card_j = mask_j.sum()

                all_corr = rtn_use.loc[:, mask_i+mask_j].corr().values
                all_corr_sum = (np.triu(all_corr) - np.diag(np.diag(all_corr))).sum()

                inner_corr_i = rtn_use.loc[:, mask_i].corr().values
                all_corr_sum_i = (np.triu(inner_corr_i) - np.diag(np.diag(inner_corr_i))).sum()        

                inner_corr_j = rtn_use.loc[:, mask_j].corr().values
                all_corr_sum_j = (np.triu(inner_corr_j) - np.diag(np.diag(inner_corr_j))).sum()   

                final_corr = all_corr_sum - all_corr_sum_i - all_corr_sum_j    

                between_cluster = final_corr / (2* card_i * card_j)
                between_corr_dict[(i,j)] = between_cluster
                
        # Within Correlation으로 S를 (i,j) 원소에 채우기... (i,j는 하나의 클러스터에 포함됨...)
        cor_matrix_cluster = pd.DataFrame(index=corr_matrix.index,
                                          columns=corr_matrix.columns)
        
        for i in range(k): # i는 각 클러스터를 의미함
            mask = (label == i)
            within_corr = within_corr_dict[i] # 이 within_corr을 각 회사의 pair 자리에 채워야함
            
            # select the rows and columns corresponding to the True values
            selected_rows = cor_matrix_cluster.loc[mask, :]
            selected_cols = cor_matrix_cluster.loc[:, mask]
            # fill in the selected values with a specific value 
            selected_rows.loc[:, selected_cols.columns] = within_corr
            selected_cols.loc[selected_rows.index, :] = within_corr
            # update the original correlation matrix with the modified values
            cor_matrix_cluster.loc[mask, :] = selected_rows
            cor_matrix_cluster.loc[:, mask] = selected_cols
        np.fill_diagonal(cor_matrix_cluster.values, 1) # 대각 행렬에 1을 채운다

        # Between corr으로 각 위치에 값을 채우기: 각 클러스터 p,q에서 pair에서 주식을 뽑고
        ## 주식 i, j자리에 행렬을 between corr으로 채운다
        for (p,q), between_corr in between_corr_dict.items(): # p,q는 클러스터를 의미함
            mask_p = (label == p)
            mask_q = (label == q)
            for i,bol_i in enumerate(mask_p): # i,j는 각각 클러스터에서 기업의 bol값을 의미함
                if bol_i:
                    for j, bol_j in enumerate(mask_q):
                        if bol_j:
                            cor_matrix_cluster.iloc[i,j] = between_corr
        
        reduced = alpha * cor_matrix_cluster + (1-alpha) * corr_matrix
        return reduced

    def run_optimizer(self, start_year:str, end_year:str, rebalancing:str, args=None, shrinkage_method="None"):
        '''
        포트폴리오 최적화를 수행합니다
        start_year, end_year : 투자기간(start_year + 1년부터 실제 투자 시작)
        rebalancing : 리벨런싱 주기 str -> [M,2M, Q, Y 등등,,,]
        shrinkage_method : str -> [None, linear, constant, clipping, clustering]
        
        args : dict -> {"alpha":int} / {"k": int}
        args 설명 -> linear: alpha=int  / clipping: k=int /clustering: alpha=float
        
        Return -> weight_df
        '''
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        shrink = {"None": self.__no_shrinkage,
                  "linear":self.__linear_shrinkage,
                  "constant":self.__constant_correlation_model,
                  "clipping":self.__eigenvalue_clipping,
                  "clustering":self.__kmeans_clustering}

        weight_df = pd.DataFrame(columns=self.rtn.columns) # weight를 담을 dataframe
        self.k_dict = {} #kmeans일경우 k를 담는다

        start_idx = pd.date_range(start_year,end_year, freq=f"{rebalancing}S")
        end_idx = pd.date_range(start_year,end_year, freq=f"{rebalancing}")

        for i in tqdm(range(len(end_idx))):
            
            # start~end의 주가를 보고 포폴 구성(Look Back Window는 1년이 된다)
            start = (start_idx[i] - pd.Timedelta(days=365)).strftime("%Y-%m") 
            end = (start_idx[i] - pd.Timedelta(days=1))             
            
            mask_sample = self.spx_mask.loc[:end].iloc[-1]
            universe = mask_sample.loc[~mask_sample.isna()].index # S&P500 구성종목을 가져옵니다
            rtn_lookback = self.rtn.loc[start:end, universe] 
           
            rtn_vol = np.diag(rtn_lookback.std())
            corr_matrix = rtn_lookback.corr() # corr_matrix를 추정하고, optimizer에 넣기 전에 cov_matrix로 변환해야함
            
            self.rtn_sample = rtn_lookback
            self.mean_return = rtn_lookback.mean() 
            
            if shrinkage_method == "None" or shrinkage_method == "constant":
                args = {"args":0}
            
            shrinked_corr_matrix = shrink[shrinkage_method](corr_matrix = corr_matrix, **args)
            cov_matrix = rtn_vol.dot(shrinked_corr_matrix).dot(rtn_vol) # corr matrix를 cov matrix로 변경
            
            if shrinkage_method == "clustering":
                self.k_dict[start_idx[i]] = self.k
                        
            bounds = tuple((0,1) for _ in range(len(rtn_lookback.columns)))
            initial_weights = np.ones(len(rtn_lookback.columns)) / len(rtn_lookback.columns)
            
            # 최적화 수행
            result = minimize(self.__obj_sharpe, 
                              initial_weights, 
                              args=(cov_matrix,),
                              method='SLSQP', 
                              constraints=constraints, 
                              bounds=bounds
                              )
            min_variance_weights = result.x
            weight_df.loc[start_idx[i], universe] = min_variance_weights
        
        print("Jobs Done...")
        print("You can check .rebalancing_date")
        return weight_df