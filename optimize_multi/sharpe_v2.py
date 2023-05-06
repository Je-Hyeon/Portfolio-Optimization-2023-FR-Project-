import warnings

from tqdm import tqdm
    
    

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
                        
            bounds = tuple((0,0.05) for _ in range(len(rtn_lookback.columns)))
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