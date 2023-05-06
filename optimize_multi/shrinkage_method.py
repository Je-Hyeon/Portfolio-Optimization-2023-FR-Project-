import warnings
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans

warnings.simplefilter(action='ignore', category=RuntimeWarning)
pd.options.mode.chained_assignment = None 


def no_shrinkage(corr_matrix):
    return corr_matrix
    
    
def linear_shrinkage(corr_matrix, alpha:float):
    '''
    alpha: float
    corr matrix를 리턴합니다
    '''
    return  alpha * np.identity(corr_matrix.shape[0]) + (1-alpha) * corr_matrix
    
    
def constant_correlation_model(corr_matrix): 
    n = len(corr_matrix)
    sum_r = np.sum(corr_matrix).sum() - np.sum(np.diag(corr_matrix)).sum()
    r = sum_r / (n*(n-1))
    return np.full(corr_matrix.shape, fill_value=r) - ((r-1) * np.identity(n))
    
    
def eigenvalue_clipping(corr_matrix, k:int):
    '''
    k:int
    '''
    eigen_value, eigen_vector = np.linalg.eigh(corr_matrix)
    eigen_value_bigger = np.where(eigen_value >= k, eigen_value, 0)
    eigen_value_smaller = eigen_value[eigen_value_bigger == 0]
    eigen_value_otherwise = np.nanmean(eigen_value_smaller)
    eigen_value_clipped = np.where(eigen_value >= k, eigen_value_bigger, eigen_value_otherwise)
    return eigen_vector @ np.diag(eigen_value_clipped) @ eigen_vector.T


def kmeans_clustering(corr_matrix, rtn_df, alpha:float): 
    '''alpha:float'''       
    t,n = len(rtn_df.index), len(rtn_df.columns)
    q = n/t
    lambda_plus = 1 + 2*(np.sqrt(q)) + q
    # Cluster의 개수를 구하기(RMT 이론에 의해)
    eigen_values = np.linalg.eigvalsh(corr_matrix)
    k = (eigen_values > lambda_plus).sum() #  k가 클러스터의 수
        
    # NaN 값 처리를 위해
    mean = rtn_df.mean(1)
    data = rtn_df.dropna(thresh=1).T.fillna(mean)
    kmean = KMeans(n_clusters=k, n_init=200,max_iter=1000)
    kmean.fit(data.values)
    label = kmean.labels_ #라벨의 순서는 cov_matrix의 (idx,col)순서와 동일하다
        
    within_corr_dict = {}
    between_corr_dict = {}
        
    # within corr 구하기
    for i in range(k): # i는 클러스터를 의미
        mask = (label == i)
        card_cluster = mask.sum()
        cluster_corr = rtn_df.loc[:,mask].corr().values

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

            all_corr = rtn_df.loc[:, mask_i+mask_j].corr().values
            all_corr_sum = (np.triu(all_corr) - np.diag(np.diag(all_corr))).sum()

            inner_corr_i = rtn_df.loc[:, mask_i].corr().values
            all_corr_sum_i = (np.triu(inner_corr_i) - np.diag(np.diag(inner_corr_i))).sum()        

            inner_corr_j = rtn_df.loc[:, mask_j].corr().values
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