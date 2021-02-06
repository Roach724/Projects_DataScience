import numpy as np
import pandas as pd
import Embedding_v2
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import time

if __name__=='__main__':
    
    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    item_pool=pd.read_csv('item_pool.csv')
    item_pool=list(item_pool['oper_obj'])
    emb=Embedding_v2.Embedding(K=64)
    emb.initmodel(train)
    #print(emb.feature_list)
    #print(emb.embedding_matrix.shape)
    
    start=time.time()
    #print(np.array(train).shape)
    print(emb.predict(train))
    #print(emb.embedding_matrix.shape)
    end=time.time()
    print('Time elapse: %s seconds'%(end-start))
    

    
    