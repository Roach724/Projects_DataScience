import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
class Embedding:
    #K:embedding向量的维度，N:召回候选集大小
    def __init__(self,K,N=75):
        self.K=K
        self.N=N
    #初始化模型参数
    def initmodel(self,data):
        self.feature_cols=[col for col in data.columns]
        names=[]
        values=[]
        embedding_dims=0
        for col in data.columns:
            dim=data[col].nunique()
            embedding_dims+=dim
            names+=[col]*dim
            values+=list(data[col].unique())
        self.feature_list=np.array([names,values]).T
        self.embedding_matrix=np.random.randn(embedding_dims,self.K)/np.sqrt(self.K)
        self.W=np.random.randn(embedding_dims).reshape(-1,1)
        
        
    #负采样
    def NegativeSampling(self,data,item_pool,ratio,user_col='user_id',item_col='item_id'):
        tmps=[]
        for user_id in data[user_col].unique():
            tmp=pd.DataFrame()
            pos_item=data.loc[data[user_col]==user_id,item_col]
            neg_item=[]
            n=0
            for i in range(0,15*ratio):
                item=item_pool[np.random.randint(0,len(item_pool)-1)]
                if item in pos_item:
                    continue
                neg_item.append(item)
                n+=1
                if n>ratio*len(pos_item):
                    break
            tmp[item_col]=neg_item
            tmp[user_col]=user_id
            tmp['label']=0
            tmp=tmp.loc[:,[user_col,item_col]]
            tmps.append(tmp)
        neg_sample=pd.concat(tmps,axis=0)
        del tmps
        other_feature=[col for col in data.columns if col not in [user_col,item_col,'label']]
        for col in other_feature:
            neg_sample[col]=data[col]
        data=pd.concat([data,neg_sample],axis=0)
        data['label']=data['label'].fillna(0)
        idx=np.random.permutation(len(data))
        data=data.iloc[idx,:].reset_index(drop=True)
        return data

    #模型得分预测
    def predict(self,data):
        data=spars.csr_matrix(data)
        row_arr,col_arr=data.nonzero()
        for col_idx in len(row_arr):
            

        return self.sigmoid(logit)
    #拟合模型
    def fit(self,data,item_pool,ratio=5,num_iterations=5,learning_rate=0.01,verbose=True,init_model=None,label_col='label',debug=False):
        #负采样，标记数据
        data[label_col]=1
        data=self.NegativeSampling(data,item_pool,ratio)
        y=data[label_col]
        data.drop(columns=[label_col],inplace=True)

        #增量训练
        if init_model is None:
            self.initmodel(data)
        else:
            self.embbeding_matrix=init_model.embedding_matrix
            self.W=init_model.W
            if init_model.K != self.K:
                print('Dimension does not match, K in the initial model is'+str(init_model.K)+', set K equal to the same value.\n')
                return 0
        for col in data.select_dtypes(include='object').columns:
            one_hot_matrix=one_hot.fit_transform(train[col].values.reshape(-1,1))
            tmp.append(one_hot_matrix)
        data=sparse.csr_matrix(sparse.hstack(tmp))
        del tmp
        m,n=data.shape
        self.losses=[]
        for k in range(num_iterations):
            loss=0.0
            for i in range(m):
                y_hat=self.predict(data[i])




                
                loss+=(-(y*np.log(y_hat)+(1-y)*np.log(1-y_hat)))/len(X)     
            if verbose:
                print('='*15+'Iteration '+str(i)+'='*15+'\n')
                print('cross entropy loss: '+str(loss))
            self.losses.append(loss)
            #self.eta*=0.9
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))
    
    def Candidates(self,user):
        rank=dict()
        for item in self.Q.items():
            if item not in rank:
                rank.setdefault(item,0)
                rank[item]=self.predict(user,item)
        return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:self.N])
    def embedding_lookup(self,feature,values=None):
        if len([feature])>1:
            print('Only support for individual feature.\n')
            return
        feature_idx=np.isin(self.feature_list[:,0],feature)
        
        if values is None:
            lookup=self.embedding_matrix[feature_idx,:]
        else:
            value_idx=np.array([np.argmax(self.feature_list[feature_idx,1]==v) for v in values])
            lookup=self.embedding_matrix[feature_idx,:]
            lookup=lookup[value_idx,:]
        return lookup

    def linear_lookup(self,feature,values=None):
        if len([feature])>1:
            print('Only support for individual feature.\n')
            return
        feature_idx= np.isin(self.feature_list[:,0],feature)
        if values is None:
            lookup=self.W[feature_idx,:].T
        else:
            value_idx=np.array([np.argmax(self.feature_list[feature_idx,1]==v) for v in values])
            lookup=self.W[feature_idx,:]
            lookup=lookup[value_idx].T
        return lookup


        
