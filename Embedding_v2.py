from lightgbm.engine import train
import pandas as pd
import numpy as np
import sqlalchemy
from sklearn.model_selection import train_test_split
class Embedding:
    #K:embedding向量的维度，N:召回候选集大小
    def __init__(self,K,N=75):
        self.K=K
        self.N=N
    #初始化模型参数
    def initmodel(self,X):
        X=X.melt().rename(columns={'variable':'feature'})
        embedding_dims=X.shape[0]
        #embedding vectors
        embedding_matrix=np.random.randn(embedding_dims,self.K)/np.sqrt(self.K)
        col_index=['V'+str(i) for i in range(self.K)]
        embedding_matrix=pd.DataFrame(embedding_matrix,columns=col_index)
        #linear coefficients
        W=np.random.randn(embedding_dims)
        W=pd.DataFrame(W,columns=['W'])
        
        
        self.embedding_matrix=pd.concat([X,embedding_matrix],axis=1)
        self.W=pd.concat([X,W],axis=1)
        #self.feature_index=feature_index
        '''
        embeddings=dict()
        w=dict()
        features=[]
        embedding_dims=0
        for col in X.columns:
            features.append(col)
            feature_values=set(X[col])
            embedding_dims+=len(feature_values)
            embeddings.setdefault(col,{})
            w.setdefault(col,{})
            for value in feature_values:
                embeddings[col].setdefault(value,[])
                embeddings[col][value]=np.random.randn(self.K)/np.sqrt(self.K)
                w[col].setdefault(value,0)
                w[col][value]=np.random.randn(1)/np.sqrt(self.K)
        self.features=features
        self.n_features=len(features)
        self.embeddings=embeddings
        self.embedding_dims=embedding_dims
        self.w=w
        '''
        
    #负采样
    def NegativeSampling(self,X,item_pool,ratio,user_col='user_id',item_col='item_id'):
        tmps=[]
        for user_id in X[user_col].unique():
            tmp=pd.DataFrame()
            pos_item=X.loc[X[user_col]==user_id,item_col]
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
        other_feature=[col for col in X.columns if col not in [user_col,item_col,'label']]
        for col in other_feature:
            neg_sample[col]=X[col]
        data=pd.concat([X,neg_sample],axis=0)
        data['label']=data['label'].fillna(0)
        idx=np.random.permutation(len(data))
        data=data.iloc[idx,:].reset_index(drop=True)
        
        return data

    #模型得分预测
    def predict(self,X):
        w=0
        v1=0
        v2=0
        for col in X.columns:
            v=np.array(self.embedding_matrix.loc[self.embedding_matrix['feature']==col&self.embedding_matrix['value']==X.loc[0,col],:])
            w+=self.W.loc[self.W['feature']==col&self.W['value']==X.loc[0,col],:]
            v1+=v
            v2+=np.linalg.norm(v)
        v1=np.linalg.norm(v1)
        logit=w+0.5*(v1-v2)
        p=self.sigmoid(logit)
        return p
    #拟合模型
    def fit(self,X,item_pool,ratio=5,num_iterations=5,learning_rate=0.01,verbose=True,init_model=None,label_col='label',debug=False):
        #负采样，标记数据
        X[label_col]=1
        X=self.NegativeSampling(X,item_pool,ratio)
        ys=X[label_col]
        X.drop(columns=[label_col],inplace=True)
        #增量训练
        if init_model is None:
            self.initmodel(X)
        else:
            self.embbedings=init_model.embeddings
            self.w=init_model.w
            if init_model.K != self.K:
                print('Dimension does not match, K in the initial model is'+str(init_model.K)+', set K equal to the same value.\n')
                return 0
        m,n=X.shape
        self.losses=[]
        for i in range(num_iterations):
            loss=0.0
            for j in range(m):
                y_hat=self.predict(X.iloc[m,:])



                
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
        for item,q in self.Q.items():
            if item not in rank:
                rank.setdefault(item,0)
                rank[item]=self.predict(user,item)
        return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:self.N])

    def recall_dataset(self,data,is_test=False,test_size=0.2):
        tmps=[]
        data['label_prior']=1
        for user in data['user_id'].unique():
            tmp=pd.DataFrame()
            tmp['item_id']=self.Candidates(user).keys()
            tmp['user_id']=user
            tmps.append(tmp)
        recall_df=pd.concat(tmps,axis=0)
        recall_df=recall_df.merge(data,how='left',on=['user_id','item_id'])
        recall_df['label']=recall_df['label_prior'].fillna(0)
        del recall_df['label_prior']

        item_features=self.item_embedding_lookup(recall_df['item_id'].unique())
        user_features=self.user_embedding_lookup(recall_df['user_id'].unique())
        recall_df=recall_df.merge(item_features,how='left',on=['item_id'])
        recall_df=recall_df.merge(user_features,how='left',on=['user_id'])
        feature_cols=[col for col in recall_df.columns if col not in ['user_id','item_id','label']]
        if is_test:
            recall_df_train,recall_df_test=train_test_split(recall_df,test_size)
            return recall_df_train.reset_index(drop=True),recall_df_test.reset_index(drop=True)
        return recall_df

