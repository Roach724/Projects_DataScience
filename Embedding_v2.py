import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
        self.W=(np.random.randn(embedding_dims)/np.sqrt(self.K)).reshape(-1,1)
        
        
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
        data=pd.concat([data,neg_sample],axis=0,sort=False)
        data['label']=data['label'].fillna(0)
        idx=np.random.permutation(len(data))
        data=data.iloc[idx,:].reset_index(drop=True)
        return data

    #模型得分预测
    def predict(self,data):
        data=np.array(data)
        if np.ndim(data)<2:
            data=data.reshape(1,-1)
        m,n=data.shape
        V=np.zeros((m,self.K,n))
        W=np.zeros((m,n))
        for j in range(n):
            values=data[:,j].astype(str)
            v,_,_=self.embedding_lookup(self.feature_cols[j],values=values)
            w,_,_=self.linear_lookup(self.feature_cols[j],values=values)
            V[:,:,j]=v
            W[:,j]=w
        logit=np.sum(W,axis=1)+0.5*(np.linalg.norm(np.sum(V,axis=2),axis=1)-np.sum(np.linalg.norm(V,axis=1),axis=1))
        return self.sigmoid(logit)
    #拟合模型
    def fit(self,data,item_pool,ratio=5,num_iterations=5,learning_rate=0.01,verbose=True,init_model=None,debug=False):
        #负采样，标记数据
        data['label']=1
        data=self.NegativeSampling(data,item_pool,ratio)
        y=np.array(data['label'])
        data.drop(columns=['label'],inplace=True)
        
        #增量训练
        if init_model is None:
            self.initmodel(data)
        else:
            self.embbeding_matrix=init_model.embedding_matrix
            self.W=init_model.W
            if init_model.K != self.K:
                print('Dimension does not match, K in the initial model is'+str(init_model.K)+', set K equal to the same value.\n')
                return 0
        data=np.array(data)
        m,n=data.shape
        self.losses=[]
        
        for t in range(num_iterations):
            loss=0.0
            for i in range(m):
                y_hat=self.predict(data[i])
                g0=y_hat-y[i]
                values=data[i]
                _,features_idx,values_idx=self.embedding_lookup(self.feature_cols,values)
                for j in range(n):
                    feature=self.feature_cols[j]
                    value=[data[i,j]]
                    _,feature_idx,value_idx=self.embedding_lookup(feature,value)
                    _,w_feature_idx,w_value_idx=self.linear_lookup(feature,value)
                    for k in range(self.K):
                        inter1=np.sum(self.embedding_matrix[features_idx][values_idx,k])
                        print(self.embedding_matrix[features_idx][values_idx,k].shape)
                        g1=inter1-self.embedding_matrix[feature_idx][value_idx,k][0]
                        
                        self.embedding_matrix[feature_idx][value_idx,k][0]-=learning_rate*(g0*g1)[0]

                    self.W[w_feature_idx][w_value_idx]-=learning_rate*g0
                
                y_hat=self.predict(data[i])
                loss+=(-(y[i]*np.log(y_hat)+(1-y[i])*np.log(1-y_hat)))/len(data)     
            if verbose:
                print('='*15+'Iteration '+str(t)+'='*15+'\n')
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
    def embedding_lookup(self,feature,values=None):
        if len([feature])>1:
            print('Only support for individual feature.\n')
            return
        feature_idx=np.isin(self.feature_list[:,0],feature)
        
        if values is None:
            lookup=self.embedding_matrix[feature_idx]
        else:
            value_idx=np.array([np.argmax(self.feature_list[feature_idx,1]==v) for v in values])
            lookup=self.embedding_matrix[feature_idx]
            lookup=lookup[value_idx]
        return lookup,feature_idx,value_idx

    def linear_lookup(self,feature,values=None):
        if len([feature])>1:
            print('Only support for individual feature.\n')
            return
        feature_idx= np.isin(self.feature_list[:,0],feature)
        if values is None:
            lookup=self.W[feature_idx].T
        else:
            value_idx=np.array([np.argmax(self.feature_list[feature_idx,1]==v) for v in values])
            lookup=self.W[feature_idx]
            lookup=lookup[value_idx].T
        return lookup,feature_idx,value_idx


        
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

