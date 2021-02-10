from lightgbm.engine import train
import pandas as pd
import numpy as np
import sqlalchemy
from sklearn.model_selection import train_test_split
class Embedding:
    def __init__(self,F,N=45):
        self.F=F
        self.N=N
    #初始化模型参数
    def initmodel(self,data):
        user_item=self.transform(data)
        P=dict()
        Q=dict()
        #2021/2/1 change initialization to be proportional to 1/sqrt(F)
        for user,items in user_item.items():
            P.setdefault(user,[])
            P[user]=np.random.rand(self.F)/np.sqrt(self.F)
            for item in items:
                Q.setdefault(item,[])
                Q[item]=np.random.rand(self.F)/np.sqrt(self.F)
        self.P=P
        self.Q=Q
    #负采样
    def NegativeSampling(self,items,item_pool,ratio):
        ret=dict()
        for i in items:
            ret.setdefault(i,0)
            ret[i]=1
        n=0
        for i in range(0,10*ratio):
            item=item_pool[np.random.randint(0,len(item_pool)-1)]
            if item in ret: ##若随机样本已在列表中则跳过
                continue
            ret[item]=0
            n+=1
            if n>ratio*len(items):
                break
        return ret
    #模型预测
    def predict(self,user,item):
        rui=self.sigmoid(np.array(self.P[user]).T @ np.array(self.Q[item]))
        return rui
    #拟合模型
    def fit(self,data,item_pool,neg_ratio=5,iters=5,eta=0.01,lamda=0.01,verbose=True,init_model=None):
        if init_model is None:
            self.initmodel(data)
        else:
            self.P=init_model.P
            self.Q=init_model.Q
            if init_model.F != self.F:
                print('Dimension does not match, F in the initial model is'+str(init_model.F)+', set F equal to the same value.\n')
                return 0
        
        user_item=self.transform(data=data)
        self.losses=[]
        for i in range(iters):
            loss=0.0
            for user,items in user_item.items():
                #user cold start
                if user not in self.P.keys():
                    print('New user entry: '+str(user)+'\n')
                    self.P.setdefault(user,[])
                    self.P[user]=np.random.rand(self.F)/np.sqrt(self.F)
                samples=self.NegativeSampling(items,item_pool,neg_ratio)
                for item,y in samples.items():
                    #Item cold start
                    if item not in self.Q.keys():
                        print('New item entry: '+str(item)+'\n')
                        self.Q.setdefault(item,[])
                        self.Q[item]=np.random.rand(self.F)/np.sqrt(self.F)
                    yhat=self.predict(user,item)
                    #eui=y-yhat
                    for f in range(self.F):
                        p=self.P[user][f]
                        q=self.Q[item][f]
                        g1=eta*(q*(yhat-y)+lamda*p)
                        g2=eta*(p*(yhat-y)+lamda*q)
                        self.P[user][f]-=g1
                        self.Q[item][f]-=g2
                    yhat=self.predict(user,item)
                    #cross entropy
                    reg=0.5*lamda*(np.linalg.norm(self.P[user])+np.linalg.norm(self.Q[item]))
                    loss+=(-(y*np.log(yhat)+(1-y)*np.log(1-yhat))+reg)/(len(samples)*len(user_item))
                    #print(loss)
            if verbose:
                print('='*15+'Iteration '+str(i)+'='*15+'\n')
                print('cross entropy loss: '+str(loss))
            self.losses.append(loss)
            #self.eta*=0.9
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))
    def transform(self,data,user_col='user_id',item_col='item_id'):
        if data is None:
            return
        return dict(data.groupby([user_col])[item_col].agg(set))

    def Candidates(self,user):
        rank=dict()
        for item,q in self.Q.items():
            if item not in rank:
                rank.setdefault(item,0)
                rank[item]=self.predict(user,item)
        return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:self.N])

    def Recall(self,train,test):
        train,test=self.transform(data=train),self.transform(data=test)
        hit=0
        tot=0
        for user in train.keys():
            try:
                tu=test[user]
            except:
                continue   
            rank=self.Recommend(user=user)
            for item in rank:
                if item in tu:
                    hit+=1
            tot+=len(tu)
        return hit/tot
    #Precision
    def Precision(self,train,test):
        train,test=self.transform(data=train),self.transform(data=test)
        hit=0
        tot=0
        for user in train.keys():
            try:
                tu=test[user]
            except:
                continue
            rank=self.Recommend(user=user)
            for item in rank:
                if item in tu:
                    hit+=1
            tot+=self.N
        return hit/tot
    '''
    def Coverage(self,train):
        train=self.transform(data=train)
        recommend_items=set()
        items=set()
        for user in train.keys():
            for item in train[user]:
                items.add(item)
            rank=self.Recommend(user)
            for item in rank:
                recommend_items.add(item)
        return len(recommend_items)/len(items)
    '''
    def getEmbeddings(self):
        ucol=dict()
        icol=dict()
        P=pd.DataFrame(self.P).T
        Q=pd.DataFrame(self.Q).T
        for col in P.columns:
            ucol.setdefault(col,'user_feature'+str(col))
            icol.setdefault(col,'item_feature'+str(col))
        P=P.rename(columns=ucol).reset_index().rename(columns={'index':'user_id'})
        Q=Q.rename(columns=icol).reset_index().rename(columns={'index':'item_id'})
        
        return P,Q
    def get_user_embedding(self):
        ucol=dict()
        P=pd.DataFrame(self.P).T
        for col in P.columns:
            ucol.setdefault(col,'user_feature'+str(col))
        P=P.rename(columns=ucol).reset_index().rename(columns={'index':'user_id'})
        return P
    def get_item_embedding(self):
        icol=dict()
        Q=pd.DataFrame(self.Q).T
        for col in Q.columns:
            icol.setdefault(col,'item_feature'+str(col))
        Q=Q.rename(columns=icol).reset_index().rename(columns={'index':'item_id'})
        return Q
    def item_embedding_lookup(self,item_ids):
        Q=self.get_item_embedding()
        item_feature=Q[Q['item_id'].isin(item_ids)].reset_index(drop=True)
        #tmp=pd.DataFrame(np.repeat(np.array(user_feature),len(Q),axis=0))
        #tmp.columns=list(user_feature.columns)
        #X=pd.concat([tmp,Q],axis=1)
        return item_feature
    def user_embedding_lookup(self,user_ids):
        P=self.get_user_embedding()
        user_feature=P.loc[P['user_id'].isin(user_ids)].reset_index(drop=True)
        return user_feature

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
        #print(recall_df)
        recall_df['label']=recall_df['label_prior'].fillna(0)
        #print(recall_df)
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


'''
        users=[]
        items=[]
        labels=[]
        for user,sample in datadict.items():
            for item,label in sample.items():
                users.append(user)
                items.append(item)
                labels.append(label)
        df=pd.DataFrame({'user_id':users,'item_id':items,'label':labels})
        P,Q=self.getEmbeddings()
        temp=pd.merge(df,P,how='inner',on='user_id')
        dataset=pd.merge(temp,Q,how='inner',on='item_id')
        idx=np.random.permutation(len(dataset))
        dataset=dataset.iloc[idx,:].reset_index(drop=True)
        return dataset
'''
def FM_Model(feature_dim):
    inputs=tf.keras.layers.Input(shape=(feature_dim,))
    linear=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                bias_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    cross=crosslayer(feature_dim)(inputs)
    add=tf.keras.layers.Add()([linear,cross])
    pred=tf.keras.layers.Activation('sigmoid')(add)
    model=tf.keras.Model(inputs=inputs,outputs=pred)
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(0.06),metrics=['AUC'])
    return model

