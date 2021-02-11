import pandas as pd
import numpy as np
import sqlalchemy
import lightgbm as lgb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
class LatentFactor:
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
        new_users=0
        new_items=0
        self.losses=[]
        for i in tqdm(range(iters)):
            print('\n'+'='*15+'Iteration '+str(i)+'='*15+'\n')
            loss=0.0
            for user,items in user_item.items():
                #user cold start
                if user not in self.P.keys():
                    #print('New user entry: '+str(user)+'\n')
                    new_users+=1
                    self.P.setdefault(user,[])
                    self.P[user]=np.random.rand(self.F)/np.sqrt(self.F)
                samples=self.NegativeSampling(items,item_pool,neg_ratio)
                for item,y in samples.items():
                    #Item cold start
                    if item not in self.Q.keys():
                        #print('New item entry: '+str(item)+'\n')
                        new_items+=1
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
                print('cross entropy loss: '+str(loss))
            self.losses.append(loss)
            #self.eta*=0.9
        print('New user entries: '+str(new_users)+'\n')
        print('New item entries: '+str(new_items)+'\n')
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
    def get_embeddings(self):
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
        recall_df['label']=recall_df['label_prior'].fillna(0)
        del recall_df['label_prior']

        item_features=self.item_embedding_lookup(recall_df['item_id'].unique())
        user_features=self.user_embedding_lookup(recall_df['user_id'].unique())
        recall_df=recall_df.merge(item_features,how='left',on=['item_id'])
        recall_df=recall_df.merge(user_features,how='left',on=['user_id'])
        idx=np.random.permutation(len(recall_df))
        recall_df=recall_df.iloc[idx,:].reset_index(drop=True)
        if is_test:
            recall_df_train,recall_df_test=train_test_split(recall_df,test_size)
            return recall_df_train.reset_index(drop=True),recall_df_test.reset_index(drop=True)
        return recall_df
    def train_prediction_model(self,train,test=None,init_model=None,init_lgb_model=None):
        if init_model is not None:
            self.P=init_model.P
            self.Q=init_model.Q
        train=self.recall_dataset(train)
        cols=[col for col in train.columns if col not in ['user_id','item_id','label']]
        lgbtrain=lgb.Dataset(train[cols],label=train['label'])
        clf_params={'objective':'binary','metric':'auc','boosting':'goss','eta':0.05,'num_leaves':12,'max_depth':6,
        'min_sum_hessian_in_leaf':0.002,'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':0.2,'reg_lambda':0.08,'is_unbalance':True}
        if test is not None:
            test=self.recall_dataset(test)
            lgbtest=lgb.Dataset(test[cols],label=test['label'])
            lgb_clf=lgb.train(clf_params,lgbtrain,num_boost_round=500,early_stopping_rounds=50,valid_sets=[lgbtrain,lgbtest],
                            valid_names=['train','test'],verbose_eval=100,init_model=init_lgb_model)
        else:
            lgb_clf=lgb.train(clf_params,lgbtrain,num_boost_round=500,early_stopping_rounds=50,valid_sets=[lgbtrain],
                            valid_names=['train'],verbose_eval=100,init_model=init_lgb_model)
        self.prediction_model=lgb_clf
        return lgb_clf

    def feeling_lucky(self,user_ids,topK=5):
        user_ids=list(user_ids)
        tmps=[]
        for user_id in user_ids:
            try:
                items=self.Candidates(user_id).keys()
            except:
                print('New user entry: '+str(user_id)+'. Initializing latent factors...')
                self.P.setdefault(user_id,[])
                self.P[user_id]=np.random.rand(self.F)/np.sqrt(self.F)
                items=self.Candidates(user_id).keys()
            tmp=pd.DataFrame()
            tmp['item_id']=items
            tmp['user_id']=user_id
            tmps.append(tmp)

        recall_df=pd.concat(tmps,axis=0)
        recall_df=recall_df.loc[:,['user_id','item_id']]
        item_features=self.item_embedding_lookup(recall_df['item_id'].unique())
        user_features=self.user_embedding_lookup(recall_df['user_id'].unique())
        recall_df=recall_df.merge(item_features,how='left',on=['item_id'])
        recall_df=recall_df.merge(user_features,how='left',on=['user_id'])
        y_scores=self.prediction_model.predict(recall_df.drop(columns=['user_id','item_id']))
        recall_df['pred']=y_scores
        
        recall_df['rank']=recall_df.groupby(['user_id'])['pred'].rank(method='first',ascending=False)
        recmd_df=recall_df.sort_values(by=['user_id','rank']).loc[:,['user_id','item_id','rank']]
        recmd_df=recmd_df.groupby(['user_id'])['item_id','rank'].apply(lambda x : x[:topK]).droplevel(1).reset_index()
        return recmd_df

