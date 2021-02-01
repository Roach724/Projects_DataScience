import pandas as pd
import numpy as np
import sqlalchemy

class LatentFactor:
    def __init__(self,F,item_pool,N=5,neg_ratio=2,iters=5,eta=0.01,lamda=0.01,verbose=False):
        self.F=F
        self.N=N
        self.iters=iters
        self.eta=eta
        self.lamda=lamda
        self.item_pool=item_pool
        self.verbose=verbose
        self.neg_ratio=neg_ratio

    def initmodel(self,data):
        '''
        users=set(data.iloc[:,0])
        items=set(data.iloc[:,1])

        p=np.random.randn(len(users),self.F)
        q=np.random.randn(self.F,len(items))

        self.P=pd.DataFrame(p,columns=range(0,self.F),index=users)
        self.Q=pd.DataFrame(q,columns=items,index=range(0,self.F))
        '''
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
    
    def RandomSelectionNegativeSampling(self,items,ratio):
        ret=dict()
        for i in items:
            ret.setdefault(i,0)
            ret[i]=1
        n=0
        for i in range(0,len(items)*3):
            item=self.item_pool[np.random.randint(0,len(self.item_pool)-1)]
            if item in ret:
                continue
            ret[item]=0
            n+=1
            if n>ratio*len(items):
                break
        return ret
    def predict(self,user,item):
        rui=self.sigmoid(np.array(self.P[user]).T @ np.array(self.Q[item]))
        return rui

    def fit(self,data):
        self.initmodel(data)
        user_item=self.transform(data=data)
        self.losses=[]
        #v1=np.zeros(self.F)
        #v2=np.zeros(self.F)
        #beta=0.9
        for i in range(self.iters):
            loss=0.0
            for user,items in user_item.items():
                samples=self.RandomSelectionNegativeSampling(items,self.neg_ratio)
                for item,y in samples.items():
                    yhat=self.predict(user,item)
                    #eui=y-yhat
                    for f in range(self.F):
                        p=self.P[user][f]
                        q=self.Q[item][f]

                        #v1[f]+=beta*v1[f]+(1-beta)*
                        #v2[f]+=beta*v2[f]+(1-beta)*
                        #self.P[user][f]-=self.eta*(self.lamda*self.P[user][f]-eui*self.Q[item][f])
                        #self.Q[item][f]-=self.eta*(self.lamda*self.Q[item][f]-eui*self.P[user][f])
                        #Add regularization
                        g1=self.eta*(q*(yhat-y)+self.lamda*p)
                        g2=self.eta*(p*(yhat-y)+self.lamda*q)
                        self.P[user][f]-=g1
                        self.Q[item][f]-=g2
                        #print('Gradients:'+str(g1)+","+str(g2))
                    #loss+=((self.predict(user,item)-samples[item])**2+self.lamda* (np.linalg.norm(self.P[user])+np.linalg.norm(self.Q[item])))/len(samples)
                    yhat=self.predict(user,item)
                    #cross entropy
                    reg=0.5*self.lamda*(np.linalg.norm(self.P[user])+np.linalg.norm(self.Q[item]))
                    loss+=(-(y*np.log(yhat)+(1-y)*np.log(1-yhat))+reg)/(len(samples)*len(user_item))
                    #print(loss)
            if self.verbose:
                print('='*15+'Iteration '+str(i)+'='*15+'\n')
                print('cross entropy loss: '+str(loss))
            self.losses.append(loss)
            #self.eta*=0.9
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))
    def transform(self,data,user_col='agent_id',item_col='oper_obj'):
        if data is None:
            return
        return dict(data.groupby([user_col])[item_col].agg(set))

    def Recommend(self,user):
        rank=dict()
        puf=self.P[user]
        for item,qfi in self.Q.items():
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
                #print('user not in test set. Using training set instead')
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
                #print('user not in test set. Using training set instead')
                continue
            rank=self.Recommend(user=user)
            for item in rank:
                if item in tu:
                    hit+=1
            tot+=self.N
        return hit/tot
    #Coverage
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
    def getPQ(self):
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
    def getDataset(self,data,ratio):
        datadict=dict()
        user_item=self.transform(data=data)
        for user,items in user_item.items():
              samples=self.RandomSelectionNegativeSampling(items,ratio)
              datadict.setdefault(user,{})
              datadict[user]=samples
        users=[]
        items=[]
        labels=[]
        for user,sample in datadict.items():
            for item,label in sample.items():
                users.append(user)
                items.append(item)
                labels.append(label)
        df=pd.DataFrame({'user_id':users,'item_id':items,'label':labels})
        P,Q=self.getPQ()
        temp=pd.merge(df,P,how='inner',on='user_id')
        dataset=pd.merge(temp,Q,how='inner',on='item_id')
        idx=np.random.permutation(len(dataset))
        dataset=dataset.iloc[idx,:].reset_index(drop=True)
        return dataset

def RecommendList(df,model,feature_col,topk=5,is_predict=True,user_col='user_id',item_col='item_id',label='label',pred='pred'):
    try:
        num=model.best_iteration
    except:
        num=model._best_iteration
    df[pred]=model.predict(df[feature_col],num_iteration=num)
    result=df[[user_col,item_col,pred]].sort_values(by=[user_col,pred])
    result['rank']=result.groupby([user_col])[pred].rank(ascending=False,method='first')
    result=result[result['rank']<=topk].set_index([user_col,'rank']).unstack(-1).reset_index()
    if is_predict:
        result.columns=[int(col) if isinstance(col,int) else col for col in result.columns.droplevel(0)]
        result=result.rename(columns={'':'user_id',1:'item_1',2:'item_2',3:'item_3',4:'item_4',5:'item_5'}).iloc[:,:(topk+1)]
    else:
        result.columns=['user_id','item_1','item_2','item_3','item_4','item_5','label1','label2','label3','label4','label5']
        result=result.iloc[:,:(topk+1)]
    return result


