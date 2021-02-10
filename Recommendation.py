import joblib
import numpy as np
import pandas as pd
import sqlalchemy
import LatentFactor
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from LatentFactor import RecommendList
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,roc_curve,plot_roc_curve,precision_score

def load_data():
#data preparation
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql1= "select agent_id,oper_obj from chsell_oper_data where agent_id is not NULL and oper_time between '2021/1/1' and '2021/1/31' and oper_type='HOME_PRODUCT'"
        sql2= "select agent_id,oper_obj from chsell_oper_data where agent_id is not NULL and oper_time between '2021/2/1' and '2021/2/2'and oper_type='HOME_PRODUCT'"
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql="select oper_obj from chsell_oper_data where agent_id is not NULL and oper_time between '2021/1/1' and '2021/1/31' and oper_type='HOME_PRODUCT'"
        #sql="select id as oper_obj from chsell_product"
        engine=sqlalchemy.create_engine(engine_str)
        tmp=pd.read_sql(sql=sql,con=engine)
        item_pool=list(tmp['oper_obj'])
        engine=sqlalchemy.create_engine(engine_str)
        dftrain=pd.read_sql(sql=sql1,con=engine)
        dftest=pd.read_sql(sql=sql2,con=engine)
        return dftrain,dftest,item_pool
#training Latent factor model, which generates embedding vectors for users and items.
def lf_train(data,item_pool,F,neg_ratio,iters,eta,lamda,verbose=False):
        if data is None:
                return
        lf=LatentFactor.LatentFactor(F=F,item_pool=item_pool,neg_ratio=neg_ratio,iters=iters,eta=eta,lamda=lamda,verbose=verbose)
        lf.fit(data)
        joblib.dump(lf,'lf.txt')
        P,Q=lf.getPQ()
        P.to_csv('P.csv')
        Q.to_csv('Q.csv')
        return lf
#Lightgbm ranking model
def lgb_model(lf,dftrain,dftest=None):
        #训练/测试集负采样
        train=lf.getDataset(dftrain,15)
        lgb_cols=[col for col in train.columns if col not in ['user_id','item_id','label']]
        lgbtrain=lgb.Dataset(train[lgb_cols],label=train['label'])
        lgb_rank_X_train=train.sort_values(by=['user_id']).reset_index(drop=True)
        group_train=train.sort_values(by=['user_id']).groupby(['user_id'],as_index=False).count()['label'].values

        if dftest is not None:
                test=lf.getDataset(dftest,15)
                lgbtest=lgb.Dataset(test[lgb_cols],label=test['label'])
                lgb_rank_X_test=test.sort_values(by=['user_id']).reset_index(drop=True)
                group_test=test.sort_values(by=['user_id']).groupby(['user_id'],as_index=False).count()['label'].values

                lgb_ranker=lgb.LGBMRanker(boosting_type='goss',n_estimators=500,num_leaves=8,min_child_weight=0.004,reg_alpha=0.02,reg_lambda=0.01,max_depth=6,
                        subsample=0.7,colsample_bytree=0.7,learning_rate=0.08)
                lgb_ranker.fit(lgb_rank_X_train[lgb_cols],lgb_rank_X_train['label'],group=group_train,eval_set=[(lgb_rank_X_test[lgb_cols],lgb_rank_X_test['label'])],
                        eval_names=['test'],eval_metric='ndcg',eval_at=[1,2,3,4,5],eval_group=[group_test],early_stopping_rounds=100,verbose=50)
                joblib.dump(lgb_ranker,'lgb_ranker.ppml')
        #lightgbm binary classification
                eval_result={}
                clf_params={'objective':'binary','metric':'auc','boosting':'goss','eta':0.08,'num_leaves':12,'num_threads':6,'max_depth':6,
                'min_sum_hessian_in_leaf':0.002,'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':0.2,'reg_lambda':0.08,'unbalance':True}
                lgb_clf=lgb.train(clf_params,lgbtrain,num_boost_round=500,early_stopping_rounds=100,valid_sets=[lgbtrain,lgbtest],valid_names=['train','test'],
                verbose_eval=300,evals_result=eval_result)
                lgb_clf.save_model('lgb_classifier.ppml')
                #result_rank=RecommendList(lgb_rank_X_test,lgb_ranker,lgb_cols)
                #result_clf=RecommendList(lgb_rank_X_test,lgb_clf,lgb_cols)
        else:
                lgb_ranker=lgb.LGBMRanker(boosting_type='goss',n_estimators=500,num_leaves=8,min_child_weight=0.004,reg_alpha=0.02,reg_lambda=0.01,max_depth=6,
                        subsample=0.7,colsample_bytree=0.7,learning_rate=0.08)
                lgb_ranker.fit(lgb_rank_X_train[lgb_cols],lgb_rank_X_train['label'],group=group_train,eval_set=[(lgb_rank_X_train[lgb_cols],train['label'])],
                        eval_names=['test'],eval_metric='ndcg',eval_at=[1,2,3,4,5],eval_group=[group_train],early_stopping_rounds=100,verbose=50)
                joblib.dump(lgb_ranker,'lgb_ranker.ppml')

        #lightgbm binary classification
                eval_result={}
                clf_params={'objective':'binary','metric':'auc','boosting':'goss','eta':0.08,'num_leaves':12,'num_threads':6,'max_depth':6,
                'min_sum_hessian_in_leaf':0.002,'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':0.2,'reg_lambda':0.08,'unbalance':True}
                lgb_clf=lgb.train(clf_params,lgbtrain,num_boost_round=500,early_stopping_rounds=100,valid_sets=[lgbtrain],valid_names=['train'],
                verbose_eval=300,evals_result=eval_result)
                lgb_clf.save_model('lgb_classifier.ppml')
                #result_rank=RecommendList(lgb_rank_X_train,lgb_ranker,lgb_cols)
                #result_clf=RecommendList(lgb_rank_X_train,lgb_clf,lgb_cols)
        return lgb_ranker,lgb_clf

def Recommend(user_id):
        
        try:
                lgb_clf=lgb.Booster(model_file='lgb_clf.ppml')
                lgb_ranker=joblib.load('lgb_ranker.ppml')
                lf=joblib.load('lf.txt')
                X=make_dataset(user_id)
                feature_cols=X.drop(columns=['user_id','item_id']).columns
                for col in feature_cols:
                        X[col]=X[col].astype('float16')
                pred_clf=RecommendList(X,lgb_clf,feature_cols)
                pred_ranker=RecommendList(X,lgb_ranker,feature_cols)
        except:
                print('Train models......')
                dftrain,dftest,item_pool=load_data()
                lf=lf_train(data=dftrain,item_pool=item_pool,F=16,neg_ratio=3,iters=0,eta=0.13,lamda=0.03,verbose=True)
                lgb_ranker,lgb_clf=lgb_model(lf,dftrain,dftest)
                lgb_clf.save_model('lgb_classifier.ppml')
                joblib.dump(lgb_ranker,'lgb_ranker.ppml')
                X=make_dataset(user_id)
                feature_cols=X.drop(columns=['user_id','item_id','Unnamed: 0']).columns
                for col in feature_cols:
                        X[col]=X[col].astype('float16')
                pred_clf=RecommendList(X,lgb_clf,feature_cols)
                pred_ranker=RecommendList(X,lgb_ranker,feature_cols)
        return pred_clf,pred_ranker
def make_dataset(user_id):
        try:
                P=pd.read_csv('P.csv')
                Q=pd.read_csv('Q.csv')
        except:
                print('P,Q matrix do not exist')
                return
        user_feature=P[P['user_id']==user_id].reset_index(drop=True)
        tmp=pd.DataFrame(np.repeat(np.array(user_feature),len(Q),axis=0))
        tmp.columns=user_feature.columns
        X=pd.concat([tmp,Q],axis=1)
        return X

if __name__=='__main__':
        user_id='000af918c2e04069824f12d31e7ec9ad'
        
        pred_clf,pred_ranker=Recommend(user_id)
        print('clf:\n')
        print(pred_clf)
        print('\n')
        print('Ranker:\n')
        print(pred_ranker)