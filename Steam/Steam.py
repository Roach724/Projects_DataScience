import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,cross_val_predict,KFold,GridSearchCV,cross_validate
import scipy.stats as stat
%matplotlib inline
#读取数据
train=pd.read_csv('zhengqi_train.txt',delimiter='\t')
test=pd.read_csv('zhengqi_test.txt',delimiter='\t')
print(train.shape)
print(test.shape)

#新增一列用以区分训练集和测试集
train['type']='train'
test['type']='test'
#合并数据集
full=pd.concat([train,test],axis=0,sort=False)
full.info()
#对每一个特征，分别在训练集和测试集绘制密度图
for col in full.iloc[:,:38].columns:
    sns.distplot(full.loc[full['type']=='train',col])
    sns.distplot(full.loc[full['type']=='test',col])
    plt.show()
#绘制热力图，找出与目标变量相关性太弱的特征   
_,ax=plt.subplots(1,1,figsize=(20,12))
sns.heatmap(full[full['type']=='train'].corr(),ax=ax,annot=True)

#标准化数据
scaler=MinMaxScaler()
y=train['target']
full.drop(['target','type'],axis=1,inplace=True)
full_minmax=pd.DataFrame(scaler.fit_transform(full),columns=full.columns)
x=full_minmax.iloc[:len(train)].reset_index(drop=True)
test=full_minmax.iloc[len(train):].reset_index(drop=True)

#剔除方差过小的变量
less_var_cols=[]
for col in x.columns:
    if x[col].var()<0.01:
        less_var_cols.append(col)
#剔除特征
x.drop(['V4','V12','V31','V33','V5','V9','V11','V14','V17','V21','V27','V28','V25','V34'],axis=1,inplace=True)
test.drop(['V4','V12','V31','V33','V5','V9','V11','V14','V17','V21','V27','V28','V25','V34'],axis=1,inplace=True)


#构建训练集和测试集
train_x,val_x,train_y,val_y=train_test_split(x,y,test_size=0.3)
"""
xgb_full=xgb.DMatrix(data=x,label=y)
xgb_train=xgb.DMatrix(data=train_x,label=train_y)
xgb_val=xgb.DMatrix(data=val_x,label=val_y)
xgb_test=xgb.DMatrix(data=test)
"""
##训练模型
def train_model(model,x,y,param,nfolds=5):
    if len(param)>0:
        gsearch=GridSearchCV(model,param,cv=nfolds,scoring='neg_mean_squared_error')
        gsearch.fit(x,y)
        model=gsearch.best_estimator_
        print(gsearch.best_params_)
        print(gsearch.best_score_)
    return model
##stacking
def stacking(reg,X,y,test=None,nfolds=10):
    kf=KFold(n_splits=nfolds)
    secondary_train_set=pd.DataFrame(np.zeros((X.shape[0],1)))
    secondary_test_set=pd.DataFrame(np.zeros((test.shape[0],nfolds)))
    for i,(train_idx,val_idx) in enumerate(kf.split(X)):
        train_x,train_y=X.iloc[train_idx],y[train_idx]
        val_x,val_y=X.iloc[val_idx],y[val_idx]
        #用第i折的训练集训练初级学习器
        first_level_reg=reg.fit(train_x,train_y)
        secondary_train_set.iloc[val_idx]=first_level_reg.predict(val_x).reshape(-1,1)
        secondary_test_set.iloc[:,i]=first_level_reg.predict(test).reshape(-1,1)
    
    secondary_test_set=secondary_test_set.mean(axis=1)    
    return secondary_train_set,secondary_test_set
##Ridge
models=[]
ridge_model=Ridge()
ridge_param={'alpha':[0.001,0.01,0.1,1,1.01]}
ridge_model=train_model(ridge_model,x,y,ridge_param)
models.append(ridge_model)
##Lasso
lasso_model=Lasso()
lasso_param={'alpha':[0.0001,0.001,0.001,0.1]}
lasso_model=train_model(lasso_model,x,y,lasso_param)
models.append(lasso_model)
##XGBoost
xgb_model=xgb.XGBRegressor(learning_rate=0.03,n_estimators=1000,colsample_bytree=0.7,subsample=0.7,
                           min_child_weight=0.5,bagging_freq=5,nthread=4,n_jobs=4)
xgb_param={'max_depth':[3,4,5],'bagging_freq':[1,3,5]}
xgb_model=train_model(xgb_model,x,y,xgb_param)
models.append(xgb_model)
##Lightgbm
lgb_model=lgb.LGBMRegressor(learning_rate=0.03,n_estimators=1000,colsample_bytree=0.8,subsample=0.8,num_leaves=16,
                           min_child_weight=0.8,bagging_freq=1)
lgb_param={'max_depth':[3,4,5],'bagging_freq':[1,3,5]}
lgb_model=train_model(lgb_model,x,y,lgb_param)
models.append(lgb_model)

##Stacking models
train_sets=[]
test_sets=[]
nfolds=10
for model in models:
    train_set,test_set=stacking(model,x,y,test,nfolds)
    train_sets.append(train_set)
    test_sets.append(test_set)
full_train=pd.concat(train_sets,axis=1)
full_test=pd.concat(test_sets,axis=1)

#线性回归次级学习器
lr=LinearRegression()
lr.fit(full_train,y)
#print(mean_squared_error(y,lr.predict(full_train)))
##submit
stacking_pred=pd.DataFrame({'prediction':lr.predict(full_test)})
stacking_pred.to_csv('submission_stacking.txt',header=False,index=False)
