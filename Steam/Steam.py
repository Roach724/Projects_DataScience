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
#剔除特征
full.drop(['V5','V9','V11','V14','V17','V21','V27','V28','V25','V34'],axis=1,inplace=True)
full.shape
#标准化数据
scaler=MinMaxScaler()
y=train['target']
full.drop(['target','type'],axis=1,inplace=True)
full_minmax=pd.DataFrame(scaler.fit_transform(full),columns=full.columns)
x=full_minmax.iloc[:len(train)]
test=full_minmax.iloc[len(train):]


#交叉验证
"""
xgbreg=xgb.XGBRegressor(max_depth=6,learning_rate=0.03,n_estimators=500,min_child_weight=0.6,subsample=0.8,colsample_bytree=0.8,
                 reg_lambda=1.3,reg_alpha=0.6,nthread=4,n_jobs=4)
param={ 'bagging_freq':[2,3,5],'max_depth':[6,10,12]}
reg=GridSearchCV(xgbreg,param,cv=5,scoring='neg_mean_squared_error')
reg_cv=reg.fit(x,y)

print(reg_cv.best_params_)
print(reg_cv.best_score_)

#构建训练集和验证集
train_x,val_x,train_y,val_y=train_test_split(x,y,test_size=0.2,random_state=15)
xgb_full=xgb.DMatrix(data=x,label=y)
xgb_train=xgb.DMatrix(data=train_x,label=train_y)
xgb_val=xgb.DMatrix(data=val_x,label=val_y)
xgb_test=xgb.DMatrix(data=test)
cv_result={}
param={'eta':0.03,'eval_metric':'rmse','tree_method':'gpu_hist','subsample':0.8,'colsample_bytree':0.8,
      'bagging_freq':2,'max_depth':6,'reg_lambda':1.3,'reg_alpha':0.6,'min_child_weight':0.6}
cv_result=xgb.cv(param,xgb_full,num_boost_round=800,early_stopping_rounds=50,verbose_eval=100)
#训练模型
result={}
xgbmodel=xgb.train(param,xgb_train,num_boost_round=1000,early_stopping_rounds=50,verbose_eval=100,
                   evals=[(xgb_train,'train'),(xgb_val,'val')], evals_result=result)

##Ridge回归，搜索参数
ridge=Ridge()
ridge_param={'alpha':[1,1.3,2,2.2,2.5]}
ridge_gcv=GridSearchCV(ridge,ridge_param,cv=5,scoring='neg_mean_squared_error')
ridge_cv=ridge_gcv.fit(x,y)
print(ridge_cv.best_params_)
print(ridge_cv.best_score_)

##Lasso回归，搜索参数
lasso=Lasso()
lasso_param={'alpha':[0.0006,0.0007,0.0008]}
lasso_gcv=GridSearchCV(lasso,lasso_param,cv=5,scoring='neg_mean_squared_error')
lasso_cv=lasso_gcv.fit(x,y)
print(lasso_cv.best_params_)
print(lasso_cv.best_score_)
"""
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

#stacking models
ridge_model=Ridge(alpha=1.3)
lasso_model=Lasso(alpha=0.0008)
xgb_model=xgb.XGBRegressor(max_depth=6,learning_rate=0.03,n_estimators=500,min_child_weight=0.6,subsample=0.8,colsample_bytree=0.8,
                 reg_lambda=1.3,reg_alpha=0.6,nthread=4,n_jobs=4)
rf_model=RandomForestRegressor(max_leaf_nodes=100)
models=[ridge_model,lasso_model,xgb_model,rf_model]
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
stacking_pred=pd.DataFrame({'prediction':lr.predict(full_test)})
stacking_pred.to_csv('submission_stacking.txt',header=False,index=False)
