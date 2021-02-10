import pandas as pd
import numpy as np
import sqlalchemy
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

def NegativeSampling(data,item_pool,ratio,user_col='user_id',item_col='item_id'):
    data['label']=1
    tmps=[]
    for user_id in data[user_col].unique():
        tmp=pd.DataFrame()
        pos_item=data.loc[data[user_col]==user_id,item_col]
        neg_item=[]
        n=0
        for i in range(0,50*ratio):
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
def process_data(data,item_pool,test_data=None,batch_size=256,sampling_ratio=10):
    ohe=OneHotEncoder(handle_unknown='ignore')
    df=data.copy()
    df=NegativeSampling(df,item_pool,sampling_ratio)
    if test_data is not None:
        df_test=test_data.copy()
        df_test=NegativeSampling(df_test,item_pool,sampling_ratio)
    matrix=[]
    matrix_test=[]
    for f in tqdm(df.select_dtypes(include='object').columns):

        ohe.fit(df[f].values.reshape(-1,1))
        encoding=ohe.transform(df[f].values.reshape(-1,1))
        matrix.append(encoding)
        if test_data is not None:
            encoding_test=ohe.transform(df_test[f].values.reshape(-1,1))
            matrix_test.append(encoding_test)
    datamatrix=sparse.hstack(matrix)
    target=df['label'].values
    user_field=df['user_id'].values
    item_field=df['item_id'].values
    dataset=tf.data.Dataset.from_tensor_slices(({'user_field':user_field,'item_field':item_field,'sparse_matrix':datamatrix.toarray()},target))
    dataset=dataset.shuffle(len(data)+1).batch(batch_size)
    if test_data is not None:
        datamatrix_test=sparse.hstack(matrix_test)
        user_field=df_test['user_id'].values
        item_field=df_test['item_id'].values
        try:
            target=df_test['label'].values
            dataset_test=tf.data.Dataset.from_tensor_slices(({'user_field':user_field,'item_field':item_field,'sparse_matrix':datamatrix_test.toarray()},target))
            dataset_test=dataset_test.shuffle(len(data)+1).batch(batch_size)
        except:
            print('No label exist in test set.\n')
            dataset_test=tf.data.Dataset.from_tensor_slices({'user_field':user_field,'item_field':item_field,'sparse_matrix':datamatrix_test.toarray()})
            dataset_test=dataset_test.shuffle(len(data)+1).batch(batch_size)
        return dataset,dataset_test
    return dataset 
    
    #return user_field,item_field,datamatrix.toarray(),target
def load_full_data(train_from,train_to,test_from=None,test_to=None,is_test=False):
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        
        sql1= "select agent_id as user_id,oper_obj as item_id from chsell_oper_data where agent_id is not NULL and oper_time between '" +train_from+ "'and'" +train_to+ "'and oper_type='HOME_PRODUCT'"
        #sql="select id as oper_obj from chsell_product"
        engine=sqlalchemy.create_engine(engine_str) 
        dftrain=pd.read_sql(sql=sql1,con=engine)
        if is_test:
                sql2= "select agent_id as user_id,oper_obj as item_id from chsell_oper_data where agent_id is not NULL and oper_time between '" +test_from+ "'and'" +test_to+ "'and oper_type='HOME_PRODUCT'"
                dftest=pd.read_sql(sql=sql2,con=engine)
                return dftrain,dftest
        else:
                return dftrain
def load_user_data(user_id,train_from,train_to):
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql1="select agent_id as user_id,oper_obj as item_id from chsell_oper_data where agent_id is not NULL and oper_time between '" +train_from+ "'and'" +train_to+ "' and oper_type='HOME_PRODUCT' and agent_id in ("+user_id+")"
        engine=sqlalchemy.create_engine(engine_str)
        dftrain=pd.read_sql(sql=sql1,con=engine)
        return dftrain
def load_item_pool(train_from,train_to):
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql="select oper_obj as item_id from chsell_oper_data where agent_id is not NULL and oper_time between '" +train_from+ "'and'" +train_to+ "' and oper_type='HOME_PRODUCT'"
        engine=sqlalchemy.create_engine(engine_str) 
        tmp=pd.read_sql(sql=sql,con=engine)
        item_pool=list(tmp['item_id'])
        return item_pool