import pandas as pd
import numpy as np
from sklearn.utils.validation import _num_samples
import sqlalchemy
import tensorflow as tf
import warnings
import time
import datetime
warnings.filterwarnings('ignore')

def load_full_log(train_from,train_to,test_from=None,test_to=None,is_test=False):
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql1=" select t1.agent_id as user_id,t1.oper_time as time_stamp,cast(ifnull(qb.age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender,t1.oper_obj as item_id,IFNULL(t2.catalog,0) as item_catalog \
                from chsell_oper_data as t1 left JOIN chsell_product as t2 on t1.oper_obj= t2.id left JOIN chsell_agent as t3 on t1.agent_id = t3.id left JOIN bigdata.chsell_quick_bi as qb on t1.agent_id=qb.channel_id \
                where agent_id is not NULL and oper_time between '" +train_from+ "'and '" +train_to+ "' and oper_type='HOME_PRODUCT' "
        #sql1= "select agent_id as user_id,oper_obj as item_id from chsell_oper_data where agent_id is not NULL and oper_time between '" +train_from+ "'and'" +train_to+ "'and oper_type='HOME_PRODUCT'"
        engine=sqlalchemy.create_engine(engine_str) 
        dftrain=pd.read_sql(sql=sql1,con=engine)
        if is_test:
                sql2= " select t1.agent_id as user_id,t1.oper_time as time_stamp,cast(ifnull(qb.age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender,t1.oper_obj as item_id,IFNULL(t2.catalog,0) as item_catalog \
                from chsell_oper_data as t1 left JOIN chsell_product as t2 on t1.oper_obj= t2.id left JOIN chsell_agent as t3 on t1.agent_id = t3.id left JOIN bigdata.chsell_quick_bi as qb on t1.agent_id=qb.channel_id \
                where agent_id is not NULL and oper_time between '" +test_from+ "'and '" +test_to+ "' and oper_type='HOME_PRODUCT' "
                dftest=pd.read_sql(sql=sql2,con=engine)
                return dftrain,dftest
        else:
                return dftrain
def load_user_log(user_ids,train_from,train_to):
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql1=" select t1.agent_id as user_id,t1.oper_time as time_stamp,cast(ifnull(qb.age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender,t1.oper_obj as item_id,IFNULL(t2.catalog,0) as item_catalog  \
                from chsell_oper_data as t1 left JOIN chsell_product as t2 on t1.oper_obj= t2.id left JOIN chsell_agent as t3 on t1.agent_id = t3.id left JOIN bigdata.chsell_quick_bi as qb on t1.agent_id=qb.channel_id \
                where agent_id is not NULL and oper_time between '" +train_from+ "'and '" +train_to+ "' and oper_type='HOME_PRODUCT' and t1.agent_id in ("+','.join(user_ids)+")"
        #sql1="select agent_id as user_id,oper_obj as item_id from chsell_oper_data where agent_id is not NULL and oper_time between '" +train_from+ "'and'" +train_to+ "' and oper_type='HOME_PRODUCT' and agent_id in ("+user_id+")"
        engine=sqlalchemy.create_engine(engine_str)
        dftrain=pd.read_sql(sql=sql1,con=engine)
        return dftrain
def load_item_pool(train_from,train_to,as_list=True):
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql="select oper_obj as item_id from chsell_oper_data where agent_id is not NULL and oper_time between '" +train_from+ "'and'" +train_to+ "' and oper_type='HOME_PRODUCT'"
        engine=sqlalchemy.create_engine(engine_str) 
        item_pool=pd.read_sql(sql=sql,con=engine)
        if as_list:
            item_pool=list(item_pool['item_id'])
        return item_pool





'''deprecated
def NegativeSampling(data,item_pool,ratio):
    user_item=dict(data.groupby(['user_id','time_stamp'])['item_id'].agg(list))
    user_log=dict(data.groupby(['user_id'])['item_id'].agg(list))

    samples=dict()
    samples.setdefault('user_id',[])
    samples.setdefault('time_stamp',[])
    samples.setdefault('item_id',[])
    samples.setdefault('label',[])
    for user_field,item in user_item.items():
        n=0
        samples['user_id'].append(user_field[0])
        samples['time_stamp'].append(user_field[1])
        samples['item_id'].append(item[0])
        samples['label'].append(1)
        for i in range(7*ratio):
            sample_item=item_pool[np.random.randint(0,len(item_pool)-1)]
            if sample_item in user_log[user_field[0]]:
                continue
            samples['user_id'].append(user_field[0])
            samples['time_stamp'].append(user_field[1])
            samples['item_id'].append(sample_item)
            samples['label'].append(0)
        n+=1
        if n>ratio*len(user_log[user_field[0]]):
            break
    samples=pd.DataFrame(samples)
    return samples
   
    user_item=dict(data.groupby([user_col])[item_col].agg(list))
    samples=dict()
    samples.setdefault('user_id',[])
    samples.setdefault('item_id',[])
    samples.setdefault('label',[])
    for user,items in user_item.items():
        n=0
        for item in items:
            samples['user_id'].append(user)
            samples['item_id'].append(item)
            samples['label'].append(1)
        for i in range(7*ratio):
            sample_item=item_pool[np.random.randint(0,len(item_pool)-1)]
            if sample_item in items:
                continue
            samples['user_id'].append(user)
            samples['item_id'].append(sample_item)
            samples['label'].append(0)
        n+=1
        if n>ratio*len(items):
            break
    samples=pd.DataFrame(data=samples)
    return samples
         
    data['label']=1
    tmps=[]
    for user_id in data[user_col].unique():
        tmp=pd.DataFrame()
        pos_item=data.loc[data[user_col]==user_id,item_col]
        neg_item=[]
        n=0
        for i in range(0,50*ratio):
            item=item_pool[np.random.randint(0,len(item_pool)-1)]
            #zi=(np.sqrt(freq[item]/0.001)+1)/0.001
            #rnd=np.random.rand(1)[0]
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
 
def process_data(data,item_pool,test_data=None,sampling_ratio=5,user_cols=['user_id','age','gender'],item_cols=['item_id','item_catalog']):
    
    df=data.copy()
    df=NegativeSampling(df,item_pool,sampling_ratio)
    user_field=data.loc[:,user_cols].drop_duplicates()
    item_field=data.loc[:,item_cols].drop_duplicates()
    df=df.merge(user_field,how='left',on=['user_id'])
    df=df.merge(item_field,how='left',on=['item_id'])
    df['weekday']=df['time_stamp'].apply(lambda x: x.weekday())
    df['time_stamp']=df['time_stamp'].apply(lambda x: time.mktime(x.timetuple())).astype('int')
    layer=tf.keras.layers.experimental.preprocessing.Normalization()
    layer.adapt(df['time_stamp'].values)
    df['time_stamp']=layer(df['time_stamp'].values).numpy()
    train_dict=dict()
    if test_data is not None:
        df_test=test_data.copy()
        df_test=NegativeSampling(df_test,item_pool,sampling_ratio)
        user_field=test_data.loc[:,user_cols].drop_duplicates()
        item_field=test_data.loc[:,item_cols].drop_duplicates()
        df_test=df_test.merge(user_field,how='left',on=['user_id'])
        df_test=df_test.merge(item_field,how='left',on=['item_id'])
        df_test.fillna(0,inplace=True)
        df_test['weekday']=df_test['time_stamp'].apply(lambda x: x.weekday())
        df_test['time_stamp']=df_test['time_stamp'].apply(lambda x: time.mktime(x.timetuple())).astype('int')
        df_test['time_stamp']=layer(df_test['time_stamp'].values).numpy()
        test_dict=dict()

    for col in df.columns:
        train_dict.setdefault(col,0)
        train_dict[col]=df[col].values
        if test_data is not None:
            test_dict.setdefault(col,0)
            test_dict[col]=df_test[col].values
    train_label=df['label'].values
    train_set=tf.data.Dataset.from_tensor_slices((train_dict,train_label)).shuffle(10_000).batch(4096)
    if test_data is not None:
        test_label=df_test['label'].values
        test_set=tf.data.Dataset.from_tensor_slices((test_dict,test_label)).shuffle(1000).batch(512)
        return train_set,test_set
    return train_set

   
    for col in df.select_dtypes(exclude='object').columns:
        df[col]=df[col].astype('int')
        if test_data is not None:
            df_test[col]=df[col].astype('int')
   
    for col in df.drop(columns=['label']).columns:
        str_lookup=tf.keras.layers.experimental.preprocessing.StringLookup()
        int_lookup=tf.keras.layers.experimental.preprocessing.IntegerLookup()
        #one_hot_encoder=tf.keras.layers.experimental.preprocessing.CategoryEncoding()

        value=df[col].values
        if df[col].dtype=='object':
            str_lookup.adapt(value)
            one_hot_encoder.adapt(str_lookup(value))
            sparse_feature_train=one_hot_encoder(str_lookup(value)).numpy()
        else:
            int_lookup.adapt(value)
            one_hot_encoder.adapt(int_lookup(value))
            sparse_feature_train=one_hot_encoder(int_lookup(value)).numpy()
        train_dict.setdefault(col,0)
        train_dict[col]=sparse_feature_train

        if test_data is not None:
            value=df_test[col].values
            if df_test[col].dtype=='object':
                #str_lookup.adapt(value)
                #one_hot_encoder.adapt(str_lookup(value))
                sparse_feature_test=one_hot_encoder(str_lookup(value)).numpy()
            else:
                #int_lookup.adapt(value)
                #one_hot_encoder.adapt(int_lookup(value))
                sparse_feature_test=one_hot_encoder(int_lookup(value)).numpy()
            test_dict.setdefault(col,0)
            test_dict[col]=sparse_feature_test
    
    train_label=df['label'].values
    train_set=tf.data.Dataset.from_tensor_slices((train_dict,train_label)).shuffle(10_000).batch(4096)
    if test_data is not None:
        test_label=df_test['label'].values
        test_set=tf.data.Dataset.from_tensor_slices((test_dict,test_label)).shuffle(1000).batch(4096)
        return train_set,test_set
    return train_set
   
    '''