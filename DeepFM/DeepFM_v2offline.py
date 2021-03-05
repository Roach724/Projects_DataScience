import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import datetime
import sqlalchemy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import json
import os
warnings.filterwarnings('ignore')
forth=21
until=1
user_tokens=10000
item_tokens=150
item_tag_tokens=5
item_catalog_tokens=13
gender_tokens=4
profit_type_tokens=4
settle_cycle_tokens=4
embedding_dim=32
hidden_units=32
user_fields=['user_id','member_type']
item_fields=['item_id','item_catalog']
## offline components
engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
start=(datetime.datetime.now()-datetime.timedelta(days=forth)).strftime('%Y/%m/%d')
end=(datetime.datetime.now()-datetime.timedelta(days=until)).strftime('%Y/%m/%d')
sql_train="select op.agent_id as user_id,qb.number_type as member_type,op.oper_time as time_stamp,op.oper_obj as item_id,IFNULL(prod.catalog,0) as item_catalog,replace(replace(replace(IFNULL(prod.tags,'不可用'),'\\n',''),'，',' '),'、','') as item_tag,replace(IFNULL(prod.profit_type,'UNK'),'NONE','UNK') as profit_type,IFNULL(settle_cycle,'UNK') as settle_cycle, 1 as label \
from (((chsell_oper_data as op left join chsell_order as od on op.agent_id=od.agent_id and op.oper_obj=od.product_id and DATE_FORMAT(op.oper_time,'%%Y/%%m/%%d')=DATE_FORMAT(od.created_on,'%%Y/%%m/%%d')) inner join chsell_product as prod on op.oper_obj=prod.id) left join chsell_bill as bill on op.agent_id=bill.agent_id and od.id=bill.source_id and DATE_FORMAT(od.created_on,'%%Y/%%m/%%d')=DATE_FORMAT(bill.business_time,'%%Y/%%m/%%d')) left join bigdata.chsell_quick_bi as qb on op.agent_id = qb.channel_id \
where op.oper_type='HOME_PRODUCT' and oper_time between '"+start +"' and '"+end+"' and op.agent_id is not NULL and prod.deleted=0"
sql_test="select op.agent_id as user_id,cast(ifnull(qb.age,'unk') as SIGNED) as age,ifnull(qb.gender,'UNK') as gender,op.oper_time as time_stamp,op.oper_obj as item_id,IFNULL(prod.catalog,0) as item_catalog,replace(replace(replace(IFNULL(prod.tags,'不可用'),'\\n',''),'，',' '),'、','') as item_tag,replace(IFNULL(prod.profit_type,'UNK'),'NONE','UNK') as profit_type,IFNULL(settle_cycle,'UNK') as settle_cycle, 1 as label \
from (((chsell_oper_data as op left join chsell_order as od on op.agent_id=od.agent_id and op.oper_obj=od.product_id and DATE_FORMAT(op.oper_time,'%%Y/%%m/%%d')=DATE_FORMAT(od.created_on,'%%Y/%%m/%%d')) inner join chsell_product as prod on op.oper_obj=prod.id) left join chsell_bill as bill on op.agent_id=bill.agent_id and od.id=bill.source_id and DATE_FORMAT(od.created_on,'%%Y/%%m/%%d')=DATE_FORMAT(bill.business_time,'%%Y/%%m/%%d')) left join bigdata.chsell_quick_bi as qb on op.agent_id = qb.channel_id \
where op.oper_type='HOME_PRODUCT'and prod.deleted=0 and oper_time between '"+(datetime.datetime.now()-datetime.timedelta(days=1)).strftime('%Y/%m/%d')+ \
"' and '"+(datetime.datetime.now()).strftime('%Y/%m/%d')+"' and op.agent_id is not NULL"
sql_item="select oper_obj as item_id from chsell_oper_data where agent_id is not NULL and oper_time between '" +start+ "'and'" +end+ "' and oper_type='HOME_PRODUCT'"
sql_transaction="select co.agent_id as user_id,co.product_id as item_id,case co.`status` when 'TO_TRIAL' then 'TRIAL_FAIL' when 'TO_RETRIAL' then 'TRIAL_FAIL' when 'RETRIAL_FAIL' then 'TRIAL_FAIL' else co.`status`  end as label \
                from chsell_order as co inner join chsell_product as prod on co.product_id=prod.id"
def sampling(data,ratio=3):
    item_pool=list(data['item_id'])
    data['label']=1
    #data['date']=data['time_stamp'].apply(lambda x: datetime.datetime.strptime(x.strftime('%Y/%m/%d'),'%Y/%m/%d'))
    user_item=dict(data.groupby(['user_id'])['item_id'].agg(list))                         
    data=data.to_dict(orient='list')
    for user_field,items in user_item.items():
        user_idx=data['user_id'].index(user_field)
        n=0
        for i in range(5*ratio):
            sample_item=item_pool[np.random.randint(0,len(item_pool)-1)]
            if sample_item in items:
                continue
            sample_idx=data['item_id'].index(sample_item)
            data['user_id'].append(user_field)
            data['time_stamp'].append(data['time_stamp'][user_idx])
            data['member_type'].append(data['member_type'][user_idx])
            data['item_id'].append(sample_item)
            data['item_catalog'].append(data['item_catalog'][sample_idx])
            data['item_tag'].append(data['item_tag'][sample_idx])
            data['profit_type'].append(data['profit_type'][sample_idx])
            data['settle_cycle'].append(data['settle_cycle'][sample_idx])
            data['label'].append(0)
            n+=1
            if n>ratio*len(items):
                break
    data=pd.DataFrame(data)
    idx=np.random.permutation(len(data))
    data=data.iloc[idx,:].reset_index(drop=True)
    #data.drop(columns='time_stamp',inplace=True)
    return data
def load_log_data(test=False):
    data=pd.read_sql(sql=sql_train,con=engine_str)
    data=data.drop_duplicates()
    if test:
        test=pd.read_sql(sql=sql_test,con=engine_str)
        test=test.drop_duplicates()
        return data,test
    return data
def load_transaction_data():
    data=pd.read_sql(sql=sql_transaction,con=engine_str)
    return data
def data_pipeline(data,test_data=None,ratio=3):
    data=sampling(data,ratio)
    #data=make_time(data)
    data.drop(columns='time_stamp',inplace=True)
    label=data.pop('label').values
    X=data.to_dict(orient='list')
    for key,values in X.items():
        X[key]=np.array(values)
    if test_data is not None:
        test_data=sampling(test_data,ratio)
        #test_data=make_time(test_data)
        test_data.drop(columns='time_stamp',inplace=True)
        label_test=test_data.pop('label').values
        X_test=test_data.to_dict(orient='list')
        for key,values in X_test.items():
            X_test[key]=np.array(values)
        return X,label,X_test,label_test
    return X,label
def make_time(df):
    df['weekday']=df['time_stamp'].apply(lambda x: x.weekday())
    df['hour']=df['time_stamp'].apply(lambda x: x.hour)
    df['minute']=df['time_stamp'].apply(lambda x: x.minute)
    df['second']=df['time_stamp'].apply(lambda x: x.second)
    return df
def cross_validation(X,y,cv=3,n_splits=5,shuffle=True,random_state=None):
    #Cross validation
    strat=StratifiedKFold(n_splits=n_splits,shuffle=shuffle,random_state=random_state)
    X_df=pd.DataFrame(X)
    losses=[]
    aucs=[]
    recalls=[]
    for i in range(cv):
        for train_idx,test_idx in strat.split(X_df,y):
            X_train,X_test=X_df.iloc[train_idx,:],X_df.iloc[test_idx,:]
            y_train,y_test=y[train_idx],y[test_idx]
            X_train=X_train.to_dict(orient='list')
            X_test=X_test.to_dict(orient='list')
            for col,values in X_train.items():
                X_train[col]=np.array(values)
                X_test[col]=np.array(X_test[col])
            model=DeepFM()
            model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss=tf.keras.losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.Recall()])
            model.fit(X_train,y_train,epochs=8,verbose=2,batch_size=512)
            loss,auc,recall=model.evaluate(X_test,y_test,batch_size=256)
            losses.append(loss)
            aucs.append(auc)
            recalls.append(recall)
    return losses,aucs,recalls

'''
def roc_auc(y_true,y_pred):
    return tf.py_function(roc_auc_score,(y_true,y_pred),tf.float16)
'''

##model definition
class crosslayer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(crosslayer,self).__init__(**kwargs)
    def build(self,input_shape):
        super(crosslayer, self).build(input_shape)
        #self.kernel=self.add_weight(shape=(input_shape[-1],16))
    def call(self,inputs):
        square_of_sum=tf.keras.backend.square(tf.keras.backend.sum(inputs,axis=1,keepdims=True))
        sum_of_square=tf.keras.backend.sum(tf.keras.backend.square(inputs),axis=1,keepdims=True)
        diff=0.5*tf.keras.backend.sum(square_of_sum-sum_of_square,axis=2,keepdims=False)
        return diff
    def get_config(self):
        config=super(crosslayer,self).get_config()
        return config
def DeepFM():
    #Input layers
    user_id_input=tf.keras.Input(shape=(1,),dtype=tf.string,name='user_id')
    item_id_input=tf.keras.Input(shape=(1,),dtype=tf.string,name='item_id')
    gender_input=tf.keras.Input(shape=(1,),dtype=tf.string,name='gender')
    item_catalog_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='item_catalog')
    #weekday_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='weekday')
    #hour_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='hour')
    #minute_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='minute')
    #second_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='second')
    #profit_type_input=tf.keras.Input(shape=(profit_type_tokens,),dtype=tf.int32,name='profit_type')
    #settle_cycle_input=tf.keras.Input(shape=(settle_cycle_tokens,),dtype=tf.int32,name='settle_cycle')
    #time_stamp_input=tf.keras.Input(shape=(1,),dtype=tf.float32,name='time_stamp')
    #Hashing
    user_id_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=user_tokens,name='user_id_hash')(user_id_input)
    item_id_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=item_tokens,name='item_id_hash')(item_id_input)
    gender_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=gender_tokens,name='gender_hash')(gender_input)
    #Embedding
    user_embedding=tf.keras.layers.Embedding(user_tokens,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='user_id_embedding')(user_id_hash)
    item_embedding=tf.keras.layers.Embedding(item_tokens,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_id_embedding')(item_id_hash)
    gender_embedding=tf.keras.layers.Embedding(gender_tokens,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='gender_embedding')(gender_hash)
    item_catalog_embedding=tf.keras.layers.Embedding(item_catalog_tokens,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_catalog_embedding')(item_catalog_input)
    #weekday_embedding=tf.keras.layers.Embedding(8,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='weekday_embedding')(weekday_input)
    #hour_embedding=tf.keras.layers.Embedding(25,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='hour_embedding')(hour_input)
    #minute_embedding=tf.keras.layers.Embedding(61,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='minute_embedding')(minute_input)
    #second_embedding=tf.keras.layers.Embedding(61,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='second_embedding')(second_input)
    #profit_type_embedding=tf.keras.layers.Embedding(profit_type_tokens,output_dim=embedding_dim,name='profit_type_embedding')(profit_type_input)
    #settle_cycle_embedding=tf.keras.layers.Embedding(settle_cycle_tokens,output_dim=embedding_dim,name='settle_cycle_embedding')(settle_cycle_input)
    #item_tag_embedding=tf.keras.layers.Embedding(200,embedding_dim,input_length=item_tag_tokens,name='item_tag_embedding')(item_tag_input)
    

    dense_features=tf.keras.layers.concatenate([user_embedding,item_embedding,gender_embedding,
                                                item_catalog_embedding],axis=1,name='embedding_concatenate')
    #sparse_features=tf.keras.layers.concatenate([item_id_input,gender_input,item_catalog_input,weekday_input,hour_input,minute_input,second_input],
                                                #name='sparse_feature_concatenate')
                            
    
    #DNN
    dnn_l1=tf.keras.layers.Dense(hidden_units,kernel_regularizer=tf.keras.regularizers.l2(0.2),
                        bias_regularizer=tf.keras.regularizers.l2(0.2),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer1')
    dropout1=tf.keras.layers.Dropout(0.7)
    dnn_l2=tf.keras.layers.Dense(hidden_units,kernel_regularizer=tf.keras.regularizers.l2(0.2),
                        bias_regularizer=tf.keras.regularizers.l2(0.2),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer2')
    dropout2=tf.keras.layers.Dropout(0.6)
    dnn_l3=tf.keras.layers.Dense(hidden_units,kernel_regularizer=tf.keras.regularizers.l2(0.2),
                        bias_regularizer=tf.keras.regularizers.l2(0.2),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer3')
    dropout3=tf.keras.layers.Dropout(0.8)
    dnn_l4=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.1),use_bias=False,
                        bias_regularizer=tf.keras.regularizers.l2(0.2),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation=None,name='dnn_layer4')
#FM
    fm_linear=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                   bias_regularizer=tf.keras.regularizers.l2(0.1),name='fm_linear')(tf.keras.layers.Flatten()(dense_features))
    fm_cross=crosslayer(name='fm_cross')(dense_features)

    fm_logit=tf.keras.layers.Add(name='fm_combine')([fm_linear,fm_cross])

    dnn1=dnn_l1(tf.keras.layers.Flatten()(dense_features))
    dnn1_drop=dropout1(dnn1)
    dnn2=dnn_l2(dnn1_drop)
    dnn2_drop=dropout2(dnn2)
    dnn3=dnn_l3(dnn2_drop)
    dnn3_drop=dropout3(dnn3)
    dnn_logit=dnn_l4(dnn3_drop)
    pred=tf.keras.layers.Activation(activation='sigmoid',name='sigmoid')(tf.keras.layers.Add()([fm_logit,dnn_logit]))

    model=tf.keras.Model(inputs=[user_id_input,item_id_input,gender_input,item_catalog_input],
                        outputs=pred)
    return model

# online components
def retrain(model,data,epochs=3,learning_rate=0.01):
    data=pd.DataFrame(data)
    data=sampling(data,0)
    data=data.to_dict(orient='list')
    for key in data.keys():
        data[key]=np.array(data[key])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate))
    model.fit(data,epochs=epochs,batch_size=32)
    #similarity_matrix(model)
    return model

def guess_you_like(model,df,topK=36,json_like=True,predict_type='single'):
        X=df
        for key in user_fields:
            X[key]=np.repeat(X[key],len(X['item_id']))
        pred=model.predict(X)
        df=pd.DataFrame(X)
        df['score']=pred
        df['rank']=df.groupby(['user_id'])['score'].rank(method='first',ascending=False).sort_values()
        df=df.sort_values(by=['user_id','rank']).reset_index(drop=True)
        recmd_list=df.loc[:,['user_id','item_id','item_name','item_catalog','rank','score']]
        if predict_type=='single':
            if topK=='all':
                pass
            else:
                recmd_list=df[df['rank']<=topK]
                recmd_list=recmd_list.sort_values(by=['user_id','rank'])
            if json_like:
                result_set=list()
                user_ids=list(recmd_list['user_id'].unique())
                for user_id in user_ids:
                    user_dict=dict()
                    user_dict.setdefault('user_id',0)
                    user_dict.setdefault('item_list',[])
                    user_dict['user_id']=user_id
                    item_list=list(recmd_list.loc[recmd_list['user_id']==user_id,'item_id'])
                    rank_list=list(recmd_list.loc[recmd_list['user_id']==user_id,'rank'])
                    name_list=list(recmd_list.loc[recmd_list['user_id']==user_id,'item_name'])
                    for i in range(len(item_list)):
                        item_dict=dict()
                        item_dict.setdefault('item_id',0)
                        item_dict.setdefault('item_name',0)
                        item_dict.setdefault('rank',0)
                        item_dict['item_id']=item_list[i]
                        item_dict['item_name']=name_list[i]
                        item_dict['rank']=rank_list[i]
                        user_dict['item_list'].append(item_dict)
                    result_set.append(user_dict)
                return result_set
            return recmd_list
        elif predict_type=='class':
            class_list=recmd_list.groupby(['user_id','item_catalog'],as_index=False)['score'].agg('mean')
            class_list['catalog_rank']=class_list.groupby(['user_id'])['score'].rank(method='first',ascending=False).sort_values()
            class_list=class_list.sort_values(by=['user_id','catalog_rank'])
            class_list=class_list.loc[:,['user_id','item_catalog','catalog_rank']]
            if json_like:
                result_set=list()
                user_ids=list(class_list['user_id'].unique())
                for user_id in user_ids:
                    user_dict=dict()
                    user_dict.setdefault('user_id',0)
                    user_dict.setdefault('item_list',[])
                    user_dict['user_id']=user_id
                    item_list=list(class_list.loc[class_list['user_id']==user_id,'item_catalog'])
                    rank_list=list(class_list.loc[class_list['user_id']==user_id,'catalog_rank'])
                    #name_list=list(recmd_list.loc[recmd_list['user_id']==user_id,'item_name'])
                    for i in range(len(item_list)):
                        item_dict=dict()
                        item_dict.setdefault('item_catalog',0)
                        #item_dict.setdefault('item_name',0)
                        item_dict.setdefault('catalog_rank',0)
                        item_dict['item_catalog']=item_list[i]
                        #item_dict['item_name']=name_list[i]
                        item_dict['catalog_rank']=rank_list[i]
                        user_dict['item_list'].append(item_dict)
                    result_set.append(user_dict)
                return result_set
            return class_list
#custom_objects={'crosslayer':dfm.crosslayer}
#with tf.keras.utils.custom_object_scope(custom_objects):
#new_model=tf.keras.models.load_model('E:\\kyk-ml\\deeplearning\DeepFM\\DeepFMv2.h5',custom_objects=custom_objects)
#new_model.set_weights('DeepFMv2_weights.h5')
def load_model_weights(model,path=r'.\deeplearning\DeepFM\DeepFM.h5'):
        print("xxxxxx")
        model.load_weights(path)
        setting_file = path.replace('.h5', '.setting')
        if os.path.isfile(setting_file):
            with open(setting_file, 'r') as f:
                model.setting = json.load(f)
                print(model.setting)
                return model.setting['MAX_ID']
def save_model_weights(model, path=r'.\deeplearning\DeepFM\DeepFM.h5', new_max=None):
        model.save_weights(path)
        if model.setting is None:
            model.setting = json.loads(json.dumps({'MAX_ID': new_max, 'UPDATE_TIME': str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 'VERSION': '0.1'}))
        else:
            model.setting['MAX_ID'] = new_max
            model.setting['UPDATE_TIME'] = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        setting_file = path.replace('.h5', '.setting')
        with open(setting_file, 'w') as f:
            json.dump(model.setting, f)



def get_users(user_id):
    # offline
    user_feature=dict()
    data=pd.read_sql(sql="select channel_id as user_id,ifnull(gender,'UNK') as gender from bigdata.chsell_quick_bi where channel_id='"+user_id+"'",
                    con=engine_str)
    user_feature=data.to_dict(orient='list')
    for key in user_feature.keys():
        user_feature[key]=np.array(user_feature[key])
    #online
    #---TODO---
    return user_feature

def get_items():
    # offline
    sql="select id as item_id,ifnull(catalog,0) as item_catalog,name as item_name,replace(IFNULL(profit_type,'UNK'),'NONE','UNK') as profit_type, \
        IFNULL(settle_cycle,'UNK') as settle_cycle from chsell_product where deleted=0"
    data=pd.read_sql(sql=sql,con=engine_str)
    item_feature=data.to_dict(orient='list')
    for key in item_feature.keys():
        item_feature[key]=np.array(item_feature[key])
    #online
    #---TODO---
    return item_feature

def similarity_matrix(model):
    #offline
    item_list=pd.read_sql(sql=sql_item,con=engine_str)
    item_list=list(item_list['item_id'].drop_duplicates())
    #online
    '''
    if online:
        item_list=get_items()
    '''
    hash_layer=model.get_layer(name='item_id_hash')
    embedding_layer=model.get_layer(name='item_id_embedding')
    data_dict=dict()
    for item in item_list:
        products=[]
        data_dict.setdefault(item,[])
        item_embedding=embedding_layer(hash_layer([item])).numpy()
        item_embedding=item_embedding/np.linalg.norm(item_embedding)
        for candidate in item_list:
            candidate_embedding=embedding_layer(hash_layer([candidate])).numpy()
            candidate_embedding=candidate_embedding/np.linalg.norm(candidate_embedding)
            product=np.dot(item_embedding,candidate_embedding.T)[0,0]
            products.append(product)
        data_dict[item]=products
    matrix=pd.DataFrame(data_dict)
    matrix.index=matrix.columns
    model.similarity_matrix=matrix

def user_cold_start(new_user,model):
    #offline
    hash_layer=model.get_layer(name='user_id_hash')
    embedding_layer=model.get_layer(name='user_id_embedding')
    user_embedding=embedding_layer(hash_layer(new_user))
    #oneline
    #todo
    return user_embedding

def item_cold_start(new_item,model):
    #offline
    hash_layer=model.get_layer(name='item_id_hash')
    embedding_layer=model.get_layer(name='item_id_embedding')
    item_embedding=embedding_layer(hash_layer(new_item))

    #online
    #todo
    return item_embedding
