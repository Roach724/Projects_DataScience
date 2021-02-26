import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import warnings
import sqlalchemy
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow._api.v2 import sparse
warnings.filterwarnings('ignore')
user_tokens=12000
item_tokens=200
item_tag_tokens=5
item_catalog_tokens=13
gender_tokens=4
profit_type_tokens=4
settle_cycle_tokens=4
def sampling(data,ratio=3):
    item_pool=list(data['item_id'])
    user_item=dict(data.groupby(['user_id','time_stamp'])['item_id'].agg(list))                         
    data=data.to_dict(orient='list')
    for user_field,items in user_item.items():
        user_idx=data['user_id'].index(user_field[0])
        n=0
        for i in range(5*ratio):
            sample_item=item_pool[np.random.randint(0,len(item_pool)-1)]
            if sample_item in items:
                continue
            sample_idx=data['item_id'].index(sample_item)
            data['user_id'].append(user_field[0])
            data['age'].append(data['age'][user_idx])
            data['gender'].append(data['gender'][user_idx])
            data['time_stamp'].append(user_field[1])
            data['item_id'].append(sample_item)
            data['item_catalog'].append(data['item_catalog'][sample_idx])
            data['item_tag'].append(data['item_tag'][sample_idx])
            data['profit_type'].append(data['profit_type'][sample_idx])
            data['settle_cycle'].append(data['settle_cycle'][sample_idx])
            data['action'].append('OMIT')
            data['interest'].append(0)
            n+=1
            if n>ratio*len(user_field[0]):
                break
    data=pd.DataFrame(data)
    idx=np.random.permutation(len(data))
    data=data.iloc[idx,:].reset_index(drop=True)
    return data
def load_data(forth=28,until=1,test=False):
    start=(datetime.datetime.now()-datetime.timedelta(days=forth)).strftime('%Y/%m/%d')
    end=(datetime.datetime.now()-datetime.timedelta(days=until)).strftime('%Y/%m/%d')
    engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
    sql="select	op.agent_id as user_id,ifnull(qb.age,0) as age,ifnull(qb.gender,'UNK') as gender,UNIX_TIMESTAMP(op.oper_time) as time_stamp,op.oper_obj as item_id,IFNULL(prod.catalog,0) as item_catalog,replace(replace(IFNULL(prod.tags,'不可用'),'\\n',''),'，',' ') as item_tag,replace(IFNULL(prod.profit_type,'UNK'),'NONE','UNK') as profit_type,IFNULL(settle_cycle,'UNK') as settle_cycle,case od.`status` when 'TO_TRIAL' then 'TRIAL_FAIL' when 'TO_RETRIAL' then 'TRIAL_FAIL' else ifnull(od.`status`,'JUST_LOOKING') end as action, 1 as interest \
    from (((chsell_oper_data as op left join chsell_order as od on op.agent_id=od.agent_id and op.oper_obj=od.product_id and DATE_FORMAT(op.oper_time,'%%Y/%%m/%%d')=DATE_FORMAT(od.created_on,'%%Y/%%m/%%d')) inner join chsell_product as prod on op.oper_obj=prod.id) left join chsell_bill as bill on op.agent_id=bill.agent_id and od.id=bill.source_id and DATE_FORMAT(od.created_on,'%%Y/%%m/%%d')=DATE_FORMAT(bill.business_time,'%%Y/%%m/%%d')) left join bigdata.chsell_quick_bi as qb on op.agent_id = qb.channel_id \
    where op.oper_type='HOME_PRODUCT' and oper_time between '"+start +"' and '"+end+"' and op.agent_id is not NULL"
    data=pd.read_sql(sql=sql,con=engine_str)
    if test:
        sql="select	op.agent_id as user_id,ifnull(qb.age,0) as age,ifnull(qb.gender,'UNK') as gender,UNIX_TIMESTAMP(op.oper_time) as time_stamp,op.oper_obj as item_id,IFNULL(prod.catalog,0) as item_catalog,replace(replace(IFNULL(prod.tags,'不可用'),'\\n',''),'，',' ') as item_tag,replace(IFNULL(prod.profit_type,'UNK'),'NONE','UNK') as profit_type,IFNULL(settle_cycle,'UNK') as settle_cycle,case od.`status` when 'TO_TRIAL' then 'TRIAL_FAIL' when 'TO_RETRIAL' then 'TRIAL_FAIL' else ifnull(od.`status`,'JUST_LOOKING') end as action, 1 as interest \
        from (((chsell_oper_data as op left join chsell_order as od on op.agent_id=od.agent_id and op.oper_obj=od.product_id and DATE_FORMAT(op.oper_time,'%%Y/%%m/%%d')=DATE_FORMAT(od.created_on,'%%Y/%%m/%%d')) inner join chsell_product as prod on op.oper_obj=prod.id) left join chsell_bill as bill on op.agent_id=bill.agent_id and od.id=bill.source_id and DATE_FORMAT(od.created_on,'%%Y/%%m/%%d')=DATE_FORMAT(bill.business_time,'%%Y/%%m/%%d')) left join bigdata.chsell_quick_bi as qb on op.agent_id = qb.channel_id \
        where op.oper_type='HOME_PRODUCT' and oper_time between '"+(datetime.datetime.now()-datetime.timedelta(days=1)).strftime('%Y/%m/%d')+ \
        "' and '"+(datetime.datetime.now()).strftime('%Y/%m/%d')+"' and op.agent_id is not NULL"
        test=pd.read_sql(sql=sql,con=engine_str)
        return data,test
    return data
def train_prep_model(data):
    model=PreprocessLayer(training=True)
    model(data)
    model.training=False
    return model
def data_pipeline(data,test_data=None,ratio=3):
    data=sampling(data,ratio)
    ohe=OneHotEncoder()
    interest_label=data.pop('interest').values
    action_label=ohe.fit_transform(data.pop('action').values.reshape(-1,1)).toarray()
    X=data.to_dict(orient='list')
    for key,values in X.items():
        X[key]=np.array(values)
    if test_data is not None:
        test_data=sampling(test_data,ratio)
        interest_label_test=test_data.pop('interest').values
        action_label_test=ohe.transform(test_data.pop('action').values.reshape(-1,1)).toarray()
        X_test=test_data.to_dict(orient='list')
        for key,values in X_test.items():
            X_test[key]=np.array(values)
        return X,interest_label,action_label,X_test,interest_label_test,action_label_test
    return X,interest_label,action_label
def roc_auc(y_true,y_pred):
    return tf.py_function(roc_auc_score,(y_true,y_pred),tf.float16)
class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self,training=None,**kwargs):
        self.training=training
        super(PreprocessLayer,self).__init__(**kwargs)
        self.time_stamp_norm=tf.keras.layers.experimental.preprocessing.Normalization()
        #string_lookup
        self.user_id_lookup=tf.keras.layers.experimental.preprocessing.StringLookup()
        self.item_id_lookup=tf.keras.layers.experimental.preprocessing.StringLookup()
        self.gender_lookup=tf.keras.layers.experimental.preprocessing.StringLookup()
        self.profit_type_lookup=tf.keras.layers.experimental.preprocessing.StringLookup()
        self.settle_cycle_lookup=tf.keras.layers.experimental.preprocessing.StringLookup()
        #one_hot
        self.user_id_encoder=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=user_tokens)
        self.item_id_encoder=tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=item_tokens)
        self.gender_encoder=tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=gender_tokens)
        self.profit_type_encoder=tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=profit_type_tokens)
        self.settle_cycle_encoder=tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=settle_cycle_tokens)
        self.item_catalog_encoder=tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=item_catalog_tokens)
        self.item_tag_encoder=tf.keras.layers.experimental.preprocessing.TextVectorization()
        
    def build(self,input_shape):
        super(PreprocessLayer, self).build(input_shape)
    def call(self,inputs,training=None):
        
        if self.training:
            #self.user_id_lookup.adapt(inputs['user_id'])
            self.item_id_lookup.adapt(inputs['item_id'])
            self.gender_lookup.adapt(inputs['gender'])
            self.profit_type_lookup.adapt(inputs['profit_type'])
            self.settle_cycle_lookup.adapt(inputs['settle_cycle'])
            self.user_id_encoder.adapt(inputs['user_id'])
            self.item_id_encoder.adapt(inputs['item_id'])
            self.gender_encoder.adapt(inputs['gender'])
            self.settle_cycle_encoder.adapt(inputs['settle_cycle'])
            self.profit_type_encoder.adapt(inputs['profit_type'])
            self.item_tag_encoder.adapt(inputs['item_tag'])
            self.item_catalog_encoder.adapt(inputs['item_catalog'])
            
        #inputs['user_id']=self.user_id_lookup(inputs['user_id'])
            return inputs
        else:
            inputs['item_id']=self.item_id_lookup(inputs['item_id'])
            inputs['gender']=self.gender_lookup(inputs['gender'])
            inputs['profit_type']=self.profit_type_lookup(inputs['profit_type'])
            inputs['settle_cycle']=self.settle_cycle_lookup(inputs['settle_cycle'])
            self.time_stamp_norm.adapt(inputs['time_stamp'])

            inputs['time_stamp']=self.time_stamp_norm(inputs['time_stamp'])
            inputs['user_id']=self.user_id_encoder(inputs['user_id'])
            inputs['item_id']=self.item_id_encoder(inputs['item_id'])
            inputs['gender']=self.gender_encoder(inputs['gender'])
            inputs['profit_type']=self.profit_type_encoder(inputs['profit_type'])
            inputs['settle_cycle']=self.settle_cycle_encoder(inputs['settle_cycle'])
            inputs['item_tag']=self.item_tag_encoder(inputs['item_tag'])
            inputs['item_catalog']=self.item_catalog_encoder(inputs['item_catalog'])
            return inputs

    def get_config(self):
        config=super(PreprocessLayer,self).get_config()
        #config.update({'output_dim':self.output_dim})
        return config
class crosslayer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(crosslayer,self).__init__(**kwargs)
    def build(self,input_shape):
        super(crosslayer, self).build(input_shape)
        self.kernel=self.add_weight(shape=(8,input_shape[1]))
    def call(self,inputs):
        square_of_sum=tf.keras.backend.square(tf.keras.backend.sum(tf.keras.backend.dot(inputs,self.kernel),axis=1,keepdims=False))
        sum_of_square=tf.keras.backend.sum(tf.keras.backend.dot(tf.keras.backend.square(inputs),tf.keras.backend.square(self.kernel)),axis=1,keepdims=False)
        diff=0.5*tf.keras.backend.sum(square_of_sum-sum_of_square,axis=1,keepdims=False)
        return diff
def DeepFM(embedding_dim=8):
    #Input layers
    
    user_id_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='user_id')
    item_id_input=tf.keras.Input(shape=(item_tokens,),dtype=tf.int32,name='item_id')
    gender_input=tf.keras.Input(shape=(gender_tokens,),dtype=tf.int32,name='gender')
    profit_type_input=tf.keras.Input(shape=(profit_type_tokens,),dtype=tf.int32,name='profit_type')
    settle_cycle_input=tf.keras.Input(shape=(settle_cycle_tokens,),dtype=tf.int32,name='settle_cycle')
    item_tag_input=tf.keras.Input(shape=(item_tag_tokens,),dtype=tf.int32,name='item_tag')
    time_stamp_input=tf.keras.Input(shape=(1,),dtype=tf.float32,name='time_stamp')
    item_catalog_input=tf.keras.Input(shape=(item_catalog_tokens,),dtype=tf.int32,name='item_catalog')

    print(time_stamp_input)
    #Embedding
    user_embedding=tf.keras.layers.Embedding(user_tokens,embedding_dim,name='user_id_embedding')(user_id_input)
    item_embedding=tf.keras.layers.Embedding(item_tokens,embedding_dim,name='item_id_embedding')(item_id_input)
    gender_embedding=tf.keras.layers.Embedding(gender_tokens,output_dim=embedding_dim,name='gender_embedding')(gender_input)
    profit_type_embedding=tf.keras.layers.Embedding(profit_type_tokens,output_dim=embedding_dim,name='profit_type_embedding')(profit_type_input)
    settle_cycle_embedding=tf.keras.layers.Embedding(settle_cycle_tokens,output_dim=embedding_dim,name='settle_cycle_embedding')(settle_cycle_input)
    item_tag_embedding=tf.keras.layers.Embedding(200,embedding_dim,input_length=item_tag_tokens,name='item_tag_embedding')(item_tag_input)
    item_catalog_embedding=tf.keras.layers.Embedding(item_catalog_tokens,embedding_dim,name='item_catalog_embedding')(item_catalog_input)
    dense_features=tf.keras.layers.concatenate([user_embedding,item_embedding,gender_embedding,profit_type_embedding,settle_cycle_embedding,
                                                item_tag_embedding,item_catalog_embedding],axis=1,name='embedding_concatenate')
    sparse_features=tf.keras.layers.concatenate([item_id_input,gender_input,profit_type_input,settle_cycle_input,item_tag_input,item_catalog_input],
                                                name='sparse_feature_concatenate')
    
    #DNN
  
    dnn_l1=tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(0.1),
                        bias_regularizer=tf.keras.regularizers.l2(0.1),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer1')

    dnn_l2=tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(0.1),
                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer2')

    dnn_l3=tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(0.1),
                        bias_regularizer=tf.keras.regularizers.l2(0.1),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer3')

    dnn_l4=tf.keras.layers.Dense(4,kernel_regularizer=tf.keras.regularizers.l2(0.1),use_bias=False,
                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation=None,name='action')
    dnn_pred=tf.keras.layers.Activation(activation='softmax',name='action_softmax')
#FM
    fm_linear=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                    bias_regularizer=tf.keras.regularizers.l2(0.1),name='fm_linear')(sparse_features)
    fm_cross=crosslayer(name='fm_cross')(dense_features)

    #square_of_sum=tf.keras.backend.square(tf.keras.backend.sum(dense_features,axis=1,keepdims=True))
    #sum_of_square=tf.keras.backend.sum(tf.keras.backend.square(dense_features),axis=1,keepdims=True)
    
    #diff=tf.keras.layers.Subtract()([square_of_sum,sum_of_square])
    #fm_cross=tf.keras.layers.Multiply()([tf.constant([0.5]),tf.keras.backend.sum(diff,axis=2,keepdims=False)])         

    fm_logit=tf.keras.layers.Add(name='fm_combine')([fm_linear,fm_cross])

    interest=tf.keras.layers.Activation(activation='sigmoid',name='interest')(fm_logit)
    dnn1=dnn_l1(tf.keras.layers.concatenate([tf.keras.layers.Flatten()(dense_features),time_stamp_input]))
    
    dnn2=dnn_l2(dnn1)
    dnn3=dnn_l3(dnn2)
    dnn4=dnn_l4(dnn3)
    action=dnn_pred(dnn4)
    model=tf.keras.Model(inputs=[user_id_input,item_id_input,gender_input,profit_type_input,
                                settle_cycle_input,item_tag_input,time_stamp_input,item_catalog_input],
                                outputs=[interest,action])
    return model
'''
    [user_id_input,item_id_input,gender_input,profit_type_input,
                                settle_cycle_input,item_tag_input,time_stamp_input,item_catalog_input]
'''
model=DeepFM()