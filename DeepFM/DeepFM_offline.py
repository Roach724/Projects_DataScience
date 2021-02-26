import numpy as np
from numpy.core.numeric import full
import pandas as pd
import datetime
import time
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.keras.backend import convert_inputs_if_ragged
from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.platform import gfile
import warnings
import sqlalchemy
import os
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
#definition of layers/models
class DNN(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(DNN,self).__init__(**kwargs)
        
        '''
        self.dense1=tf.keras.layers.Dense(256,kernel_regularizer=tf.keras.regularizers.l2(1.2),
                                            bias_regularizer=tf.keras.regularizers.l2(1.2),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer1')
        self.dropout1=tf.keras.layers.Dropout(0.4)
        '''
        self.dense2=tf.keras.layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(1.15),
                                            bias_regularizer=tf.keras.regularizers.l2(1.15),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer2')
        self.dropout2=tf.keras.layers.Dropout(0.8)
        
        self.dense3=tf.keras.layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(0.3),
                                            bias_regularizer=tf.keras.regularizers.l2(0.3),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer3')
        self.dropout3=tf.keras.layers.Dropout(0.7)
        self.dense4=tf.keras.layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(0.3),
                                            bias_regularizer=tf.keras.regularizers.l2(0.3),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer4')
        self.dropout4=tf.keras.layers.Dropout(0.3)
        self.dense5=tf.keras.layers.Dense(1,use_bias=False,kernel_initializer=tf.keras.initializers.glorot_normal(),name='dnn_layer5')
    def build(self,input_shape):
        super(DNN, self).build(input_shape)
    def call(self,inputs,training=None):
        '''
        dense1=self.dense1(inputs)
        if training:
            dense1=self.dropout1(dense1)
        '''
        dense2=self.dense2(inputs)
        if training:
            dense2=self.dropout2(dense2)
        
        dense3=self.dense3(dense2)
        if training:
            dense3=self.dropout3(dense3)
        dense4=self.dense4(dense3)
        if training:
            dense4=self.dropout4(dense4)
        dense5=self.dense5(dense4)
        return dense5
    def get_config(self):
        config=super(DNN,self).get_config()
        return config
class FM(tf.keras.layers.Layer):
    def __init__(self,output_dim,**kwargs):
        super(FM,self).__init__(**kwargs)
        self.output_dim=output_dim
    def build(self,input_shape):
        self.linear=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.8),
                                          bias_regularizer=tf.keras.regularizers.l2(1.2),name='linear')
        self.kernel = self.add_weight(shape=(input_shape[-1],self.output_dim),
                                     initializer=tf.keras.initializers.glorot_normal(),
                                     trainable=True,name='cross')
        self.logit=tf.keras.layers.Add()
        super(FM, self).build(input_shape)
    def call(self,inputs):
        linear=self.linear(inputs)
        a=tf.keras.backend.pow(tf.keras.backend.dot(inputs,self.kernel),2)
        b=tf.keras.backend.dot(tf.keras.backend.pow(inputs,2),tf.keras.backend.pow(self.kernel,2))
        cross=0.5*tf.keras.backend.sum(a-b,1,keepdims=True) 
        logit=self.logit([linear,cross])
        return logit

    def get_config(self):
        config=super(FM,self).get_config()
        config.update({'output_dim':self.output_dim})
        return config

#definition of the deepFM model     
class DeepFatorizationMachine(tf.keras.Model):
    def __init__(self,embedding_dim=64,records=1000,num_bins=31268,**kwargs):
        super(DeepFatorizationMachine,self).__init__(**kwargs)
        self.embedding_dim=embedding_dim
        self.num_bins=num_bins
        self.user_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=self.num_bins)
        self.gender_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=4)
        self.item_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=500)
        self.records=records

        self.user_embedding=tf.keras.layers.Embedding(input_dim=self.num_bins,output_dim=self.embedding_dim,name='user_id_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.gender_embedding=tf.keras.layers.Embedding(input_dim=5,output_dim=8,name='gender_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.age_embedding=tf.keras.layers.Embedding(input_dim=100,output_dim=self.embedding_dim,name='age_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.weekday_embedding=tf.keras.layers.Embedding(input_dim=8,output_dim=16,name='weekday_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.hour_embedding=tf.keras.layers.Embedding(input_dim=25,output_dim=8,name='hour_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.minute_embedding=tf.keras.layers.Embedding(input_dim=61,output_dim=8,name='minute_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.second_embedding=tf.keras.layers.Embedding(input_dim=61,output_dim=8,name='second_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.item_embedding=tf.keras.layers.Embedding(input_dim=500,output_dim=self.embedding_dim,name='item_id_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.catalog_embedding=tf.keras.layers.Embedding(input_dim=40,output_dim=16,name='catalog_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.id_layer=IDLayer()

        self.flatten=tf.keras.layers.Flatten()
        self.FM=FM(32)
        self.DNN=DNN()
        self.add=tf.keras.layers.Add()
        self.pred=tf.keras.layers.Activation(activation='sigmoid')
    def build(self,input_shape):
        super(DeepFatorizationMachine, self).build(input_shape)
    def call(self,inputs,training=None):
        
        user_input=inputs['user_id']
        age_input=inputs['age']
        gender_input=inputs['gender']
        weekday_input=inputs['weekday']
        hour_input=inputs['hour']
        minute_input=inputs['minute']
        second_input=inputs['second']
        item_input=inputs['item_id']
        catalog_input=inputs['item_catalog']
        #Embedding
        user_embedding=self.flatten(self.user_embedding(user_input))
        age_embedding=self.flatten(self.age_embedding(age_input))
        gender_embedding=self.flatten(self.gender_embedding(gender_input))
        weekday_embedding=self.flatten(self.weekday_embedding(weekday_input))
        hour_embedding=self.flatten(self.hour_embedding(hour_input))
        minute_embedding=self.flatten(self.minute_embedding(minute_input))
        second_embedding=self.flatten(self.second_embedding(second_input))
        item_embedding=self.flatten(self.item_embedding(item_input))
        catalog_embedding=self.flatten(self.catalog_embedding(catalog_input))

        embeddings=tf.concat([user_embedding,age_embedding,gender_embedding,weekday_embedding,
                            hour_embedding,minute_embedding,second_embedding,item_embedding,catalog_embedding],axis=1)
        
        #FM
        fm_pred=self.FM(embeddings)
        #DNN
        dnn_pred=self.DNN(embeddings)
        #combine the outputs from the two layers
        add=self.add([fm_pred,dnn_pred])
        #map the output by sigmoid function
        outputs=self.pred(add)

       
        
        return outputs
    def get_config(self):
        config=super(DeepFatorizationMachine,self).get_config()
        config.update({'embedding_dim':self.embedding_dim,'nim_bins':self.num_bins})
        return config

    #tool function
    def create_profile(self,data,item_pool=None,sampling_ratio=5):
        #用户/产品画像在此计算
        df=data.copy()
        if item_pool is not None:
            df=NegativeSampling(df,item_pool,sampling_ratio)
            user_field=data.loc[:,self.user_cols].drop_duplicates()
            item_field=data.loc[:,self.item_cols].drop_duplicates()
            df=df.merge(user_field,how='inner',on='user_id')
            df=df.merge(item_field,how='inner',on='item_id')
        df['weekday']=df['time_stamp'].apply(lambda x: x.weekday())
        df['hour']=df['time_stamp'].apply(lambda x: x.hour)
        df['minute']=df['time_stamp'].apply(lambda x: x.minute)
        df['second']=df['time_stamp'].apply(lambda x: x.second)
        self.age_mean=int(df.loc[df['age']!=0,'age'].mean())
        df.loc[df['age']==0,'age']=self.age_mean
        df.drop(columns='time_stamp',inplace=True)
        return df

    def apply_tensorflow_prerocessing(self,data):
        df=data.copy()
        df['user_id']=self.user_hash(df['user_id'].values).numpy()
        df['gender']=self.gender_hash(df['gender'].values).numpy()
        df['item_id']=self.item_hash(df['item_id'].values).numpy()
        return df
    
    def data_pipeline(self,sampling_ratio=5,full_train=False,increment=True,test=False,user_cols=['user_id','age','gender'],item_cols=['item_id','item_catalog'],forth=28,until=1):
        #加载数据
        delta0=datetime.timedelta(days=until)
        delta=datetime.timedelta(days=forth)
        train_to=(datetime.datetime.now()-delta0).strftime(format='%Y/%m/%d')
        train_from=(datetime.datetime.now()-delta).strftime(format='%Y/%m/%d')
        item_pool=load_item_pool((datetime.datetime.now()-datetime.timedelta(days=28)).strftime(format='%Y/%m/%d'),train_to=datetime.datetime.now().strftime(format='%Y/%m/%d'))
        if increment:
            train_from=(datetime.datetime.now()-datetime.timedelta(days=forth)).strftime(format='%Y/%m/%d')
            train_to=(datetime.datetime.now()).strftime(format='%Y/%m/%d')
            data=load_full_log(train_from,train_to,is_test=False)
        else:
            if test:
                test_from=(datetime.datetime.now()-datetime.timedelta(days=1)).strftime(format='%Y/%m/%d')
                test_to=(datetime.datetime.now()).strftime(format='%Y/%m/%d')
                data,test_data=load_full_log(train_from,train_to,test_from,test_to,is_test=True)
            else:
                data=load_full_log(train_from,train_to,is_test=False)

        self.user_cols=user_cols
        self.item_cols=item_cols
        #负采样，创建数据字典
        df=self.create_profile(data,item_pool,sampling_ratio)
        train_dict=dict()
        if full_train:
            pass
            '''
            self.user_hash.adapt(df['user_id'].values,reset_state=False)
            self.gender_hash.adapt(df['gender'].values,reset_state=False)
            self.item_hash.adapt(df['item_id'].values,reset_state=False)
            '''
           
        df=self.apply_tensorflow_prerocessing(df)
        if test:
            test_dict=dict()
            df_test=self.create_profile(test_data,item_pool,sampling_ratio)
            df_test=self.apply_tensorflow_prerocessing(df_test)
        #建立tensorflow dataset对象
        for col in df.columns:
            train_dict.setdefault(col,0)
            train_dict[col]=df[col].values
            if test:
                test_dict.setdefault(col,0)
                test_dict[col]=df_test[col].values
        train_label=df['label'].values
        train_set=tf.data.Dataset.from_tensor_slices((train_dict,train_label))

        if test:
            test_label=df_test['label'].values
            test_set=tf.data.Dataset.from_tensor_slices((test_dict,test_label))

            return train_set,test_set
        return train_set
    ##for inference
    def feeling_lucky(self,user_ids,topK=5,scores=False):
        #extract items over the past delta days
        delta=datetime.timedelta(days=7)
        now=datetime.datetime.now().strftime(format='%Y/%m/%d')
        past=(datetime.datetime.now()-delta).strftime(format='%Y/%m/%d')
        #use all items as retrieval candidates
        candidates=load_item_pool(past,now,as_list=False)
        candidates=candidates.drop_duplicates()
        #get item profile
        engine_str0='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql0="select id as item_id,catalog as item_catalog ,name as item_name from chsell_product"
        item_type=pd.read_sql(sql=sql0,con=engine_str0).fillna(0)
        item_field=candidates.merge(item_type,how='inner',on='item_id')
        #get user profile
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/bigdata'
        sql="select channel_id as user_id,cast(ifnull(age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender\
            from chsell_quick_bi where channel_id in ("+','.join(user_ids)+")"
        user_field=pd.read_sql(sql=sql,con=engine_str)
        #construct pandas dataframe and join to match side information
        tmp=[]
        for user_id in user_ids:
            df=pd.DataFrame()
            df['item_id']=item_field['item_id']
            df['user_id']=user_id.replace("'",'')
            tmp.append(df)
        df=pd.concat(tmp,axis=0)
        df['time_stamp']=datetime.datetime.now()
        df=df.merge(user_field,how='inner',on='user_id')
        df=df.merge(item_field,how='inner',on='item_id')
        #calculate profiles and apply preprocessing
        X=self.create_profile(df)
        X=self.apply_tensorflow_prerocessing(X)
        #make a prediction
        X=X.to_dict(orient='list')
        for col,values in X.items():
            X[col]=np.array(values)
        dataset=tf.data.Dataset.from_tensor_slices(X).batch(8192)
        pred=self.predict(dataset)
        X=pd.DataFrame(X)
        #output a recommendation list
        df['score']=pred
        df['rank']=df.groupby(['user_id'])['score'].rank(method='first',ascending=False).sort_values()
        df=df.sort_values(by=['user_id','rank']).reset_index(drop=True)
        df=df.loc[:,['user_id','item_id','rank','score']]
        if not scores:
            df=df.merge(item_field,how='inner',on='item_id').loc[:,['user_id','item_name','rank']]
            recmd_list=df[df['rank']<=topK].set_index(['user_id','rank']).unstack(-1).reset_index()
            #.droplevel(1).reset_index().set_index(['user_id','rank']).unstack(-1).reset_index()
            recmd_list.columns=[int(col) if isinstance(col,int) else col for col in recmd_list.columns.droplevel(0)]
            name_dict={'':'user_id'}
            for i in range(topK):
                prefix=float(i+1)
                name_dict.setdefault(prefix,'')
                name_dict[prefix]='item_'+str(int(i+1))
            recmd_list.rename(columns=name_dict,inplace=True)
            return recmd_list
        else:
            df=df.merge(item_field,how='inner',on='item_id').loc[:,['user_id','item_id','item_name','rank','score']]
            recmd_list=df[df['rank']<=topK]
            recmd_list.drop(columns=['rank'],inplace=True)
            recmd_list=recmd_list.sort_values(by=['user_id','score'],ascending=False)
            return recmd_list
            
    def load_model_weights(self,path=r'.\deeplearning\DeepFM\DeepFM.h5'):
        data={'user_id':np.array([1,1]),'item_id':np.array([1,1]),
            'gender':np.array([1,1]),'age':np.array([1,1]),'item_catalog':np.array([1,1]),'hour':np.array([1,1]),
            'minute':np.array([1,1]),'second':np.array([1,1]),'weekday':np.array([1,1])}
        self(data)
        self.load_weights(path)

    def save_model_weights(self,path=r'.\deeplearning\DeepFM\DeepFM.h5'):
        self.save_weights(path)
    def save_pb(self,path=r'.\deeplearning\DeepFM'):
        self.load_model_weights(r'.\deeplearning\DeepFM\DeepFM.h5')
        full_model=tf.function(lambda x: self(x))
        full_model=full_model.get_concrete_function({'user_id':np.array([1,1]),'item_id':np.array([1,1]),
                'gender':np.array([1,1]),'age':np.array([1,1]),'item_catalog':np.array([1,1]),'hour':np.array([1,1]),
                'minute':np.array([1,1]),'second':np.array([1,1]),'weekday':np.array([1,1])})
        frozen_func=convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=path,
                    name='DeepFM.pb',
                    as_text=False)
        '''
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 50)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)
    
       '''
    def guess_you_like(self,df,topK=36,json_like=True):
        X=self.apply_tensorflow_prerocessing(df)
        X=X.to_dict(orient='list')
        for feature,values in X.items():
            X[feature]=np.array(values)
        ds=tf.data.Dataset.from_tensor_slices(X).batch(32)
        pred=self.predict(ds)
        X=pd.DataFrame(X)

        df['score']=pred
        df['rank']=df.groupby(['user_id'])['score'].rank(method='first',ascending=False).sort_values()
        df=df.sort_values(by=['user_id','rank']).reset_index(drop=True)
        recmd_list=df.loc[:,['user_id','item_id','item_name','rank']]
        
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




    def mock_json(self,user_id):
        delta=datetime.timedelta(days=14)
        now=datetime.datetime.now().strftime(format='%Y/%m/%d')
        past=(datetime.datetime.now()-delta).strftime(format='%Y/%m/%d')
        #use all items as retrieval candidates
        candidates=load_item_pool(past,now,as_list=False)
        candidates=candidates.drop_duplicates()
        #get item profile
        engine_str0='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql0="select id as item_id,catalog as item_catalog ,name as item_name from chsell_product"
        item_type=pd.read_sql(sql=sql0,con=engine_str0).fillna(0)
        item_field=candidates.merge(item_type,how='inner',on='item_id')
        #get user profile
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/bigdata'
        sql="select channel_id as user_id,cast(ifnull(age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender\
            from chsell_quick_bi where channel_id = '"+user_id+"'" 
        user_field=pd.read_sql(sql=sql,con=engine_str)
        #construct pandas dataframe and join to match side information
        df=pd.DataFrame()
        df['item_id']=item_field['item_id']
        df['user_id']=user_id.replace("'",'')
       
        df['time_stamp']=datetime.datetime.now()

        df=df.merge(user_field,how='inner',on='user_id')
        df=df.merge(item_field,how='inner',on='item_id')
        df=self.create_profile(df)
        return df.to_json(orient='records')

    def mock_user_pipeline(self,user_id,records=10,user_cols=['user_id','age','gender'],item_cols=['item_id','item_catalog']):

        train_from=(datetime.datetime.now()-datetime.timedelta(days=9)).strftime(format='%Y/%m/%d')
        item_pool=load_item_pool((datetime.datetime.now()-datetime.timedelta(days=14)).strftime(format='%Y/%m/%d'),train_to=datetime.datetime.now().strftime(format='%Y/%m/%d'))
        data=load_user_log(user_id,train_from,records)
        df=NegativeSampling(data,item_pool,ratio=7)
        user_field=data.loc[:,user_cols].drop_duplicates()
        item_field=data.loc[:,item_cols].drop_duplicates()
        df=df.merge(user_field,how='inner',on='user_id')
        df=df.merge(item_field,how='inner',on='item_id')
        print(df.columns)
        df['weekday']=df['time_stamp'].apply(lambda x: x.weekday())
        df['hour']=df['time_stamp'].apply(lambda x: x.hour)
        df['minute']=df['time_stamp'].apply(lambda x: x.minute)
        df['second']=df['time_stamp'].apply(lambda x: x.second)
        self.age_mean=int(df.loc[df['age']!=0,'age'].mean())
        df.loc[df['age']==0,'age']=self.age_mean
        df.drop(columns='time_stamp',inplace=True)

        train_dict=dict()
        df=self.apply_tensorflow_prerocessing(df)
        for col in df.columns:
            train_dict.setdefault(col,0)
            train_dict[col]=df[col].values
        train_label=df['label'].values
        train_set=tf.data.Dataset.from_tensor_slices((train_dict,train_label))
        train_set=train_set.shuffle(len(train_set)).batch(16)
        return train_set
    def mock_full_pipeline(self,records=10,user_cols=['user_id','age','gender'],item_cols=['item_id','item_catalog']):
    
        train_from=(datetime.datetime.now()-datetime.timedelta(days=1)).strftime(format='%Y/%m/%d')
        train_to=(datetime.datetime.now()).strftime(format='%Y/%m/%d')
        item_pool=load_item_pool((datetime.datetime.now()-datetime.timedelta(days=28)).strftime(format='%Y/%m/%d'),train_to=datetime.datetime.now().strftime(format='%Y/%m/%d'))
        data=load_full_log(train_from,train_to,records)
        df=NegativeSampling(data,item_pool,ratio=3)
        user_field=data.loc[:,user_cols].drop_duplicates()
        item_field=data.loc[:,item_cols].drop_duplicates()
        df=df.merge(user_field,how='inner',on='user_id')
        df=df.merge(item_field,how='inner',on='item_id')
        df['weekday']=df['time_stamp'].apply(lambda x: x.weekday())
        df['hour']=df['time_stamp'].apply(lambda x: x.hour)
        df['minute']=df['time_stamp'].apply(lambda x: x.minute)
        df['second']=df['time_stamp'].apply(lambda x: x.second)
        self.age_mean=int(df.loc[df['age']!=0,'age'].mean())
        df.loc[df['age']==0,'age']=self.age_mean
        df.drop(columns='time_stamp',inplace=True)

        train_dict=dict()
        df=self.apply_tensorflow_prerocessing(df)
        for col in df.columns:
            train_dict.setdefault(col,0)
            train_dict[col]=df[col].values
        train_label=df['label'].values
        train_set=tf.data.Dataset.from_tensor_slices((train_dict,train_label))
        train_set=train_set.shuffle(len(train_set)).batch(4096)
        return train_set

    
    def sampling(self,data,ratio=3):
        item_pool=load_item_pool((datetime.datetime.now()-datetime.timedelta(days=14)).strftime(format='%Y/%m/%d'),
                                    train_to=datetime.datetime.now().strftime(format='%Y/%m/%d'))
        user_item=dict(data.groupby(['user_id'])['item_id'].agg(list))
        data['label']=1                           
        data=data.to_dict(orient='list')
        data.setdefault('label',[])
        for user_id,items in user_item.items():
            user_idx=data['user_id'].index(user_id)
            n=0
        for i in range(5*ratio):
            sample_item=item_pool[np.random.randint(0,len(item_pool)-1)]
            if sample_item in items:
                continue
            data['user_id'].append(user_id)
            data['item_id'].append(sample_item)
            data['label'].append(0)
            data['age'].append(data['age'][user_idx])
            data['gender'].append(data['gender'][user_idx])
            data['item_name'].append(data['item_name'][user_idx])
            data['item_catalog'].append(data['item_catalog'][user_idx])
            time_stamp=datetime.datetime.now()
            data['weekday'].append(data['weekday'][user_idx])
            data['hour'].append(data['hour'][user_idx])
            data['minute'].append(time_stamp.minute)
            data['second'].append(time_stamp.second)
            n+=1
            if n>ratio*len(user_item[user_id]):
                break
        
        data=pd.DataFrame(data)
        idx=np.random.permutation(len(data))
        data=data.iloc[idx,:].reset_index(drop=True)

        label=data.pop('label')
        data=self.apply_tensorflow_prerocessing(data)
        data=tf.data.Dataset.from_tensor_slices((dict(data),label))
        return data

def load_pb():
    graph_def=tf.compat.v1.GraphDef()
    loaded=graph_def.ParseFromString(open(r'.\deeplearning\DeepFM\DeepFM.pb','rb').read())
    deepfm_func=wrap_frozen_graph(graph_def,inputs=['x:0','x_1:0','x_2:0','x_3:0','x_4:0','x_5:0','x_6:0',
                                  'x_7:0','x_8:0'],outputs='Identity:0')
    return deepfm_func

def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
      tf.nest.map_structure(import_graph.as_graph_element, inputs),
      tf.nest.map_structure(import_graph.as_graph_element, outputs))            


def roc_auc(y_true,y_pred):
    return tf.py_function(roc_auc_score,(y_true,y_pred),tf.float16)

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
        for i in range(5*ratio):
            sample_item=item_pool[np.random.randint(0,len(item_pool)-1)]
            epsilon=np.random.rand(1)[0]
            if sample_item in user_log[user_field[0]] and epsilon<0.9:
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

def load_full_log(train_from,train_to,test_from=None,test_to=None,is_test=False,records=False):
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        if not records:
            sql1=" select t1.agent_id as user_id,t1.oper_time as time_stamp,cast(ifnull(qb.age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender,t1.oper_obj as item_id,IFNULL(t2.catalog,0) as item_catalog \
                    from chsell_oper_data as t1 left JOIN chsell_product as t2 on t1.oper_obj= t2.id left JOIN chsell_agent as t3 on t1.agent_id = t3.id left JOIN bigdata.chsell_quick_bi as qb on t1.agent_id=qb.channel_id \
                    where agent_id is not NULL and oper_time between '" +train_from+ "'and '" +train_to+ "' and oper_type='HOME_PRODUCT' "
        else:
            sql1=" select t1.agent_id as user_id,t1.oper_time as time_stamp,cast(ifnull(qb.age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender,t1.oper_obj as item_id,IFNULL(t2.catalog,0) as item_catalog \
                    from chsell_oper_data as t1 left JOIN chsell_product as t2 on t1.oper_obj= t2.id left JOIN chsell_agent as t3 on t1.agent_id = t3.id left JOIN bigdata.chsell_quick_bi as qb on t1.agent_id=qb.channel_id \
                    where agent_id is not NULL and oper_time = '" +train_from+"' and oper_type='HOME_PRODUCT'"+" limit 0," + str(records)
        engine=sqlalchemy.create_engine(engine_str) 
        dftrain=pd.read_sql(sql=sql1,con=engine)
        if is_test:
            if test_from!=test_to:
                sql2= " select t1.agent_id as user_id,t1.oper_time as time_stamp,cast(ifnull(qb.age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender,t1.oper_obj as item_id,IFNULL(t2.catalog,0) as item_catalog \
                from chsell_oper_data as t1 left JOIN chsell_product as t2 on t1.oper_obj= t2.id left JOIN chsell_agent as t3 on t1.agent_id = t3.id left JOIN bigdata.chsell_quick_bi as qb on t1.agent_id=qb.channel_id \
                where agent_id is not NULL and oper_time between '" +test_from+ "'and '" +test_to+ "' and oper_type='HOME_PRODUCT' "
            else:
                sql2= " select t1.agent_id as user_id,t1.oper_time as time_stamp,cast(ifnull(qb.age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender,t1.oper_obj as item_id,IFNULL(t2.catalog,0) as item_catalog \
                from chsell_oper_data as t1 left JOIN chsell_product as t2 on t1.oper_obj= t2.id left JOIN chsell_agent as t3 on t1.agent_id = t3.id left JOIN bigdata.chsell_quick_bi as qb on t1.agent_id=qb.channel_id \
                where t1.agent_id is not NULL and oper_time= '" +test_from+"' and oper_type='HOME_PRODUCT' "
            dftest=pd.read_sql(sql=sql2,con=engine)
            return dftrain,dftest
        else:
                return dftrain
def load_user_log(user_id,train_from,records=10):
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql1= " select t1.agent_id as user_id,t1.oper_time as time_stamp,cast(ifnull(qb.age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender,t1.oper_obj as item_id,IFNULL(t2.catalog,0) as item_catalog \
        from chsell_oper_data as t1 left JOIN chsell_product as t2 on t1.oper_obj= t2.id left JOIN chsell_agent as t3 on t1.agent_id = t3.id left JOIN bigdata.chsell_quick_bi as qb on t1.agent_id=qb.channel_id\
        where agent_id is not NULL and date_format(oper_time,'%%Y/%%m/%%d') = DATE('" +train_from + "') and oper_type='HOME_PRODUCT' and t1.agent_id= '"+user_id+"' limit 0,"+str(records)
        engine=sqlalchemy.create_engine(engine_str)
        dftrain=pd.read_sql(sql=sql1,con=engine)
        return dftrain
def load_item_pool(train_from,train_to,as_list=True):
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        engine=sqlalchemy.create_engine(engine_str) 
        sql="select oper_obj as item_id from chsell_oper_data where agent_id is not NULL and oper_time between '" +train_from+ "'and'" +train_to+ "' and oper_type='HOME_PRODUCT'"
        item_pool=pd.read_sql(sql=sql,con=engine)
        if as_list:
            item_pool=list(item_pool['item_id'])
        return item_pool
def load_user_id():
    engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/bigdata'
    engine=sqlalchemy.create_engine(engine_str)
    sql="select distinct channel_id as user_id from chsell_quick_bi"
    data=pd.read_sql(sql=sql,con=engine)
    return data
def insert_data(data):
    engine_str='mysql+pymysql://ksr-app:Ka-user@2506@keyike.ltd/kyk_ml'
    engine=sqlalchemy.create_engine(engine_str)
    data.to_sql('chsell_quick_bi_Product_recommendation',if_exists='replace',con=engine,index=False,chunksize=256000)
'''
class FactorizationMachine(tf.keras.Model):
    def __init__(self,output_dim=64,linear_reg=0.01,bias_reg=0.01):
        super(FactorizationMachine,self).__init__()
        self.output_dim=output_dim
        self.linear_reg=linear_reg
        self.bias_reg=bias_reg
    def build(self,input_shape):
        self.linear=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(self.linear_reg),
                                          bias_regularizer=tf.keras.regularizers.l2(self.bias_reg),name='linear')
        self.crosslayer=crosslayer(self.output_dim)
        self.logit=tf.keras.layers.Add()
        self.pred=tf.keras.layers.Activation(activation='sigmoid')
    def call(self,x):
        linear=self.linear(x)
        cross=self.crosslayer(x)
        logit=self.logit([linear,cross])
        pred=self.pred(logit)
        return pred
'''
'''
class crosslayer(tf.keras.layers.Layer):
    def __init__(self,output_dim=64,**kwargs):
        super(crosslayer,self).__init__(**kwargs)
        self.output_dim=output_dim
    def build(self,input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                     shape=(input_shape[-1],self.output_dim),
                                     initializer=tf.keras.initializers.glorot_normal(),
                                     trainable=True)
        super(crosslayer, self).build(input_shape)
    def call(self,x):
        a=tf.keras.backend.pow(tf.keras.backend.dot(x,self.kernel),2)
        b=tf.keras.backend.dot(tf.keras.backend.pow(x,2),tf.keras.backend.pow(self.kernel,2))
        return 0.5*tf.keras.backend.mean(a-b,1,keepdims=True) 
    def get_config(self):
        config=super(crosslayer,self).get_config()
        config.update({'output_dim':self.output_dim})
        return config
'''
'''
class FM(tf.keras.layers.Layer):
    def __init__(self,embedding_dim=64,linear_reg=0.01,bias_reg=0.01,**kwargs):
        super(FM,self).__init__(**kwargs)
        self.embedding_dim=embedding_dim
        self.linear_reg=linear_reg
        self.bias_reg=bias_reg
    def build(self,input_shape):
        self.linear=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(self.linear_reg),
                                          bias_regularizer=tf.keras.regularizers.l2(self.bias_reg),name='linear')
        self.crosslayer=crosslayer(self.embedding_dim)
        self.logit=tf.keras.layers.Add()
        super(FM, self).build(input_shape)
    def call(self,inputs):
        #sparse_feature,embedding=inputs
        linear=self.linear(inputs)
        cross=self.crosslayer(inputs)
        logit=self.logit([linear,cross])
        return logit
    def get_config(self):
        config=super(FM,self).get_config()
        config.update({'embedding_dim':self.embedding_dim,'linear_reg':self.linear_reg,'bias_reg':self.bias_reg})
        return config
'''