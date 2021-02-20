import numpy as np
import pandas as pd
import datetime
import time
import tensorflow as tf
import warnings
from sklearn.metrics import roc_auc_score
import preprocess as prep
warnings.filterwarnings('ignore')

class crosslayer(tf.keras.layers.Layer):
    def __init__(self,output_dim=64,**kwargs):
        super(crosslayer,self).__init__(**kwargs)
        self.output_dim=output_dim
    def build(self,input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                     shape=(input_shape[-1],self.output_dim),
                                     initializer=tf.keras.initializers.glorot_normal(),
                                     trainable=True)
    def call(self,x):
        a=tf.keras.backend.pow(tf.keras.backend.dot(x,self.kernel),2)
        b=tf.keras.backend.dot(tf.keras.backend.pow(x,2),tf.keras.backend.pow(self.kernel,2))
        return 0.5*tf.keras.backend.mean(a-b,1,keepdims=True) 
               
class FM(tf.keras.layers.Layer):
    def __init__(self,embedding_dim=64,linear_reg=0.01,bias_reg=0.01):
        super(FM,self).__init__()
        self.embedding_dim=embedding_dim
        self.linear_reg=linear_reg
        self.bias_reg=bias_reg
    def build(self,input_shape):
        self.linear=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(self.linear_reg),
                                          bias_regularizer=tf.keras.regularizers.l2(self.bias_reg),name='linear')
        self.crosslayer=crosslayer(self.embedding_dim)
        self.logit=tf.keras.layers.Add()
    def call(self,inputs):
        #sparse_feature,embedding=inputs
        linear=self.linear(inputs)
        cross=self.crosslayer(inputs)
        logit=self.logit([linear,cross])
        return logit
class DNN(tf.keras.layers.Layer):
    def __init__(self):
        super(DNN,self).__init__()
    def build(self,input_shape):
        
        self.dense1=tf.keras.layers.Dense(256,kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer1')
        self.dropout1=tf.keras.layers.Dropout(0.4)
       
        self.dense2=tf.keras.layers.Dense(256,kernel_regularizer=tf.keras.regularizers.l2(0.08),
                                            bias_regularizer=tf.keras.regularizers.l2(0.08),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer2')
        self.dropout2=tf.keras.layers.Dropout(0.8)
        self.dense3=tf.keras.layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(0.06),
                                            bias_regularizer=tf.keras.regularizers.l2(0.06),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer3')
        self.dropout3=tf.keras.layers.Dropout(0.7)
        self.dense4=tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(0.03),
                                            bias_regularizer=tf.keras.regularizers.l2(0.03),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer4')
        self.dropout4=tf.keras.layers.Dropout(0.3)
        self.dense5=tf.keras.layers.Dense(1,kernel_initializer=tf.keras.initializers.glorot_normal(),name='dnn_layer5')
        
    def call(self,inputs,training=None):
        
        dense1=self.dense1(inputs)
        if training:
            dense1=self.dropout1(dense1)
        
        dense2=self.dense2(dense1)
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
#definition of the deepFM model     
class DeepFatorizationMachine(tf.keras.Model):
    def __init__(self,embedding_dim=64,num_bins=12800):
        super(DeepFatorizationMachine,self).__init__()
        self.embedding_dim=embedding_dim
        self.num_bins=num_bins

    def build(self,input_shape):
        #embedding for sparse/categorical features
        #user_id(string)
        self.user_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=self.num_bins)
        self.user_embedding=tf.keras.layers.Embedding(input_dim=self.num_bins,output_dim=self.embedding_dim,name='user_id_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        #user_age(int)
        #self.age_bucket=tf.keras.layers.experimental.preprocessing.Discretization(bins=[0,20,25,28,32,35,40])
        self.age_embedding=tf.keras.layers.Embedding(input_dim=10,output_dim=self.embedding_dim,name='age_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        #user_gender(string)
        self.gender_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=4)
        self.gender_embedding=tf.keras.layers.Embedding(input_dim=5,output_dim=self.embedding_dim,name='gender_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        #time_stamp(int)
        #self.time_stamp_norm=tf.keras.layers.experimental.preprocessing.Normalization()
        #weekday(int)
        self.weekday_embedding=tf.keras.layers.Embedding(input_dim=8,output_dim=self.embedding_dim,name='weekday_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        #hms(hour/minute/second int)
        self.hour_embedding=tf.keras.layers.Embedding(input_dim=61,output_dim=self.embedding_dim,name='hour_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.minute_embedding=tf.keras.layers.Embedding(input_dim=61,output_dim=self.embedding_dim,name='minute_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        self.second_embedding=tf.keras.layers.Embedding(input_dim=61,output_dim=self.embedding_dim,name='second_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        #item_id(string)
        self.item_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=200)
        self.item_embedding=tf.keras.layers.Embedding(input_dim=200,output_dim=self.embedding_dim,name='item_id_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        #item_catalog(int)
        self.catalog_embedding=tf.keras.layers.Embedding(input_dim=50,output_dim=self.embedding_dim,name='catalog_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        #flatten embedding
        self.flatten=tf.keras.layers.Flatten()

        self.FM=FM(self.embedding_dim*9)
        self.DNN=DNN()
        self.add=tf.keras.layers.Add()
        self.pred=tf.keras.layers.Activation(activation='sigmoid')

    def call(self,inputs,training=None):
        
        user_id=inputs['user_id']
        age=inputs['age']
        gender=inputs['gender']
        weekday=inputs['weekday']
        hour=inputs['hour']
        minute=inputs['minute']
        second=inputs['second']
        item_id=inputs['item_id']
        item_catalog=inputs['item_catalog']
        if training:
            self.user_hash.adapt(user_id)
            self.item_hash.adapt(item_id)
            self.gender_hash.adapt(gender)
        #embedding layers for user/item related sparse features
        #user field
        user_embedding=self.flatten(self.user_embedding(self.user_hash(user_id)))
        age_embedding=self.flatten(self.age_embedding(age))
        gender_embedding=self.flatten(self.gender_embedding(self.gender_hash(gender)))
        #time embedding
        weekday_embedding=self.flatten(self.weekday_embedding(weekday))
        hour_embedding=self.flatten(self.hour_embedding(hour))
        minute_embedding=self.flatten(self.minute_embedding(minute))
        second_embedding=self.flatten(self.second_embedding(second))
        #item field
        item_embedding=self.flatten(self.item_embedding(self.item_hash(item_id)))
        catalog_embedding=self.flatten(self.catalog_embedding(item_catalog))
        #concat embeddings
        embeddings=tf.concat([user_embedding,age_embedding,gender_embedding,weekday_embedding,
                            hour_embedding,minute_embedding,second_embedding,item_embedding,catalog_embedding],axis=1)
        #FM
        fm_pred=self.FM(embeddings)
        #DNN
        dnn_pred=self.DNN(embeddings)
        #combine the outputs from the two layers
        add=self.add([fm_pred,dnn_pred])
        #map the output by sigmoid function
        pred=self.pred(add)
        return pred

    def feature_extraction(self,df,user_field,item_field):
        df=df.merge(user_field,how='inner',on='user_id')
        df=df.merge(item_field,how='inner',on='item_id')
        df['weekday']=df['time_stamp'].apply(lambda x: x.weekday())
        df['hour']=df['time_stamp'].apply(lambda x: x.hour)
        df['minute']=df['time_stamp'].apply(lambda x: x.minute)
        df['second']=df['time_stamp'].apply(lambda x: x.second)
        self.age_mean=df.loc[df['age']!=0,'age'].mean()
        df.loc[df['age']==0,'age']=self.age_mean
        df.drop(columns='time_stamp',inplace=True)

        return df
    #数据加载管道
    def data_pipeline(self,sampling_ratio=5,test=False,user_cols=['user_id','age','gender'],item_cols=['item_id','item_catalog']):
        #加载数据
        delta0=datetime.timedelta(days=2)
        delta=datetime.timedelta(days=16)
        train_to=(datetime.datetime.now()-delta0).strftime(format='%Y/%m/%d')
        train_from=(datetime.datetime.now()-delta).strftime(format='%Y/%m/%d')
        item_pool=prep.load_item_pool(train_from,train_to)
        if test:
            test_from=(datetime.datetime.now()-datetime.timedelta(days=1)).strftime(format='%Y/%m/%d')
            test_to=(datetime.datetime.now()).strftime(format='%Y/%m/%d')
            data,test_data=prep.load_full_log(train_from,train_to,test_from,test_to,is_test=test)
        else:   
            data=prep.load_full_log(train_from,train_to,is_test=test)

        #tensorflow preprocessing layers here
        self.age_bucket=tf.keras.layers.experimental.preprocessing.Discretization(bins=[0,20,25,28,32,35])
        
        df=data.copy()
        train_dict=dict()
        df=NegativeSampling(df,item_pool,sampling_ratio)
        user_field=data.loc[:,user_cols].drop_duplicates()
        item_field=data.loc[:,item_cols].drop_duplicates()

        df=self.feature_extraction(df,user_field,item_field)

        self.age_bucket.adapt(df['age'].values)
        df['age']=self.age_bucket(df['age'].values).numpy()
        

        if test:
            test_dict=dict()
            df_test=test_data.copy()
            df_test=NegativeSampling(df_test,item_pool,1)
            user_field=test_data.loc[:,user_cols].drop_duplicates()
            item_field=test_data.loc[:,item_cols].drop_duplicates()

            df_test=self.feature_extraction(df_test,user_field,item_field)

            df_test['age']=self.age_bucket(df_test['age'].values).numpy()
            
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
    def feeling_lucky(self,user_ids,topK=5):
        delta=datetime.timedelta(days=28)
        now=datetime.datetime.now().strftime(format='%Y/%m/%d')
        past=(datetime.datetime.now()-delta).strftime(format='%Y/%m/%d')
        #use all items as retrieval candidates
        candidates=prep.load_item_pool(past,now,as_list=False)
        candidates=candidates.drop_duplicates()
        #get user/item real time info
        engine_str0='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        sql0="select id as item_id,catalog as item_catalog ,name as item_name from chsell_product"
        item_type=pd.read_sql(sql=sql0,con=engine_str0).fillna(0)
        item_field=candidates.merge(item_type,how='inner',on='item_id')

        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/bigdata'
        sql="select channel_id as user_id,cast(ifnull(age,'unk') as SIGNED) as age,ifnull(Gender,'unk') as gender\
            from chsell_quick_bi where channel_id in ("+','.join(user_ids)+")"
        user_field=pd.read_sql(sql=sql,con=engine_str)
        tmp=[]
        for user_id in user_ids:
            df=pd.DataFrame()
            df['item_id']=item_field['item_id']
            df['user_id']=user_id.replace("'",'')
            tmp.append(df)
        df=pd.concat(tmp,axis=0)

        df['time_stamp']=datetime.datetime.now()
        df=self.feature_extraction(df,user_field,item_field)

        df['age']=self.age_bucket(df['age'].values).numpy()
        X=df.to_dict(orient='list')
        for col,values in X.items():
            X[col]=np.array(values)
        pred=self.predict(X)
        X=pd.DataFrame(X)


        X['score']=pred
        X['rank']=X.groupby(['user_id'])['score'].rank(method='first',ascending=False).sort_values()
        X=X.sort_values(by=['user_id','rank']).reset_index(drop=True)
        X=X.loc[:,['user_id','item_id','rank']]
        X=X.merge(item_type,how='inner',on='item_id').loc[:,['user_id','item_name','rank']]
        recmd_list=X[X['rank']<=topK].set_index(['user_id','rank']).unstack(-1).reset_index()
        #.droplevel(1).reset_index().set_index(['user_id','rank']).unstack(-1).reset_index()
        recmd_list.columns=[int(col) if isinstance(col,int) else col for col in recmd_list.columns.droplevel(0)]

        return recmd_list
    def hot_start(self):
        train_set=self.data_pipeline(sampling_ratio=0)
        train_set=train_set.shuffle(100).batch(64)
        self.predict(train_set)
    
def roc_auc(y_true,y_pred):
    return tf.py_function(roc_auc_score,(y_true,y_pred),tf.float16)
def get_prediction(model,dataset):
    if not isinstance(dataset,tf.data.Dataset):
        print('Input must be a tensorflow dataset: tf.data.Dataset object.\n')
        return 0
    user_id=[]
    item_id=[]
    item_catalog=[]
    y_true=[]
    for i in dataset.as_numpy_iterator():
        user_id.append(i[0]['user_id'])
        item_id.append(i[0]['item_id'])
        item_catalog.append(i[0]['item_catalog'])
        y_true.append(i[1])
    user_id=np.concatenate(user_id)
    item_id=np.concatenate(item_id)
    item_catalog=np.concatenate(item_catalog)
    X_test={'user_id':user_id,'item_id':item_id,'item_catalog':item_catalog}
    y_true=np.hstack(y_true)
    y_score=model(X_test)
    df=pd.DataFrame(X_test)
    df['score']=y_score
    return y_true,y_score,df

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