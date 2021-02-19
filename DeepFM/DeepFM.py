import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import os
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
        self.dense1=tf.keras.layers.Dense(256,kernel_regularizer=tf.keras.regularizers.l2(0.02),
                                            bias_regularizer=tf.keras.regularizers.l2(0.02),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer1')
        self.dropout1=tf.keras.layers.Dropout(0.4)
        self.dense2=tf.keras.layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(0.02),
                                            bias_regularizer=tf.keras.regularizers.l2(0.02),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer2')
        self.dropout2=tf.keras.layers.Dropout(0.8)
        self.dense3=tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(0.02),
                                            bias_regularizer=tf.keras.regularizers.l2(0.02),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer3')
        self.dropout3=tf.keras.layers.Dropout(0.7)
        self.dense4=tf.keras.layers.Dense(32,kernel_regularizer=tf.keras.regularizers.l2(0.02),
                                            bias_regularizer=tf.keras.regularizers.l2(0.02),
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
class DeepFatorizationMachine(tf.keras.Model):
    def __init__(self,embedding_dim=64,num_bins=512):
        super(DeepFatorizationMachine,self).__init__()
        self.embedding_dim=embedding_dim
        self.num_bins=num_bins

    def build(self,input_shape):
        #embedding for sparse/categorical features


        #
        self.user_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=self.num_bins)
        self.user_embedding=tf.keras.layers.Embedding(input_dim=self.num_bins,output_dim=self.embedding_dim,name='user_id_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        #
        self.item_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=self.num_bins)
        self.item_embedding=tf.keras.layers.Embedding(input_dim=self.num_bins,output_dim=self.embedding_dim,name='item_id_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        #
        '''
        self.region_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=self.num_bins)
        self.region_embedding=tf.keras.layers.Embedding(input_dim=self.num_bins,output_dim=self.embedding_dim,name='region_id_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        #
        self.city_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=self.num_bins)
        self.city_embedding=tf.keras.layers.Embedding(input_dim=self.num_bins,output_dim=self.embedding_dim,name='city_id_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
        '''
        #
        #self.catalog_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=self.num_bins)
        self.catalog_embedding=tf.keras.layers.Embedding(input_dim=self.num_bins,output_dim=self.embedding_dim,name='catalog_embedding',
        embeddings_initializer=tf.keras.initializers.glorot_normal())
      
        #flatten embedding
        self.flatten=tf.keras.layers.Flatten()

        self.FM=FM(self.embedding_dim*2)
        self.DNN=DNN()
        self.add=tf.keras.layers.Add()
        self.pred=tf.keras.layers.Activation(activation='sigmoid')

    def call(self,inputs,training=None):
        
        user_id=inputs['user_id']
        item_id=inputs['item_id']
        #region_id=inputs['region_id']
        #city_id=inputs['city_id']
        item_catalog=inputs['item_catalog']
        #sparse=tf.concat([user_id,item_id,region,city,item_catalog],axis=1)

        if training:
            self.user_hash.adapt(user_id)
            self.item_hash.adapt(item_id)
            #self.region_hash.adapt(region_id)
            #self.city_hash.adapt(city_id)
            #self.catalog_hash.adapt(item_catalog)
        #embedding layers for user/item related sparse features

        #user field
        user_embedding=self.flatten(self.user_embedding(self.user_hash(user_id)))
        #region_embedding=self.flatten(self.region_embedding(self.region_hash(region_id)))
        #city_embedding=self.flatten(self.city_embedding(self.city_hash(city_id)))
        #item field
        item_embedding=self.flatten(self.item_embedding(self.item_hash(item_id)))
        catalog_embedding=self.flatten(self.catalog_embedding(item_catalog))

        #concat embeddings
        embeddings=tf.concat([user_embedding,item_embedding,catalog_embedding],axis=1)

        #FM
        fm_pred=self.FM(embeddings)
        #DNN
        dnn_pred=self.DNN(embeddings)
        #combine the outputs from the two layers
        add=self.add([fm_pred,dnn_pred])
        #map the output by sigmoid function
        pred=self.pred(add)
        return pred

def roc_auc(y_true,y_pred):
    return tf.py_function(roc_auc_score,(y_true,y_pred),tf.float16)
def get_prediction(model,dataset,feature_cols=['user_id','item_id','item_catalog']):
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
    y_score=model.predict(X_test)
    df=pd.DataFrame(X_test)
    df['score']=y_score
    return y_true,y_score,df

def feeling_lucky(model,user_ids,topK=5,feature_cols=['user_id','item_id','item_catalog']):

    delta=datetime.timedelta(days=14)
    now=datetime.datetime.now().strftime(format='%Y/%m/%d')
    past=(datetime.datetime.now()-delta).strftime(format='%Y/%m/%d')
    #use all items as retrieval candidates
    candidates=prep.load_item_pool(past,now,as_list=False)
    candidates=candidates.drop_duplicates()

    engine_str0='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
    sql0="select id as item_id,catalog as item_catalog ,name as item_name from chsell_product"
    item_type=pd.read_sql(sql=sql0,con=engine_str0).fillna(0)
    item_data=candidates.merge(item_type,how='inner',on='item_id')
    engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/bigdata'
    sql="select channel_id as user_id,ifnull(region_id,'unk') as region_id,ifnull(city_id,'unk') as city_id \
        from chsell_quick_bi where channel_id in ("+','.join(user_ids)+")"
    user_data=pd.read_sql(sql=sql,con=engine_str)
    tmp=[]
    for user_id in user_ids:
        df=pd.DataFrame()
        df['item_id']=item_data['item_id']
        df['user_id']=user_id.replace("'",'')
        tmp.append(df)
    df=pd.concat(tmp,axis=0)
    df=df.merge(user_data,how='inner',on='user_id')
    df=df.merge(item_data,how='inner',on='item_id').loc[:,feature_cols]
    X=df.to_dict(orient='list')
    for col,values in X.items():
        X[col]=np.array(values)
    pred=model.predict(X)
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





'''Single Factorization Machine
model=fm.FactorizationMachine(output_dim=128)   
if os.path.exists(path_model+'factors.h5'):
    print('loading model.\n')
    model.predict(train_set)
    model.load_weights(path_model+'factors.h5')
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.01),metrics=[tf.keras.metrics.BinaryAccuracy(),fm.roc_auc,tf.keras.metrics.Recall()])
    model.evaluate(test_set)
    model.fit(train_set,epochs=5)
else:
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.01),metrics=[tf.keras.metrics.BinaryAccuracy(),fm.roc_auc,tf.keras.metrics.Recall()])
    model.fit(train_set,epochs=25)
    model.evaluate(test_set)
    model.summary()
    model.save_weights(path_model+'factors.h5')
'''
'''Deep Factorization Machine
'''