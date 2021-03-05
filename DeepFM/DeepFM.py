import json
import os.path

import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import warnings
import sqlalchemy
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
    def __init__(self,embedding_dim=64,num_bins=31268,**kwargs):
        self.MAX_ID=tf.Variable(initial_value=0,trainable=False)
        super(DeepFatorizationMachine,self).__init__(**kwargs)
        self.embedding_dim=embedding_dim
        self.num_bins=num_bins
        self.user_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=self.num_bins)
        self.gender_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=4)
        self.item_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=500)
        self.setting=None
    

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
       
        fm_pred=self.FM(embeddings)
       
        dnn_pred=self.DNN(embeddings)
        
        add=self.add([fm_pred,dnn_pred])
        
        outputs=self.pred(add)
        return outputs
    def get_config(self):
        config=super(DeepFatorizationMachine,self).get_config()
        config.update({'embedding_dim':self.embedding_dim,'nim_bins':self.num_bins})
        return config

    
    #加载模型     
    def load_model_weights(self,path=r'.\deeplearning\DeepFM\DeepFM.h5'):
        print("xxxxxx")
        #假数据以启动模型
        data={'user_id':np.array([1,1]),'item_id':np.array([1,1]),
            'gender':np.array([1,1]),'age':np.array([1,1]),'item_catalog':np.array([1,1]),'hour':np.array([1,1]),
            'minute':np.array([1,1]),'second':np.array([1,1]),'weekday':np.array([1,1])}
        self(data)
        self.load_weights(path)
        setting_file = path.replace('.h5', '.setting')
        if os.path.isfile(setting_file):
            with open(setting_file, 'r') as f:
                self.setting = json.load(f)
                print(self.setting)
                return self.setting['MAX_ID']
    #保存模型
    def save_model_weights(self, path=r'.\deeplearning\DeepFM\DeepFM.h5', new_max=None):
        self.save_weights(path)
        if self.setting is None:
            self.setting = json.loads(json.dumps({'MAX_ID': new_max, 'UPDATE_TIME': str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 'VERSION': '0.1'}))
        else:
            self.setting['MAX_ID'] = new_max
            self.setting['UPDATE_TIME'] = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        setting_file = path.replace('.h5', '.setting')
        with open(setting_file, 'w') as f:
            json.dump(self.setting, f)

        # 训练函数

    def retrain(self, data, epochs=10, batch_size=32, learning_rate=0.01):
        # 增加负样本
        data_set = self.sampling(data)
        data_set = data_set.shuffle(len(data_set)).batch(batch_size)
        # 编译并训练模型
        self.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate))
        self.fit(data_set, epochs=epochs)

    #预测函数
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
    #增量采样
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

    #tensorflow 预处理
    def apply_tensorflow_prerocessing(self,data):
        df=data.copy()
        df['user_id']=self.user_hash(df['user_id'].values).numpy()
        df['gender']=self.gender_hash(df['gender'].values).numpy()
        df['item_id']=self.item_hash(df['item_id'].values).numpy()
        return df
    #加载模型     
    # def load_model_weights(self,path=r'.\deeplearning\DeepFM\DeepFM.h5'):
    #     #假数据以启动模型
    #     data={'user_id':np.array([1,1]),'item_id':np.array([1,1]),
    #         'gender':np.array([1,1]),'age':np.array([1,1]),'item_catalog':np.array([1,1]),'hour':np.array([1,1]),
    #         'minute':np.array([1,1]),'second':np.array([1,1]),'weekday':np.array([1,1])}
    #     self(data)
    #     self.load_weights(path)
#保存模型
    # def save_model_weights(self,path=r'.\deeplearning\DeepFM\DeepFM.h5'):
    #     self.save_weights(path)




def load_item_pool(train_from,train_to,as_list=True):
        engine_str='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
        engine=sqlalchemy.create_engine(engine_str) 
        sql="select oper_obj as item_id from chsell_oper_data where agent_id is not NULL and oper_time between '" +train_from+ "'and'" +train_to+ "' and oper_type='HOME_PRODUCT'"
        item_pool=pd.read_sql(sql=sql,con=engine)
        if as_list:
            item_pool=list(item_pool['item_id'])
        return item_pool

def roc_auc(y_true,y_pred):
    return tf.py_function(roc_auc_score,(y_true,y_pred),tf.float16)



 