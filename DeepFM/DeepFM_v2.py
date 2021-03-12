import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import datetime
from sklearn.metrics import roc_auc_score
import json
import joblib
import os
warnings.filterwarnings('ignore')
user_tokens=60000
item_tokens=300
item_tag_vocab=300
sequence_length=10
item_catalog_tokens=11
user_fields=['user_id','member_type','user_type']
item_fields=['item_id','item_catalog','item_tags']
#model parameters
embedding_dim=12
hidden_units=12
l2_regularization=3.2
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
def Item2vec(X):
    #string_col=['item_name','tag','interestb4pr','flexible_return','tax_rating']
    #int_col=['term','age_lower','age_upper','holder_identity','establish_yr','fapiao']
    #cont_col=['credit','rate_lower','rate_upper','fapiao_income']
    #string col
    item_name_vectorize_layer=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',
    output_sequence_length=1,name='item_name_vectorize_layer',vocabulary=np.unique(X['item_id']))
    item_name_embedding_layer=tf.keras.layers.Embedding(len(np.unique(X['item_id']))+2,embedding_dim,name='item_name_embedding_layer')
    tag_vectorize_layer=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',
    output_sequence_length=2,name='tag_vectorize_layer')
    tag_vectorize_layer.adapt(X['tag'])
    tag_embedding_layer=tf.keras.layers.Embedding(tag_tokens,embedding_dim,name='tag_embedding_layer')

    interestb4pr_vectorize_layer=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',
    name='interestb4pr_vectorize_layer',vocabulary=np.unique(X['interestb4pr']),output_sequence_length=1)
    interestb4pr_embedding_layer=tf.keras.layers.Embedding(len(np.unique(X['interestb4pr']))+2,embedding_dim,name='interestb4pr_embedding_layer')
    
    flexible_return_vectorize_layer=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',
    name='flexible_return_vectorize_layer',vocabulary=np.unique(X['flexible_return']),output_sequence_length=1)
    flexible_return_embedding_layer=tf.keras.layers.Embedding(len(np.unique(X['flexible_return']))+2,embedding_dim,name='flexible_return_embedding_layer')

    tax_rating_vectorize_layer=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',
    output_sequence_length=6,name='tax_rating_vectorize_layer')
    tax_rating_embedding_layer=tf.keras.layers.Embedding(len(np.unique(X['tax_rating']))+2,embedding_dim,name='tax_rating_embedding_layer')
    tax_rating_vectorize_layer.adapt(X['tax_rating'])
    
    #integer col
    term_vectorize_layer=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['term']),name='term_vectorize_layer',mask_value=None)
    term_embedding_layer=tf.keras.layers.Embedding(len(np.unique(X['term']))+2,embedding_dim,name='term_embedding_layer')

    age_lower_vectorize_layer=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['age_lower']),name='age_lower_vectorize_layer',mask_value=None)
    age_lower_embedding_layer=tf.keras.layers.Embedding(len(np.unique(X['age_lower']))+2,embedding_dim,name='age_lower_embedding_layer')

    age_upper_vectorize_layer=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['age_upper']),name='age_upper_vectorize_layer',mask_value=None)
    age_upper_embedding_layer=tf.keras.layers.Embedding(len(np.unique(X['age_upper']))+2,embedding_dim,name='age_upper_embedding_layer')

    holder_identity_vectorize_layer=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['holder_identity']),name='holder_identity_vectorize_layer',mask_value=None)
    holder_identity_embedding_layer=tf.keras.layers.Embedding(len(np.unique(X['holder_identity']))+2,embedding_dim,name='holder_identity_embedding_layer')

    establish_yr_vectorize_layer=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['establish_yr']),name='establish_yr_vectorize_layer',mask_value=None)
    establish_yr_embedding_layer=tf.keras.layers.Embedding(len(np.unique(X['establish_yr']))+2,embedding_dim,name='establish_yr_embedding_layer')

    fapiao_vectorize_layer=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['fapiao']),name='fapiao_vectorize_layer',mask_value=None)
    fapiao_embedding_layer=tf.keras.layers.Embedding(len(np.unique(X['fapiao']))+2,embedding_dim,name='fapiao_embedding_layer')
    
    #string input
    item_name=tf.keras.layers.Input(shape=(1,),dtype=tf.string,name='item_id')
    tag=tf.keras.layers.Input(shape=(1,),dtype=tf.string,name='tag')
    interestb4pr=tf.keras.layers.Input(shape=(1,),dtype=tf.string,name='interestb4pr')
    flexible_return=tf.keras.layers.Input(shape=(1,),dtype=tf.string,name='flexible_return')
    tax_rating=tf.keras.layers.Input(shape=(1,),dtype=tf.string,name='tax_rating')
    
    #integer input
    term=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='term')
    age_lower=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='age_lower')
    age_upper=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='age_upper')
    holder_identity=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='holder_identity')
    establish_yr=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='establish_yr')
    fapiao=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='fapiao')
    #continuous input
    credit=tf.keras.layers.Input(shape=(1,),dtype=tf.float32,name='credit')
    rate_lower=tf.keras.layers.Input(shape=(1,),dtype=tf.float32,name='rate_lower')
    rate_upper=tf.keras.layers.Input(shape=(1,),dtype=tf.float32,name='rate_upper')
    fapiao_income=tf.keras.layers.Input(shape=(1,),dtype=tf.float32,name='fapiao_income')
    #embeddings
    #string embedding
    #item
    item_name_embedding=item_name_embedding_layer(item_name_vectorize_layer(item_name))
    tag_embedding=tag_embedding_layer(tag_vectorize_layer(tag))
    interestb4pr_embedding=interestb4pr_embedding_layer(interestb4pr_vectorize_layer(interestb4pr))
    flexible_return_embedding=flexible_return_embedding_layer(flexible_return_vectorize_layer(flexible_return))
    tax_rating_embedding=tax_rating_embedding_layer(tax_rating_vectorize_layer(tax_rating))
    #integer embedding
    term_embedding=term_embedding_layer(term_vectorize_layer(term))
    age_lower_embedding=age_lower_embedding_layer(age_lower_vectorize_layer(age_lower))
    age_upper_embedding=age_upper_embedding_layer(age_upper_vectorize_layer(age_upper))
    holder_identity_embedding=holder_identity_embedding_layer(holder_identity_vectorize_layer(holder_identity))
    establish_yr_embedding=establish_yr_embedding_layer(establish_yr_vectorize_layer(establish_yr))
    fapiao_embedding=fapiao_embedding_layer(fapiao_vectorize_layer(fapiao))

    item_vector=tf.keras.layers.GlobalAveragePooling1D()(tf.concat([item_name_embedding,tag_embedding,interestb4pr_embedding,flexible_return_embedding,
    flexible_return_embedding,tax_rating_embedding,term_embedding,age_lower_embedding,age_upper_embedding,holder_identity_embedding,establish_yr_embedding,fapiao_embedding],axis=1))

    vector=tf.keras.layers.concatenate([item_vector,credit,rate_lower,rate_upper,fapiao_income],axis=1)
    dnn1=tf.keras.layers.Dense(128,activation='relu')(vector)
    dnn2=tf.keras.layers.Dense(128,activation='relu')(dnn1)
    dnn3=tf.keras.layers.Dense(128,activation='relu')(dnn2)
    dnn4=tf.keras.layers.Dense(128,activation='relu')(dnn3)
    pred=tf.keras.layers.Dense(1,activation='sigmoid')(dnn4)
    model=tf.keras.Model(inputs=[item_name,tag,interestb4pr,flexible_return,tax_rating,term,age_lower,age_upper,holder_identity,
                                establish_yr,fapiao,credit,rate_lower,rate_upper,fapiao_income],outputs=pred)
    return model

def get_item_vector(item2vec,item_features,item_id,weights={'age_lower':2,'age_upper':2,'tax_rating':3,
                    'interestb4pr':1.5,'flexible_return':1.5},cold_start=None):
    cat_col=['item_name','tag','interestb4pr','flexible_return','tax_rating','term','age_lower','age_upper','holder_identity','establish_yr','fapiao']
    str_col=['item_name','tag','interestb4pr','flexible_return','tax_rating']
    int_col=['term','age_lower','age_upper','holder_identity','establish_yr','fapiao']
    num_col=['credit','rate_lower','rate_upper','fapiao_income']
    for col in int_col:
        item_features[col]=item_features[col].astype('int')
    vector_dict={}
    n=0
    for col in cat_col+num_col:
        if col not in weights.keys():
            weights.setdefault(col,1)
        if col not in num_col:
            n+=weights[col]
        if col=='item_name':
            item_feature=item_features.loc[item_features['item_id']==item_id,'item_id'].values
        else:
            item_feature=item_features.loc[item_features['item_id']==item_id,col].values
        if col in str_col:
            vector_dict.setdefault(col+'_embedding_layer',0)
            vector_dict[col+'_embedding_layer']=item2vec.get_layer(col+'_embedding_layer')(item2vec.get_layer(col+'_vectorize_layer')(item_feature)).numpy()[0,:,:]
        elif col in int_col:
            vector_dict.setdefault(col+'_embedding_layer',0)
            vector_dict[col+'_embedding_layer']=item2vec.get_layer(col+'_embedding_layer')(item2vec.get_layer(col+'_vectorize_layer')(item_feature)).numpy()
        else:
            vector_dict.setdefault(col,0)
            vector_dict[col]=item_feature.reshape(1,-1)
    for key in weights.keys():
        if key in cat_col:
            vector_dict[key+'_embedding_layer']=weights[key]*vector_dict[key+'_embedding_layer']
        else:
            vector_dict[key]=weights[key]*vector_dict[key]
    vector_array=[]
    for col in cat_col:
        vector_array.append(vector_dict[col+'_embedding_layer'])
    item_vector=np.concatenate(vector_array,axis=0)
    pooling_avg=(item_vector.sum(axis=0).reshape(1,-1))/n
    for col in num_col:
        pooling_avg=np.concatenate([pooling_avg,vector_dict[col]],axis=1)
    return pooling_avg
def get_similar_items(item2vec,item_features,item_id,cold_start=None):
    target_vec=get_item_vector(item2vec,item_features,item_id)
    target_vec_unify=target_vec/np.linalg.norm(target_vec)
    sim_dict={}
    sim_dict.setdefault('target',item_id)
    sim_dict.setdefault('item_list',0)
    cos_dict={}
    for item in item_features['item_id'].unique():
        if item==item_id:
            continue
        item_vec=get_item_vector(item2vec,item_features,item)
        item_vec_unify=item_vec/np.linalg.norm(item_vec)
        inner_product=np.round(np.dot(target_vec_unify,item_vec_unify.T),3)
        cos_dict.setdefault(item,0)
        cos_dict[item]=inner_product[0,0]
    cos_dict=dict(sorted(cos_dict.items(),key=lambda x: x[1],reverse=True))
    sim_dict['item_list']=cos_dict
    return sim_dict
def Prep_model():
    #Input layers
    #user field
    user_id_input=tf.keras.Input(shape=(1,),dtype=tf.string,name='user_id')
    member_type_input=tf.keras.Input(shape=(1,),dtype=tf.string,name='member_type')
    user_type_input=tf.keras.Input(shape=(1,),dtype=tf.string,name='user_type')
    #item_field
    item_id_input=tf.keras.Input(shape=(1,),dtype=tf.string,name='item_id')
    item_catalog_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='item_catalog')
    item_tag_input=tf.keras.Input(shape=(1,),dtype=tf.string,name='item_tag')
    #vectorize
    user_id_vectorize=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',output_sequence_length=1,
    name='user_id_vectorize')(user_id_input)
    member_type_vectorize=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',output_sequence_length=1,
    name='member_type_vectorize')(member_type_input)
    user_type_vectorize=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',output_sequence_length=1
    ,name='user_type_vectorize')(user_type_input)

    item_id_vectorize=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',output_sequence_length=1,
    name='item_id_vectorize')(item_id_input)
    item_tag_vectorize=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',output_sequence_length=sequence_length,
    name='item_tag_vectorize')(item_tag_input)
    item_catalog_intlookup=tf.keras.layers.experimental.preprocessing.IntegerLookup(name='item_catalog_intlookup')(item_catalog_input)

    model=tf.keras.Model(inputs=[user_id_input,user_type_input,member_type_input,item_id_input,item_catalog_input,item_tag_input],
                        outputs=[user_id_vectorize,user_type_vectorize,member_type_vectorize,item_id_vectorize,item_catalog_intlookup,item_tag_vectorize])
    return model
def preprocess(prep_model,data,features=['user_id','user_type','member_type','item_id','item_catalog','item_tag']):
    data=prep_model(data)
    data_dict={}
    i=0
    for feat in features:
        data_dict.setdefault(feat,0)
        data_dict[feat]=data[i].numpy()
        i+=1
    return data_dict
def DeepFM():
    #Input layers
    #user field
    user_id_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='user_id')
    member_type_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='member_type')
    user_type_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='user_type')
    #item_field
    item_id_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='item_id')
    item_catalog_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='item_catalog')
    item_tag_input=tf.keras.Input(shape=(sequence_length,),dtype=tf.int32,name='item_tag')

    #Embedding
    user_id_embedding=tf.keras.layers.Embedding(user_tokens,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='user_id_embedding')(user_id_input)
    member_type_embedding=tf.keras.layers.Embedding(5,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='member_type_embedding')(member_type_input)
    user_type_embedding=tf.keras.layers.Embedding(4,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='user_type_embedding')(user_type_input)
    item_id_embedding=tf.keras.layers.Embedding(item_tokens,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_id_embedding')(item_id_input)
    item_tag_embedding=tf.keras.layers.Embedding(item_tag_vocab+1,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_tag_embedding')(item_tag_input)
    item_catalog_embedding=tf.keras.layers.Embedding(item_catalog_tokens+1,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_catalog_embedding')(item_catalog_input)
    dense_features=tf.keras.layers.concatenate([user_id_embedding,member_type_embedding,user_type_embedding,item_id_embedding,
                                                item_catalog_embedding,item_tag_embedding],axis=1,name='embedding_concatenate')
    #sparse_features=tf.keras.layers.concatenate([item_id_input,gender_input,item_catalog_input,weekday_input,hour_input,minute_input,second_input],
                                                #name='sparse_feature_concatenate')
    #DNN
    dnn_l1=tf.keras.layers.Dense(hidden_units,kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        bias_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer1')
    dropout1=tf.keras.layers.Dropout(0.9)
    dnn_l2=tf.keras.layers.Dense(hidden_units,kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        bias_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer2')
    dropout2=tf.keras.layers.Dropout(0.9)
    dnn_l3=tf.keras.layers.Dense(hidden_units,kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        bias_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer3')
    dropout3=tf.keras.layers.Dropout(0.9)
    dnn_l4=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),use_bias=False,
                        bias_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation=None,name='dnn_layer4')
    #FM
    fm_linear=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                                   bias_regularizer=tf.keras.regularizers.l2(l2_regularization),name='fm_linear')(tf.keras.layers.Flatten()(dense_features))
    fm_cross=crosslayer(name='fm_cross')(dense_features)

    fm_logit=tf.keras.layers.Add(name='fm_combine')([fm_linear,fm_cross])
    
    #forward propagation
    dnn1=dnn_l1(tf.keras.layers.Flatten()(dense_features))
    dnn1_drop=dropout1(dnn1)
    dnn2=dnn_l2(dnn1_drop)
    dnn2_drop=dropout2(dnn2)
    dnn3=dnn_l3(dnn2_drop)
    dnn3_drop=dropout3(dnn3)
    dnn_logit=dnn_l4(dnn3_drop)
    pred=tf.keras.layers.Activation(activation='sigmoid',name='sigmoid')(tf.keras.layers.Add()([fm_logit,dnn_logit]))
    model=tf.keras.Model(inputs=[user_id_input,user_type_input,member_type_input,item_id_input,item_catalog_input,item_tag_input],
                        outputs=pred)
    return model

# online components
def retrain(model,prep_model,X,epochs=3,learning_rate=0.01):
    X=pd.DataFrame(X)
    X=sampling(X,0)
    y=X.pop('label').values
    X=X.to_dict(orient='list')
    for key in X.keys():
        X[key]=np.array(X[key])
    X=preprocess(prep_model,X)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.BinaryCrossentropy())
    model.fit(X,y,epochs=epochs,batch_size=64)
    return model
def guess_you_like(model,prep_model,df,topK=36,json_like=True,predict_type='single'):
        X=df
        for key in user_fields:
            X[key]=np.repeat(X[key],len(X['item_id']))
        X_prep=preprocess(prep_model,X)
        pred=model.predict(X_prep)
        df=pd.DataFrame(X)
        df['score']=pred
        df['rank']=df.groupby(['user_id'])['score'].rank(method='first',ascending=False).sort_values()
        df=df.sort_values(by=['user_id','rank']).reset_index(drop=True)
        recmd_list=df.loc[:,['user_id','item_id','item_catalog','rank','score']]
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
                    #name_list=list(recmd_list.loc[recmd_list['user_id']==user_id,'item_name'])
                    for i in range(len(item_list)):
                        item_dict=dict()
                        item_dict.setdefault('item_id',0)
                        #item_dict.setdefault('item_name',0)
                        item_dict.setdefault('rank',0)
                        item_dict['item_id']=item_list[i]
                        #item_dict['item_name']=name_list[i]
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
def load_model(path):
    print("loading", path)
    model=tf.keras.models.load_model(path)
    setting_file = path+'/saved_model.setting'
    if os.path.isfile(setting_file):
        with open(setting_file, 'r') as f:
            model.setting = json.load(f)
    return model

def save_model(model, path, new_max):
    print("saving model", path)
    model.save(path)
    print("saved model")
    if model.setting is not None:
        model.setting['MAX_ID'] = new_max
        model.setting['UPDATE_TIME'] = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        try:
            interations = int(model.setting['ITERATIONS'])
            model.setting['ITERATIONS'] = interations + 1
        except:
            model.setting['ITERATIONS'] = 1
    else:
        try:
            model.setting = json.loads(json.dumps({'MAX_ID': 1, 'UPDATE_TIME': str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 'VERSION': '0.1'}))
        except:
            print("model setting not exist.")
    setting_file = path+'\\saved_model.setting'
    print(setting_file)
    with open(setting_file, 'w') as f:
        json.dump(model.setting, f)

def sampling(data,ratio=3):
    item_pool=list(data['item_id'])
    data['label']=1
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
            data['member_type'].append(data['member_type'][user_idx])
            data['user_type'].append(data['user_type'][user_idx])
            data['item_id'].append(sample_item)
            data['item_catalog'].append(data['item_catalog'][sample_idx])
            data['item_tag'].append(data['item_tag'][sample_idx])
            data['label'].append(0)
            n+=1
            if n>ratio*len(items):
                break
    data=pd.DataFrame(data)
    idx=np.random.permutation(len(data))
    data=data.iloc[idx,:].reset_index(drop=True)
    return data
