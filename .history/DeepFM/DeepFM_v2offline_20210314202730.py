import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import warnings
import datetime
import sqlalchemy
from sklearn.model_selection import KFold
import json
import os
warnings.filterwarnings('ignore')
user_tokens=60000
user_type_tokens=5
member_type_tokens=5
item_tokens=300
item_tag_vocab=300
sequence_length=8
item_catalog_tokens=13
tag_tokens=10
user_fields=['user_id','member_type','user_type']
item_fields=['item_id','item_catalog','item_tag']
int_fields=['item_catalog']
context_fields=['similarity']
#model hyper parameters
embedding_dim=8
l2_regularization=3.2
## offline components
'''
sql="select op.id as ID,channel_id as user_id,Channel_use_type as user_type,number_type as member_type, \
is_Tax_credit as is_tax,is_Ticket_loan as is_fapiao,is_Personal_loan as is_personal,is_Credit_card,is_Mortgage_loan as is_mortgage,is_financial,is_accumulation_fund,is_bill,is_Merchant_loan as is_merchant_loan, \
oper_obj as item_id,IFNULL(prod.catalog,0) as item_catalog,replace(replace(replace(IFNULL(prod.tags,'不可用'),'\\n',''),'，',' '),'、',' ') as item_tag \
from (bigdata.chsell_quick_bi as qbi left JOIN chsell_oper_data as op on qbi.channel_id=op.agent_id) inner join chsell_product as prod on prod.id=op.oper_obj \
where op.oper_type='HOME_PRODUCT'"
'''
sql="select op.id as ID,channel_id as user_id,Channel_use_type as user_type,number_type as member_type, \
oper_obj as item_id,IFNULL(prod.catalog,0) as item_catalog,replace(replace(replace(IFNULL(prod.tags,'不可用'),'\\n',''),'，',' '),'、',' ') as item_tag \
from (bigdata.chsell_quick_bi as qbi left JOIN chsell_oper_data as op on qbi.channel_id=op.agent_id) inner join chsell_product as prod on prod.id=op.oper_obj \
where op.oper_type='HOME_PRODUCT'"
def sampling(data,ratio=20,hard_negative_proba=0.01):
    item_pool=list(data['item_id'])
    data['label']=1
    user_item=dict(data.groupby(['user_id'])['item_id'].agg(list))                     
    data=data.to_dict(orient='list')
    for user_field,items in user_item.items():
        n=0
        user_idx=data['user_id'].index(user_field)
        for i in range(ratio*len(items)):
            sample_item=item_pool[np.random.randint(0,len(item_pool)-1)]
            flag=np.random.rand(1)[0]
            if sample_item in items and flag>hard_negative_proba:
                continue
            #data[user_field].append(sample_item)
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

def cross_validation(X,y,train_batch_size=16932,test_batch_size=4096,epochs=5,cv=2,n_splits=5,shuffle=True,random_state=None):
    #Cross validation
    kf=KFold(n_splits=n_splits,shuffle=shuffle,random_state=random_state)
    X_df=pd.DataFrame(X)
    losses=[]
    aucs=[]
    recalls=[]
    models=[]
    for i in range(cv):
        for train_idx,test_idx in kf.split(X_df):
            X_train,X_test=X_df.iloc[train_idx,:],X_df.iloc[test_idx,:]
            y_train,y_test=y[train_idx],y[test_idx]
            X_train=X_train.to_dict(orient='list')
            X_test=X_test.to_dict(orient='list')
            for col,values in X_train.items():
                X_train[col]=np.array(values)
                X_test[col]=np.array(X_test[col])
            model=DeepFM()
            vectorizer=Vectorizer()
            for feature in user_fields+item_fields:
                if feature not in int_fields:
                    vectorizer.get_layer(feature+'_vectorize').adapt(np.unique(X[feature]))
                else:
                    vectorizer.get_layer(feature+'_intlookup').adapt(np.unique(X[feature]))
            
            model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss=tf.keras.losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.Recall()])
            X_train=vectorizer.predict(X_train,batch_size=2**14)
            X_test=vectorizer.predict(X_test,batch_size=2**13)
            model.fit(X_train,y_train,epochs=epochs,batch_size=train_batch_size)
            loss,auc,recall=model.evaluate(X_test,y_test,batch_size=test_batch_size)
            models.append(model)
            losses.append(loss)
            aucs.append(auc)
            recalls.append(recall)
    min_loss=min(losses)
    idx=losses.index(min_loss)
    opt_model=models[idx]
    return opt_model
def Item2vec(X,hidden_units=[128,128,128,128]):
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

    DNN=tf.keras.Sequential(name='DNN')
    for layer_size in hidden_units:
        DNN.add(tf.keras.layers.Dense(layer_size,activation='relu'))
    DNN.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    pred=DNN(vector)

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
##model definition
class crosslayer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(crosslayer,self).__init__(**kwargs)
    def build(self,input_shape):
        super(crosslayer, self).build(input_shape)
    def call(self,inputs):
        square_of_sum=tf.keras.backend.square(tf.keras.backend.sum(inputs,axis=1,keepdims=True))
        sum_of_square=tf.keras.backend.sum(tf.keras.backend.square(inputs),axis=1,keepdims=True)
        diff=0.5*tf.keras.backend.sum(square_of_sum-sum_of_square,axis=2,keepdims=False)
        return diff
    def get_config(self):
        config=super(crosslayer,self).get_config()
        return config
def Vectorizer():
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
    item_catalog_intlookup=tf.keras.layers.experimental.preprocessing.IntegerLookup(name='item_catalog_vectorize')(item_catalog_input)

    model=tf.keras.Model(inputs=[user_id_input,user_type_input,member_type_input,item_id_input,item_catalog_input,item_tag_input],
                        outputs=[user_id_vectorize,user_type_vectorize,member_type_vectorize,
                                item_id_vectorize,item_catalog_intlookup,item_tag_vectorize])
    return model

def fit_vectorizer(vectorizer,X):
    for feature in user_fields+item_fields:
        if feature not in int_fields:
            vectorizer.get_layer(feature+'_vectorize').adapt(np.unique(X[feature]))
        else:
            vectorizer.get_layer(feature+'_intlookup').adapt(np.unique(X[feature]))
    vectorizer.compile()
    return vectorizer
def vectorize(vectorizer,data):
    data=vectorizer(data)
    data_dict={}
    i=0
    for feat in user_fields+item_fields:
        data_dict.setdefault(feat,0)
        data_dict[feat]=data[i]
        i+=1
    return data_dict
def Retrieval(hidden_units=[16,16,16,embedding_dim]):
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
    member_type_embedding=tf.keras.layers.Embedding(member_type_tokens+1,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='member_type_embedding')(member_type_input)
    user_type_embedding=tf.keras.layers.Embedding(user_type_tokens+1,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='user_type_embedding')(user_type_input)
    item_id_embedding=tf.keras.layers.Embedding(item_tokens,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_id_embedding')(item_id_input)
    item_tag_embedding=tf.keras.layers.Embedding(item_tag_vocab+1,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_tag_embedding')(item_tag_input)
    item_catalog_embedding=tf.keras.layers.Embedding(item_catalog_tokens+1,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_catalog_embedding')(item_catalog_input)
    user_feature=tf.keras.layers.concatenate([user_id_embedding,member_type_embedding,user_type_embedding],axis=1)
    item_feature=tf.keras.layers.concatenate([item_id_embedding,item_tag_embedding,item_catalog_embedding],axis=1)
    user_tower=tf.keras.Sequential(name='user_tower')
    item_tower=tf.keras.Sequential(name='item_tower')
    for layer_size in hidden_units:
        user_tower.add(tf.keras.layers.Dense(layer_size,activation='tanh',kernel_initializer=tf.keras.initializers.glorot_normal()))
        item_tower.add(tf.keras.layers.Dense(layer_size,activation='tanh',kernel_initializer=tf.keras.initializers.glorot_normal()))
    user_tower.add(tf.keras.layers.Dense(embedding_dim,name='user_output',kernel_initializer=tf.keras.initializers.glorot_normal()))
    item_tower.add(tf.keras.layers.Dense(embedding_dim,name='item_output',kernel_initializer=tf.keras.initializers.glorot_normal()))
    
    user_vector=user_tower(tf.keras.layers.Flatten()(user_feature))
    item_vector=item_tower(tf.keras.layers.Flatten()(item_feature))

    inner_product=tf.reduce_sum(tf.keras.layers.Multiply()([user_vector,item_vector]),axis=1)
    pred=tf.keras.layers.Activation(activation='sigmoid')(inner_product)
    model=tf.keras.Model(inputs=[user_id_input,user_type_input,member_type_input,item_id_input,item_catalog_input,item_tag_input],
                        outputs=pred)
    
    return model
def DeepFM(hidden_units=[256,256,256,256],dropouts=[0.4,0.5,0.6,0.7]):
    if len(hidden_units)!=len(dropouts):
        print('The number of hidden layer in DNN must equal to that of dropouts.')
        return 0
    #user field
    user_id_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='user_id')
    member_type_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='member_type')
    user_type_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='user_type')
    #item_field
    item_id_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='item_id')
    item_catalog_input=tf.keras.Input(shape=(1,),dtype=tf.int32,name='item_catalog')
    item_tag_input=tf.keras.Input(shape=(sequence_length,),dtype=tf.int32,name='item_tag')
    #context feature
    similarity=tf.keras.Input(shape=(1,),dtype=tf.float32,name='similarity')
    #Embedding
    user_id_embedding=tf.keras.layers.Embedding(user_tokens,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='user_id_embedding')(user_id_input)
    member_type_embedding=tf.keras.layers.Embedding(member_type_tokens+1,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='member_type_embedding')(member_type_input)
    user_type_embedding=tf.keras.layers.Embedding(user_type_tokens+1,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='user_type_embedding')(user_type_input)
    item_id_embedding=tf.keras.layers.Embedding(item_tokens,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_id_embedding')(item_id_input)
    item_tag_embedding=tf.keras.layers.Embedding(item_tag_vocab+1,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_tag_embedding')(item_tag_input)
    item_catalog_embedding=tf.keras.layers.Embedding(item_catalog_tokens+1,embedding_dim,embeddings_initializer=tf.keras.initializers.glorot_normal(),name='item_catalog_embedding')(item_catalog_input)
    
    user_id_embedding_sparse=tf.keras.layers.Embedding(user_tokens,1,name='user_id_embedding_sparse')(user_id_input)
    member_type_embedding_sparse=tf.keras.layers.Embedding(member_type_tokens+1,1,name='member_type_embedding_sparse')(member_type_input)
    user_type_embedding_sparse=tf.keras.layers.Embedding(user_type_tokens+1,1,name='user_type_embedding_sparse')(user_type_input)
    item_id_embedding_sparse=tf.keras.layers.Embedding(item_tokens,1,name='item_id_embedding_sparse')(item_id_input)
    item_tag_embedding_sparse=tf.keras.layers.Embedding(item_tag_vocab+1,1,name='item_tag_embedding_sparse')(item_tag_input)
    item_catalog_embedding_sparse=tf.keras.layers.Embedding(item_catalog_tokens+1,1,name='item_catalog_embedding_sparse')(item_catalog_input)
    #concatenation
    dense_features=tf.keras.layers.concatenate([user_id_embedding,member_type_embedding,user_type_embedding,item_id_embedding,
                                                item_catalog_embedding,item_tag_embedding],axis=1,name='embedding_concatenate')
    sparse_features=tf.keras.layers.concatenate([user_id_embedding_sparse,member_type_embedding_sparse,user_type_embedding_sparse,item_id_embedding_sparse,
                                                item_catalog_embedding_sparse,item_tag_embedding_sparse],axis=1,
                                                name='sparse_feature_concatenate')
    #DNN
    DNN=tf.keras.Sequential(name='DNN')
    for i,layer_size in enumerate(hidden_units):
        DNN.add(tf.keras.layers.Dense(layer_size,kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        bias_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer'+str(i+1)))
        DNN.add(tf.keras.layers.Dropout(dropouts[i]))
    DNN.add(tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),use_bias=False,
                        bias_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=tf.keras.initializers.glorot_normal(),activation=None,name='dnn_layer'+str(i+1)))
    #FM
    fm_linear=tf.keras.backend.sum(tf.keras.layers.Flatten(name='sparse_flatten')(sparse_features),axis=1)
    
    fm_cross=crosslayer(name='fm_cross')(dense_features)

    fm_logit=tf.keras.layers.Add(name='fm_combine')([fm_linear,fm_cross])
    
    #forward propagation
    dense_features_flatten=tf.keras.layers.Flatten(name='dense_flatten')(dense_features)
    dense_features_flatten_context=tf.keras.layers.concatenate([dense_features_flatten,similarity],axis=1,name='dense_feature_concatenate')
    dnn_logit=DNN(dense_features_flatten_context)
    pred=tf.keras.layers.Activation(activation='sigmoid',name='output')(tf.keras.layers.Add()([fm_logit,dnn_logit]))
    model=tf.keras.Model(inputs=[user_id_input,user_type_input,member_type_input,item_id_input,item_catalog_input,item_tag_input],
                        outputs=pred)
    return model
# online components

def guess_you_like(ranker,vectorizer,data,topK=36,json_like=True,predict_type='single'):
    X=data.copy()
    for key in user_fields:
        X[key]=np.repeat(X[key],len(X['item_id']))
    X_transform=vectorize(vectorizer,X)
    pred=ranker.predict(X_transform)
    data=pd.DataFrame(X)
    data['score']=pred
    data['rank']=data.groupby(['user_id'])['score'].rank(method='first',ascending=False).sort_values()
    recmd_list=data.sort_values(by=['user_id','rank']).reset_index(drop=True).loc[:,['user_id','item_id','item_catalog','rank','score']]
    if predict_type=='single':
        if topK=='all':
            pass
        else:
            recmd_list=data[data['rank']<=topK].sort_values(by=['user_id','rank'])
        if json_like:
            return jsonify(recmd_list)
        return recmd_list
    elif predict_type=='class':
        class_list=recmd_list.groupby(['user_id','item_catalog'],as_index=False)['score'].agg('mean')
        class_list['catalog_rank']=class_list.groupby(['user_id'])['score'].rank(method='first',ascending=False).sort_values()
        class_list=class_list.sort_values(by=['user_id','catalog_rank']).loc[:,['user_id','item_catalog','catalog_rank']]
        if json_like:
            return jsonify(class_list)
        return class_list

def get_lastNitem_embedding(vecotizer,retriever,user_log,user_sets=None,N=10):
    user_log['time_stamp']=pd.to_datetime(user_log['time_stamp'],dayfirst=True,infer_datetime_format=True)
    lastNitem_embedding_df=[]
    if user_sets is None:
        user_sets=user_log.user_id.unique()
    for user_id in user_sets:
        click_history=user_log.loc[user_log['user_id']==user_id,
                                ['user_id','item_id','item_tag','item_catalog','time_stamp']][-N:].reset_index(drop=True)
        max_record=len(click_history)
        click_history['time_weight']=1/(datetime.datetime.now()-click_history['time_stamp']).apply(lambda x: x.days)
        time_weight=click_history['time_weight'].values
        global_embedding_matrix=[]
        for idx in range(max_record):
            feature_embedding_matrix=[]
            for feature in item_fields:
                feature_input=np.array([click_history.loc[idx,feature]])
                vectorize=vecotizer.get_layer(feature+'_vectorize')(feature_input)
                embedding=retriever.get_layer(feature+'_embedding')(vectorize).numpy()
                if embedding.ndim==3:
                    embedding=np.array(tf.keras.layers.Flatten()(embedding).numpy())
                feature_embedding_matrix.append(embedding)
            feature_embedding_matrix=np.concatenate(feature_embedding_matrix,axis=1)
            embedding=retriever.get_layer(index=17)(feature_embedding_matrix).numpy()
            global_embedding_matrix.append(embedding.reshape(1,-1))
        global_avg_pooling_embedding=np.concatenate(global_embedding_matrix,axis=0)
        global_avg_pooling_embedding=np.average(global_avg_pooling_embedding,axis=0,weights=time_weight)
        global_avg_pooling_embedding_df=pd.DataFrame(global_avg_pooling_embedding.reshape(1,-1),
                                                columns=['last'+str(N)+'clicks_embeeding_V'+str(k+1) for k in range(embedding_dim)])
        global_avg_pooling_embedding_df['user_id']=user_id
        lastNitem_embedding_df.append(global_avg_pooling_embedding_df)
    lastNitem_embedding_df=pd.concat(lastNitem_embedding_df,axis=0)
    return lastNitem_embedding_df

def get_user_embedding(vectorizer,retriever,rank_data):
    user_sets=rank_data.user_id.unique()
    embedding_dfs=[]
    for user_id in user_sets:
        user_feature=rank_data.loc[rank_data['user_id']==user_id,user_fields].drop_duplicates()
        feature_embedding_matrix=[]
        for feature in user_fields:
            feature_input=user_feature[feature]
            vectorize=vectorizer.get_layer(feature+'_vectorize')(feature_input)
            embedding=retriever.get_layer(feature+'_embedding')(vectorize).numpy()
            if embedding.ndim==3:
                embedding=tf.keras.layers.Flatten()(embedding).numpy()
            feature_embedding_matrix.append(embedding)
        feature_embedding_matrix=np.concatenate(feature_embedding_matrix,axis=1)
        embedding=retriever.get_layer(index=16)(feature_embedding_matrix).numpy()
        embedding_df=pd.DataFrame(embedding.reshape(1,-1),columns=['user_embedding_V'+str(k+1) for k in range(embedding_dim)])
        embedding_df['user_id']=user_id
        embedding_dfs.append(embedding_df)
    user_embedding_df=pd.concat(embedding_dfs,axis=0)
    return user_embedding_df

def get_item_embedding(vectorizer,retriever,rank_data):
    item_sets=rank_data.item_id.unique()
    embedding_dfs=[]
    for item_id in item_sets:
        item_feature=rank_data.loc[rank_data['item_id']==item_id,item_fields].drop_duplicates()
        feature_embedding_matrix=[]
        for feature in item_fields:
            feature_input=np.array([item_feature[feature]])
            vectorize=vectorizer.get_layer(feature+'_vectorize')(feature_input)
            embedding=retriever.get_layer(feature+'_embedding')(vectorize).numpy()
            if embedding.ndim==3:
                embedding=tf.keras.layers.Flatten()(embedding).numpy()
            feature_embedding_matrix.append(embedding)
        feature_embedding_matrix=np.concatenate(feature_embedding_matrix,axis=1)
        embedding=retriever.get_layer(index=17)(feature_embedding_matrix).numpy()
        embedding_df=pd.DataFrame(embedding.reshape(1,-1),columns=['item_embedding_V'+str(k+1) for k in range(embedding_dim)])
        embedding_df['item_id']=item_id
        embedding_dfs.append(embedding_df)
    item_embedding_df=pd.concat(embedding_dfs,axis=0)
    return item_embedding_df


def load_model(path=r'.\DeepFM'):
    model=tf.keras.models.load_model(path)
    #setting_file = path.replace('.h5', '.setting')
    setting_file = path+'.setting'
    if os.path.isfile(setting_file):
        with open(setting_file, 'r') as f:
            model.setting = json.load(f)
            print(model.setting)
            #return model.setting['MAX_ID']
    return model
def save_model(model, path=r'.\DeepFM', new_max=None):
    model.save(path)
    try:
        model.setting = json.loads(json.dumps({'MAX_ID': new_max, 'UPDATE_TIME': str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 'VERSION': '0.1'}))
    except:
        model.setting['MAX_ID'] = new_max
        model.setting['UPDATE_TIME'] = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    #setting_file = path.replace('.h5', '.setting')
    setting_file = path+'.setting'
    with open(setting_file, 'w') as f:
        json.dump(model.setting, f)
def retrain(ranker,data,epochs=3,learning_rate=0.01):
    data=pd.DataFrame(data)
    data=sampling(data,0)
    data=data.to_dict(orient='list')
    for key in data.keys():
        data[key]=np.array(data[key])
    ranker.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate))
    ranker.fit(data,epochs=epochs,batch_size=32)
    return ranker

def jsonify(data):
    if not isinstance(data,pd.DataFrame):
        print('Input data must be a pandas DataFrame.')
        return 0
    result_set=list()
    user_ids=list(data['user_id'].unique())
    for user_id in user_ids:
        user_dict=dict()
        user_dict.setdefault('user_id',0)
        user_dict.setdefault('item_list',[])
        user_dict['user_id']=user_id
        item_list=list(data.loc[data['user_id']==user_id,'item_catalog'])
        rank_list=list(data.loc[data['user_id']==user_id,'catalog_rank'])
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
