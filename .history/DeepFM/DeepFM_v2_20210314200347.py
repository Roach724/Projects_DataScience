import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import datetime
import json
import os
warnings.filterwarnings('ignore')
user_fields=['user_id','member_type','user_type']
item_fields=['item_id','item_catalog','item_tags']
#tool functions
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
   
def sampling(data,ratio=20,hard_negative_proba=0.01):
    item_pool=list(data['item_id'])
    data['label']=1
    user_item=dict(data.groupby(['user_id'])['item_id'].agg(list))
    #user_item.setdefault('label',0)
    #user_item['label']=list(data['label'])                       
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

def vectorize(vectorizer,data):
    data=vectorizer(data)
    data_dict={}
    i=0
    for feat in user_fields+item_fields:
        data_dict.setdefault(feat,0)
        data_dict[feat]=data[i]
        i+=1
    return data_dict

# online components
def retrain(ranker,data,epochs=3,learning_rate=0.01):
    data=pd.DataFrame(data)
    data=sampling(data,0)
    data=data.to_dict(orient='list')
    for key in data.keys():
        data[key]=np.array(data[key])
    ranker.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate))
    ranker.fit(data,epochs=epochs,batch_size=32)
    return ranker
def guess_you_like(ranker,vectorizer,df,topK=36,json_like=True,predict_type='single'):
    X=df.copy()
    for key in user_fields:
        X[key]=np.repeat(X[key],len(X['item_id']))
    X_transform=vectorize(vectorizer,X)
    pred=ranker.predict(X_transform)
    df=pd.DataFrame(X)
    df['score']=pred
    df['rank']=df.groupby(['user_id'])['score'].rank(method='first',ascending=False).sort_values()
    recmd_list=df.sort_values(by=['user_id','rank']).reset_index(drop=True).loc[:,['user_id','item_id','item_catalog','rank','score']]
    if predict_type=='single':
        if topK=='all':
            pass
        else:
            recmd_list=df[df['rank']<=topK].sort_values(by=['user_id','rank'])
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

