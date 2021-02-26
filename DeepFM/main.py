import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import os
from tensorflow.keras import callbacks
import DeepFM_offline as dfm
import datetime
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
inference=True
inc_train=False
evaluate=False
model=dfm.DeepFatorizationMachine(records=10000)
'''
ids=dfm.load_user_id()
user_ids=[]
for id in np.array(ids)[:,0]:
    user_ids.append("'" + id + "'")
'''
user_ids=["'00d481e75bed4c4aa2cdc0799711fe68'","'017a5094fe544967ad557489a3c97189'","'714a4f3891024e1daf6753e01a14cbb8'",
"'8fd1e9489a4249e28d9e7c39a7a56ee6'","'ea27a55e616345afbcdea83cab383e8a'"]

if os.path.exists(r'.\deeplearning\DeepFM\DeepFM.h5'):
    #加载模型进行预测
    if inference:
        print('loading model.\n' )
        model.load_model_weights()
        #model.save_pb()
        #recmd=model.feeling_lucky(user_ids,topK=10,scores=True)
        #dfm.insert_data(recmd)
        #print(recmd)
        print(model.MAX_ID)
    if evaluate:
        print('loading model.\n' )
        model.load_model_weights()
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),metrics=[dfm.roc_auc,tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
        eval_set=model.data_pipeline(full_train=False,increment=True,forth=1)
        eval_set=eval_set.shuffle(len(eval_set)).batch(1024)
        model.evaluate(eval_set)
    #增量训练
    if inc_train:
        print('loading model.\n' )
        model.load_model_weights()
        train_inc_set=model.data_pipeline(sampling_ratio=3,full_train=False,increment=True,forth=1)
        train_inc_set=train_inc_set.shuffle(len(train_inc_set)).batch(4096)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.02),metrics=[dfm.roc_auc,tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision()])
        hist=model.fit(train_inc_set,epochs=3)
        model.evaluate(train_inc_set)
        model.save_model_weights()
        
else:
    #全量重新训练
    train_set,test_set=model.data_pipeline(sampling_ratio=3,full_train=True,increment=False,test=True,forth=14)
    train_set=train_set.shuffle(len(train_set)).batch(90000)
    test_set=test_set.shuffle(len(test_set)).batch(1024)
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.01),metrics=[dfm.roc_auc,tf.keras.metrics.Recall(),
    tf.keras.metrics.Precision()])
    #log_dir=r'.\deeplearning\DeepFM'
    #call_back=TensorBoard(log_dir=log_dir,histogram_freq=0,write_graph=True,write_images=True,update_freq='epoch',embeddings_freq=0,profile_batch=10)
    earlystop_callback=tf.keras.callbacks.EarlyStopping(monitor='val_roc_auc',min_delta=0.01,patience=5,mode='max')
    history=model.fit(train_set,epochs=5,validation_data=test_set,callbacks=earlystop_callback)
    
    model.evaluate(test_set)
    model.save_model_weights()

