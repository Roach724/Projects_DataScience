import numpy as np
import pandas as pd
import joblib
import sqlalchemy
import tensorflow as tf
import warnings
import os
import FactorizationMachine as fm
import Preprocess as prep
from sklearn.metrics import roc_curve,confusion_matrix,recall_score
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
path_model='E:\\kyk-ml\\Recommendation_FactorModel+lightgbm\\'

if __name__=='__main__':
    train=prep.load_full_data('2021/2/1','2021/2/7')
    item_pool=prep.load_item_pool('2021/2/1','2021/2/7')

    data=prep.NegativeSampling(train,item_pool,ratio=10)
    y=data['label'].values
    X=data.drop(columns='label')
    X=prep.process_data(X)
    _,feature_dim=X.shape
    trainset=tf.data.Dataset.from_tensor_slices((X.toarray(),y))
    trainset=trainset.shuffle(60000).batch(256)
    del X
    

    model=fm.FactorizationMachine(feature_dim,output_dim=64)
    #model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(0.01),metrics=['AUC','binary_accuracy','Recall'])
    #model.fit(trainset,epochs=2)


    
    if os.path.exists(path_model+'factors.pb'):
        model=tf.saved_model.load(path_model+'factors.pb')
        #model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(0.01),metrics=['AUC','binary_accuracy','Recall'])
        #model.evaluate(trainset)
        #model.fit(trainset,epochs=5)
        #model.evaluate(trainset)
    else:
        model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(0.01),metrics=['AUC','binary_accuracy','Recall'])
        model.fit(trainset,epochs=10)
        model.evaluate(trainset)
        model.summary()
        model.save_model()
   
    y_pred=model.predict(trainset)
    fpr,tpr,_=roc_curve(y,y_pred)
    plt.plot(fpr,tpr)
    plt.show()
    

    
    