import numpy as np
import pandas as pd
import joblib
import sqlalchemy
import tensorflow as tf
import warnings
import os
import FactorizationMachine as fm
import Preprocess as prep
warnings.filterwarnings('ignore')
path_model='E:\\kyk-ml\\Recommendation_FactorModel+lightgbm\\'

if __name__=='__main__':
    
    
    train=prep.load_full_data('2021/2/1','2021/2/7')
    item_pool=prep.load_item_pool('2021/2/1','2021/2/7')

    

    
    data=prep.NegativeSampling(train,item_pool,ratio=10)
    y=data['label'].values
    X=data.drop(columns='label')
    X=prep.process_data(X)

    model=fm.FactorizationMachine(feature_dim=X.shape[1],output_dim=64)
    if os.path.exists(path_model+'latentfactor.h5'):
        model.load_model()
        #model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(0.01),metrics=['AUC','binary_accuracy'])
        #model.load_weights(path_model+'latentfactor.h5')
        print(model.candidates(X.toarray()[:1]))
    else:
        model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(0.01),metrics=['AUC','binary_accuracy'])
        model.fit(X.toarray(),y,epochs=10,batch_size=512)
        model.summary()
        model.save_model()
   
    

    
    