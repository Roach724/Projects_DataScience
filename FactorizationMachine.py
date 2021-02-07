import pandas as pd
import numpy as np
import tensorflow as tf
import sqlalchemy
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import os
import warnings
warnings.filterwarnings('ignore')
path_model='E:\\kyk-ml\\Recommendation_FactorModel+lightgbm\\'
class crosslayer(tf.keras.layers.Layer):
    def __init__(self,input_dim,output_dim=64,**kwargs):
        super(crosslayer,self).__init__(**kwargs)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.kernel = self.add_weight(name='kernel', 
                                     shape=(self.input_dim, self.output_dim),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self,x):
        a=tf.keras.backend.pow(tf.keras.backend.dot(x,self.kernel),2)
        b=tf.keras.backend.dot(tf.keras.backend.pow(x,2),tf.keras.backend.pow(self.kernel,2))
        return 0.5*tf.keras.backend.mean(a-b,1,keepdims=True)
    
  
class FactorizationMachine(tf.keras.Model):
    def __init__(self,feature_dim,output_dim=64,linear_reg=0.01,bias_reg=0.01):
        super(FactorizationMachine,self).__init__()
        self.feature_dim=feature_dim
        self.output_dim=output_dim
        self.linear_reg=linear_reg
        self.bias_reg=bias_reg
        self.linear=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(self.linear_reg),
                                          bias_regularizer=tf.keras.regularizers.l2(self.bias_reg))
        self.crosslayer=crosslayer(self.feature_dim,self.output_dim)
        self.logit=tf.keras.layers.Add()
        self.pred=tf.keras.layers.Activation(activation='sigmoid')
    def call(self,x):
        linear=self.linear(x)
        cross=self.crosslayer(x)
        logit=self.logit([linear,cross])
        pred=self.pred(logit)
        return pred
    def candidates(self,x,recommend_dim=45):
        probs=self.predict(x)
        if np.ndim(probs)<2:
            return np.sort(probs)[:(recommend_dim+1)]
        return np.sort(probs,axis=1)[:,:(recommend_dim+1)]
    def save_model(self):
        self.save_weights(path_model+'factors.h5')
    def load_model(self):
        if os.path.exists(path_model+'factors.h5'):
            self.load_weights(path_model+'factors.h5')


 



   