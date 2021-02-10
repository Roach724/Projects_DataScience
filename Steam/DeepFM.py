import numpy as np
import tensorflow as tf
import warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
path_model='E:\\kyk-ml\\Recommendation_FactorModel+lightgbm\\'
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
        sparse_feature,embedding=inputs
        linear=self.linear(sparse_feature)
        cross=self.crosslayer(embedding)
        logit=self.logit([linear,cross])
        return logit
class DNN(tf.keras.layers.Layer):
    def __init__(self):
        super(DNN,self).__init__()
    def build(self,input_shape):
        self.dense1=tf.keras.layers.Dense(200,kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                            bias_regularizer=tf.keras.regularizers.l2(0.01),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer1')
        self.dropout1=tf.keras.layers.Dropout(0.6)
        self.dense2=tf.keras.layers.Dense(200,kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                            bias_regularizer=tf.keras.regularizers.l2(0.01),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer2')
        self.dropout2=tf.keras.layers.Dropout(0.9)
        self.dense3=tf.keras.layers.Dense(200,kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                            bias_regularizer=tf.keras.regularizers.l2(0.01),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer3')
        self.dropout3=tf.keras.layers.Dropout(0.8)
        self.dense4=tf.keras.layers.Dense(200,kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                            bias_regularizer=tf.keras.regularizers.l2(0.01),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),activation='relu',name='dnn_layer4')
        self.dropout4=tf.keras.layers.Dropout(0.7)
        self.dense5=tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                            bias_regularizer=tf.keras.regularizers.l2(0.01),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(),name='dnn_layer5')
        
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
    def __init__(self,embedding_dim=64,hash_bins=100000):
        super(DeepFatorizationMachine,self).__init__()
        self.embedding_dim=embedding_dim
        self.hash_bins=hash_bins
    def build(self,input_shape):

        #Hashing and embedding for sparse/categorical features
        self.user_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=self.hash_bins,name='user_field_hashing')
        self.user_embedding=tf.keras.layers.Embedding(input_dim=self.hash_bins,output_dim=self.embedding_dim,name='user_field_embedding')
        self.item_hash=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=self.hash_bins,name='item_field_hashing')
        self.item_embedding=tf.keras.layers.Embedding(input_dim=self.hash_bins,output_dim=self.embedding_dim,name='item_field_embedding')

        self.flatten=tf.keras.layers.Flatten()
        self.FM=FM(self.embedding_dim*2)
        self.DNN=DNN()
        self.add=tf.keras.layers.Add()
        self.pred=tf.keras.layers.Activation(activation='sigmoid')
    def call(self,inputs,training=None):
        
        user_field=inputs['user_field']
        item_field=inputs['item_field']
        sparse_matrix=inputs['sparse_matrix']
        #Hashing for user/item ids
        if training:
            self.user_hash.adapt(user_field)
            self.item_hash.adapt(item_field)

        user_hash=self.user_hash(user_field)
        item_hash=self.item_hash(item_field)
        #embedding layers for user/item related sparse features
        user_embedding=self.user_embedding(user_hash)
        user_embedding=self.flatten(user_embedding)
        item_embedding=self.item_embedding(item_hash)
        item_embedding=self.flatten(item_embedding)
        
        embedding=tf.concat([user_embedding,item_embedding],axis=1)

        #FM layer
        fm_pred=self.FM([sparse_matrix,embedding])
        #DNN layer
        dnn_pred=self.DNN(embedding)
        #combine the outputs from the two layers
        add=self.add([fm_pred,dnn_pred])
        #map the output by sigmoid function
        pred=self.pred(add)
        return pred

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
def roc_auc(y_true,y_pred):
    return tf.py_function(roc_auc_score,(y_true,y_pred),tf.float16)


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
