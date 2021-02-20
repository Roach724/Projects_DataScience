import tensorflow as tf
import pandas as pd
import warnings
import os
import DeepFM as dfm
import preprocess as prep
warnings.filterwarnings('ignore')

model=dfm.DeepFatorizationMachine(64,12800)
if os.path.exists(r'.\Recommendation_FactorModel+lightgbm\DeepFM\DeepFM.h5'):
    #加载模型进行预测
    print('loading model.\n' )
    model.hot_start()
    model.load_weights('DeepFM.h5')
    user_ids=["'00d481e75bed4c4aa2cdc0799711fe68'","'017a5094fe544967ad557489a3c97189'","'714a4f3891024e1daf6753e01a14cbb8'","'8fd1e9489a4249e28d9e7c39a7a56ee6'"]
    recmd=model.feeling_lucky(user_ids,topK=10)
    print(recmd)
else:
    #全量重新训练
    train_set,test_set=model.data_pipeline(sampling_ratio=3,test=True)
    train_set=train_set.shuffle(10000).batch(4096)
    test_set=test_set.shuffle(1000).batch(512)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.01),metrics=[dfm.roc_auc,tf.keras.metrics.Recall(),
    tf.keras.metrics.Precision(),tf.keras.metrics.RecallAtPrecision(0.6)])
    model.fit(train_set,epochs=1)
    model.evaluate(test_set)
    model.summary()
    model.save_weights('.\Recommendation_FactorModel+lightgbm\DeepFM\DeepFM.h5')

    user_ids=["'00d481e75bed4c4aa2cdc0799711fe68'","'017a5094fe544967ad557489a3c97189'","'714a4f3891024e1daf6753e01a14cbb8'","'8fd1e9489a4249e28d9e7c39a7a56ee6'"]
    recmd=model.feeling_lucky(user_ids,topK=10)
    print(recmd)