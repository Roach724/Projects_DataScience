import pandas as pd
import numpy as np
import sqlalchemy
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
def RFM(feature_quantile=[0,0.2,0.4,0.6,0.8,1],rfm_quantile=[0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1]):
    engine='mysql+pymysql://kykviewer:$KykForView@keyikedb.mysql.rds.aliyuncs.com/wechat_finance_db'
    sql="select prod.id as item_id, cb.money/100 as income,cb.updated_on \
    from (chsell_product as prod INNER join chsell_order as co on prod.id=co.product_id) INNER JOIN chsell_bill as cb on cb.source_id=co.id \
    where cb.`status` like '%%PASS%%' order by updated_on"
    data=pd.read_sql(sql=sql,con=engine)
    data['updated_on']=pd.to_datetime(data['updated_on'],format='%Y/%m/%d')
    data['Datediff']=datetime.datetime.now()-data['updated_on']
    data['Datediff']=data['Datediff'].dt.days
    R_agg=data.groupby(['item_id'],as_index=False)['Datediff'].agg(min)
    F_agg=data.groupby(['item_id'],as_index=False)['updated_on'].agg('count')
    M_income_agg=data.groupby(['item_id'],as_index=False)['income'].agg(sum)
    agg=R_agg.merge(F_agg,on='item_id').merge(M_income_agg,on='item_id')
    agg.columns=['item_id','Recency','Frequency','Monetary_Income']
    cols=agg.drop(columns='item_id').columns
    for col in cols:
        bins=np.unique(agg[col].quantile(q=feature_quantile))
        bins[0]=0
        if col=='Recency':
            labels=[len(bins)-1-i for i in range(len(bins)-1)]
        else:
            labels=[i+1 for i in range(len(bins)-1)]

        col_bins=pd.cut(agg[col],bins,labels=labels,duplicates='drop')
        
        col_name=col+'_S'
        if col=='Recency':
            agg[col_name]=len(bins)-1-col_bins.values.codes
            agg.loc[agg[col_name]==len(bins)+1,col_name]=len(bins)
        else:
            agg[col_name]=col_bins.values.codes+1
            agg.loc[agg[col_name]==len(bins)+1,col_name]=1
    agg['RFM_Income']=0.2*agg['Recency_S'].astype('int')+0.3*agg['Frequency_S'].astype('int')+0.5*agg['Monetary_Income_S'].astype('int')
    bins=np.unique(agg['RFM_Income'].quantile(q=rfm_quantile))
    bins[0]=0
    rfmlabels=[i+1 for i in range(len(bins)-1)]
    agg['level_income']=pd.cut(agg['RFM_Income'],bins,labels=rfmlabels,duplicates='drop')
    std=StandardScaler()
    agg['item_weight']=std.fit_transform(agg['RFM_Income'].values.reshape(-1,1))
    return agg

