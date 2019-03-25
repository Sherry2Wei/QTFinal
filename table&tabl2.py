#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dfply import *
import pandas as pd
import datetime
np.set_printoptions(suppress=True) 
os.chdir(r"D:\03lecture\QT\codeBackup\data")
from math import *


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


def end_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    return next_month - datetime.timedelta(days=next_month.day)

# formatting of msf
msf = pd.read_csv("msf.csv")
msf["date"] = pd.to_datetime(msf["date"], format='%Y%m%d')
msf["date"] = [end_of_month(date) for date in msf["date"]]
msf.columns = [i.lower() for i in msf.columns]
msf["month"] = [date.month for date in msf["date"]]
msf["year"] = msf.date.apply(lambda x:x.year)
# formatting of mse


# In[4]:


mse = pd.read_csv("mse.csv")


# In[5]:


mse.columns = [i.lower() for i in mse.columns]


# In[6]:


mse["exdt"] = pd.to_datetime(mse["exdt"], format='%Y%m%d')
mse["dclrdt"] = pd.to_datetime(mse["dclrdt"], format='%Y%m%d')
mse["paydt"] = pd.to_datetime(mse["paydt"], format='%Y%m%d')
mse = mse >> mask(X.divamt == X.divamt) ## dropna subset == divamt


# In[36]:


mse['date'] = mse['date'] = mse.exdt.agg(end_of_month)
# merge mse and msf


# In[37]:


mse['freq'] = mse.distcd.apply(lambda x:str(x)[:3])
mse['freq'] = mse.freq.astype('int64')
mse['freq'].loc[mse['freq'] <= 123] = 123 ## 012全部作为quarterly data 对待

mseUse = pd.DataFrame(mse.groupby(['permno','date','distcd','freq'])['divamt'].sum()).reset_index()


# In[299]:


data = pd.merge(msf,mseUse,how = 'left',on = ['permno','date'])


# In[300]:


data.pop('bidlo')
data.pop('askhi')
data["mcap"] = data["prc"] * data["shrout"]
# fill na with distcd if the stock has distcd before so that we can identify stock that pay dividend
data = data >> arrange(X.permno,X.date)
data['freq'] = data.groupby('permno')['freq'].fillna(method = 'ffill')


# In[301]:


data['distcd'] = data.groupby('permno')['distcd'].fillna(method = 'ffill')


# In[302]:


divDataSource = data.drop_duplicates(subset = ['permno','date']) ## 如果同一个月即有123 又有非123的 保留，


# In[303]:


## creat lag
divDataSource.sort_values(['permno','date'],inplace = True)
divDataSource['prc_lag1'] = divDataSource.groupby('permno')['prc'].shift(1)
divDataSource['mcap_lag1'] = divDataSource.groupby('permno')['mcap'].shift(1)
divDataSource['freq_lag1'] = divDataSource.groupby('permno')['freq'].shift(1)
divDataSource['turnover'] = divDataSource['vol']/divDataSource['shrout']
divDataSource['spread_2'] = divDataSource['ask'] -divDataSource['bid']


# In[304]:


divDataSource['Div_Yield'] = divDataSource['divamt']/divDataSource['prc']


# In[305]:


for i in range(12):
    div_lag = 'div_lag' + str(i + 1)
    divDataSource[div_lag] = divDataSource.groupby('permno')['divamt'].shift(i+1)


# In[306]:


import feather


# In[307]:


linkTable = feather.read_dataframe('monthly_gvkey_permno_link.feather')
linkTable['date'] = pd.to_datetime(linkTable['date'])

linkTable['N_permno']=linkTable.groupby('permno')['permno'].transform('count')
linkTable['year'] = linkTable.date.apply(lambda x:x.year)

linkTable.drop_duplicates(subset = ['gvkey','permno','year'],inplace = True)

linkTable['N_permno'] = linkTable.groupby(['year','gvkey'])['permno'].transform('count')

linkTable.query('N_permno >=2').sort_values(by = ['gvkey','year'])

BMdata = pd.read_csv('BMdata.csv')
BMdata['datadate'] = pd.to_datetime(BMdata.datadate,format = '%Y%m%d')
BMdata = BMdata[['gvkey','datadate','bkvlps']]

BMdata.rename(columns= {'datadate':'date'},inplace = True)
BMdata['year'] = BMdata.date.apply(lambda x:x.year)

BMdata_permno = BMdata.merge(linkTable,on = ['gvkey','year'],how= 'left',suffixes = ['_BM','_link'])
BMdata_permno.dropna(subset = ['bkvlps'],inplace= True)
BMdata_permno['permno'] = BMdata_permno.groupby('gvkey')['permno'].fillna(method = 'ffill')
BMdata_permno.drop_duplicates(subset = ['permno','date_BM','bkvlps'],inplace = True)
BMdata_permno.dropna(subset = ['permno'],inplace = True)
BMdata_permno['N_permno'] = BMdata_permno.groupby(['permno','date_BM'])['permno'].transform('count')
BMuse = pd.DataFrame(BMdata_permno.groupby(['permno','date_BM'])['bkvlps'].mean()).reset_index()
BMdata_permno.rename(columns = {'date_BM':'date'},inplace = True)


# In[308]:


BMdata_permno.head()


# In[309]:


divDataSourceBM = pd.merge(divDataSource,BMdata_permno.drop(columns = ['year']),on = ['permno','date'],how = 'left')
divDataSourceBM = divDataSourceBM >> arrange(X.permno,X.date)
divDataSourceBM['bkvlps'] = divDataSourceBM.groupby('permno')['bkvlps'].fillna(method = 'ffill')
divDataSourceBM['bm'] = divDataSourceBM['bkvlps']/divDataSource['prc']


# In[310]:


dataUse = divDataSourceBM.query('date <= "2011-12-31" &'
                          'shrcd in([10,11]) &'
                          'hexcd in([1,2,3]) &'
                          'prc_lag1 >= 5 &'
                          'ret not in(["B","C"])')


# In[311]:


dataUse['ret'] = dataUse.ret.astype('float')


# In[312]:


divData = dataUse.query('freq in(["120","121","123","124","125"])')


# ### 1 creat table1

# #### 1.1  creat table1PanelA

# In[313]:


table1PanelApart1 = divData[['mcap','turnover','spread_2','Div_Yield']].describe().T


# In[314]:


table1PanelApart2 = divData.query('bm <500 & bm >0')[['bm']].describe().T


# In[315]:


table1PanelA = pd.concat([table1PanelApart1,table1PanelApart2])


# In[316]:


N_firms = pd.DataFrame({'count':len(set(divData['permno']))},index = ['N_firms'])


# In[317]:


table1PanelA = pd.concat([table1PanelA,N_firms])


# In[318]:


table1PanelA


# #### 1.2 Creat table1PanelB

# In[319]:


NoDivData = dataUse.query('freq not in(["120","121","123","124","125"])')


# In[320]:


table1PanelBpart1 = NoDivData[['mcap','turnover','spread_2']].describe().T
table1PanelBpart2 = NoDivData.query('bm <1000&bm >0')[['bm']].describe().T


# In[321]:


table1PanelBpart1


# In[322]:


N_firms2 = pd.DataFrame({'count':len(set(NoDivData.permno))},index = ['N_firms'])


# In[323]:


N_firms2


# In[324]:


table1PanelB= pd.concat([table1PanelBpart1,table1PanelBpart2,N_firms2])


# In[325]:


table1PanelB


# ### <font face = 'Times New Roman'> 1.3 Panel C column 1: Percent of firm-months with dividend in the last year

# In[326]:


dataUse.columns


# In[328]:


dataUse['divType'] = dataUse.distcd.apply(lambda x:str(x)[:3])


# In[329]:


DivFreq = pd.DataFrame(dataUse.groupby(['year','divType'])['divType'].count())
DivFreq['percent'] = DivFreq.divType/DivFreq.groupby(level = 0).divType.transform(np.sum)
DivFreq.columns = ['count','percent']
DivFreq = DivFreq.reset_index()


# In[296]:


DivFreq


# In[291]:


dataUse['divType'] = dataUse.groupby('permno')['divType'].fillna(method = 'ffill')


# In[294]:


dataUse[['permno','divType','freq']]


# In[398]:


DivFreq= pd.DataFrame(DivFreq.groupby('divType')['percent'].mean())


# In[395]:


div_freq = ['120','122','123','124','125','126','127']
Any_freq = table1PanelC_part1.loc[div_freq,].sum()
div_freq_index = {'monthly':'122','quarterly':'123','semiAnnualy':'124','Annualy':'125'}


# In[432]:


DivFreq.head(10)


# In[434]:


table1PanelC_part1 = DivFreq.loc[['122','123','124','125','121'],]


# In[435]:


table1PanelC_part1


# In[436]:


table1PanelC_part1.loc['Any_freq']= {'percent':Any_freq.tolist()[0]}


# In[437]:


table1PanelC_part1


# In[412]:


table1PanelC_part1.rename(index= {"122":'monthly',"123":"quarterly","124":"semiAnnualy","125":"Annualy"},inplace = True)


# In[413]:


table1PanelC_part1


# ### <font face = 'Times New Roman'>  Panel C-Column 2: Percent of dividend observations

# In[418]:


divData['divType'] = divData.distcd.apply(lambda x:str(x)[:3])


# In[419]:


DivFreq2 = pd.DataFrame(divData.groupby('divType')['divType'].count())


# In[420]:


DivFreq2.columns = ['count']


# In[421]:


DivFreq2


# In[422]:


DivFreq2['percent'] = DivFreq2['count']/DivFreq2['count'].sum()


# In[429]:


table1PanelC_part2 = DivFreq2[['percent']].loc[['122','123','124','125','121'],]


# In[448]:


table1PanelC = pd.merge(table1PanelC_part1,table1PanelC_part2,left_index = True,right_index = True,how = 'outer')


# In[450]:


table1PanelC.rename(index = {'121':'unknow_freq','122':'monthly','123':'quarterly','124':'semiAnnualy','125':'Annualy'},inplace = True)


# In[454]:


table1PanelC = round(table1PanelC.apply(lambda x:x*100),2)


# In[455]:


table1PanelC


# ### 2  creat Table2

# #### 2.1 creat Table2 PanelA

# In[218]:


table2_mean = {}
table2_std = {}
table2_all_div_prob = {}
table2_quar_div_prob = {}


# In[210]:


divData.columns.tolist().index('div_lag1')


# In[211]:


divData.columns.tolist().index('div_lag12')


# In[214]:


div_lag_list = divData.columns[23:35].tolist()


# In[215]:


div_lag_list


# In[219]:


for i,div_lag in enumerate(div_lag_list):
    table2_mean[str(i + 1)] = divData.dropna(subset = [div_lag])['ret'].mean()*100
    table2_std[str(i + 1)] = divData.dropna(subset = [div_lag] )['ret'].std()*100
    table2_all_div_prob[str(i+1)] = len(divData.dropna(subset = [div_lag,'divamt']))/len(divData.dropna(subset = [div_lag]))
    divData_quar = divData.query('freq == "123"')
    table2_quar_div_prob[str(i + 1)] = len(divData_quar.dropna(subset = [div_lag,'divamt']))/len(divData_quar.dropna(subset = [div_lag]))


# In[220]:


table2_panelA = pd.DataFrame(pd.Series(table2_mean),columns = ['Mean return'])
table2_panelA['Standard Deviation'] = table2_std.values()
table2_panelA['All dividends'] = table2_all_div_prob.values()
table2_panelA['quar dividends'] = table2_quar_div_prob.values()


# In[222]:


table2_panelA#  ## quartly dividends 的return 有些偏高，因该是120，121，122都归类为123的原因，后面可以再调整下。


# #### 2.2 creat table2 PanelB

# In[196]:


# assign signal
divData["signal"] = np.where((divData.div_lag3.notna() | divData.div_lag6.notna() |
                                 divData.div_lag9.notna() | divData.div_lag12.notna()), "L",
                                np.where(divData.div_lag1.notna() | divData.div_lag2.notna() |
                                         divData.div_lag4.notna() | divData.div_lag5.notna() |
                                         divData.div_lag7.notna() | divData.div_lag8.notna() |
                                         divData.div_lag10.notna() | divData.div_lag11.notna(), "S", ""))

portfolio1 = pd.DataFrame(divData.groupby(["date", "signal"])['ret'].mean())


# In[229]:


probs = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]


# In[257]:


table2PanelB_part1 = portfolio1.groupby('signal')['ret'].describe(percentiles = probs).loc[['L','S']].drop(
    columns = ['count','max','min']).apply(lambda x:x*100)


# In[258]:


table2PanelB_part1


# In[272]:


table2PanelB_part2 = pd.DataFrame(NoDivData.groupby('date')['ret'].mean()).describe(percentiles = probs).T.drop(columns = ['count','min','max']).apply(lambda x:x*100)


# In[273]:


table2PanelB = pd.concat([table2PanelB_part1,table2PanelB_part2])


# In[279]:


table2PanelB['index'] = ['div_predM','div_NoPredM','NoDiv']


# In[282]:


table2PanelB.set_index('index',inplace = True)


# In[283]:


table2PanelB

