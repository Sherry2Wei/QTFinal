#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from dfply import *
import pandas as pd
import datetime
np.set_printoptions(suppress=True) 
os.chdir(r"D:\03lecture\QT\codeBackup\data")
from math import *


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


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


# In[5]:


mse = pd.read_csv("mse.csv")


# In[6]:


mse.columns = [i.lower() for i in mse.columns]


# In[7]:


mse["exdt"] = pd.to_datetime(mse["exdt"], format='%Y%m%d')
mse["dclrdt"] = pd.to_datetime(mse["dclrdt"], format='%Y%m%d')
mse["paydt"] = pd.to_datetime(mse["paydt"], format='%Y%m%d')
mse = mse >> mask(X.divamt == X.divamt) ## dropna subset == divamt


# In[8]:


mse['date'] = mse['date'] = mse.exdt.agg(end_of_month)
# merge mse and msf


# In[9]:


mse['freq'] = mse.distcd.apply(lambda x:str(x)[:3])
mse['freq'] = mse.freq.astype('int64')
mse['freq'].loc[mse['freq'] <= 123] = 123 ## 012全部作为quarterly data 对待

mseUse = pd.DataFrame(mse.groupby(['permno','date','distcd','freq'])['divamt'].sum()).reset_index()


# In[10]:


data = pd.merge(msf,mseUse,how = 'left',on = ['permno','date'])


# In[11]:


data.pop('bidlo')
data.pop('askhi')
data["mcap"] = data["prc"] * data["shrout"]
# fill na with distcd if the stock has distcd before so that we can identify stock that pay dividend
data = data >> arrange(X.permno,X.date)
data['freq'] = data.groupby('permno')['freq'].fillna(method = 'ffill')


# In[12]:


data['distcd'] = data.groupby('permno')['distcd'].fillna(method = 'ffill')


# In[13]:


divDataSource = data.drop_duplicates(subset = ['permno','date']) ## 如果同一个月即有123 又有非123的 保留，


# In[14]:


## creat lag
divDataSource.sort_values(['permno','date'],inplace = True)
divDataSource['prc_lag1'] = divDataSource.groupby('permno')['prc'].shift(1)
divDataSource['mcap_lag1'] = divDataSource.groupby('permno')['mcap'].shift(1)
divDataSource['freq_lag1'] = divDataSource.groupby('permno')['freq'].shift(1)
divDataSource['turnover'] = divDataSource['vol']/divDataSource['shrout']
divDataSource['spread_2'] = divDataSource['ask'] -divDataSource['bid']


# In[15]:


divDataSource['Div_Yield'] = divDataSource['divamt']/divDataSource['prc']


# In[16]:


for i in range(12):
    div_lag = 'div_lag' + str(i + 1)
    divDataSource[div_lag] = divDataSource.groupby('permno')['divamt'].shift(i+1)


# In[17]:


import feather


# In[18]:


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


# In[19]:


divDataSourceBM = pd.merge(divDataSource,BMdata_permno.drop(columns = ['year']),on = ['permno','date'],how = 'left')
divDataSourceBM = divDataSourceBM >> arrange(X.permno,X.date)
divDataSourceBM['bkvlps'] = divDataSourceBM.groupby('permno')['bkvlps'].fillna(method = 'ffill')
divDataSourceBM['bm'] = divDataSourceBM['bkvlps']/divDataSource['prc']


# In[20]:


dataUse = divDataSourceBM.query('date <= "2011-12-31" &'
                          'shrcd in([10,11]) &'
                          'hexcd in([1,2,3]) &'
                          'prc_lag1 >= 5 &'
                          'ret not in(["B","C"])')


# In[21]:


dataUse['ret'] = dataUse.ret.astype('float')


# In[22]:


divData = dataUse.query('freq in(["120","121","123","124","125"])')


# ### 1 creat table1

# #### 1.1  creat table1PanelA

# In[23]:


table1PanelApart1 = divData[['mcap','turnover','spread_2','Div_Yield']].describe().T


# In[24]:


table1PanelApart2 = divData.query('bm <500 & bm >0')[['bm']].describe().T


# In[25]:


table1PanelA = pd.concat([table1PanelApart1,table1PanelApart2])


# In[26]:


N_firms = pd.DataFrame({'count':len(set(divData['permno']))},index = ['N_firms'])


# In[27]:


table1PanelA = pd.concat([table1PanelA,N_firms])


# In[28]:


table1PanelA


# #### 1.2 Creat table1PanelB

# In[29]:


NoDivData = dataUse.query('freq not in(["120","121","123","124","125"])')


# In[42]:


table1PanelBpart1 = NoDivData[['mcap','turnover','spread_2']].describe().T
table1PanelBpart2 = NoDivData.query('bm <1000&bm >0')[['bm']].describe().T


# In[43]:


table1PanelBpart2


# In[44]:


N_firms2 = pd.DataFrame({'count':len(set(NoDivData.permno))},index = ['N_firms'])


# In[45]:


N_firms2


# In[46]:


table1PanelB= pd.concat([table1PanelBpart1,table1PanelBpart2,N_firms2])


# In[47]:


table1PanelB


# ### <font face = 'Times New Roman'> 1.3 Panel C column 1: Percent of firm-months with dividend in the last year

# In[48]:


dataUse.columns


# In[49]:


dataUse['divType'] = dataUse.distcd.apply(lambda x:str(x)[:3])


# In[74]:


DivFreq = pd.DataFrame(dataUse.groupby(['year','divType'])['divType'].count())
DivFreq['percent'] = DivFreq.divType/DivFreq.groupby(level = 0).divType.transform(np.sum)
DivFreq.columns = ['count','percent']
DivFreq = DivFreq.reset_index()


# In[76]:


DivFreq.head()


# In[77]:


dataUse['divType'] = dataUse.groupby('permno')['divType'].fillna(method = 'ffill')


# In[78]:


DivFreq= pd.DataFrame(DivFreq.groupby('divType')['percent'].mean())


# In[79]:


div_freq = ['120','122','123','124','125','126','127']
Any_freq = DivFreq.loc[div_freq,].sum()
div_freq_index = {'monthly':'122','quarterly':'123','semiAnnualy':'124','Annualy':'125'}


# In[80]:


DivFreq.head(10)


# In[81]:


table1PanelC_part1 = DivFreq.loc[['122','123','124','125','121'],]


# In[82]:


table1PanelC_part1


# In[83]:


table1PanelC_part1.loc['Any_freq']= {'percent':Any_freq.tolist()[0]}


# In[84]:


table1PanelC_part1


# In[63]:


table1PanelC_part1


# ### <font face = 'Times New Roman'>  Panel C-Column 2: Percent of dividend observations

# In[85]:


divData['divType'] = divData.distcd.apply(lambda x:str(x)[:3])


# In[86]:


DivFreq2 = pd.DataFrame(divData.groupby('divType')['divType'].count())


# In[87]:


DivFreq2.columns = ['count']


# In[88]:


DivFreq2


# In[89]:


DivFreq2['percent'] = DivFreq2['count']/DivFreq2['count'].sum()


# In[90]:


table1PanelC_part2 = DivFreq2[['percent']].loc[['122','123','124','125','121'],]


# In[91]:


table1PanelC = pd.merge(table1PanelC_part1,table1PanelC_part2,left_index = True,right_index = True,how = 'outer')


# In[92]:


table1PanelC.rename(index = {'121':'unknow_freq','122':'monthly','123':'quarterly','124':'semiAnnualy','125':'Annualy'},inplace = True)


# In[93]:


table1PanelC = round(table1PanelC.apply(lambda x:x*100),2)


# In[94]:


table1PanelC


# ### 2  creat Table2

# #### 2.1 creat Table2 PanelA

# In[95]:


table2_mean = {}
table2_std = {}
table2_all_div_prob = {}
table2_quar_div_prob = {}


# In[101]:


index_start = divData.columns.tolist().index('div_lag1')
index_end = divData.columns.tolist().index('div_lag12')


# In[104]:


div_lag_list = divData.columns[index_start:index_end+1].tolist()


# In[105]:


div_lag_list


# In[106]:


for i,div_lag in enumerate(div_lag_list):
    table2_mean[str(i + 1)] = divData.dropna(subset = [div_lag])['ret'].mean()*100
    table2_std[str(i + 1)] = divData.dropna(subset = [div_lag] )['ret'].std()*100
    table2_all_div_prob[str(i+1)] = len(divData.dropna(subset = [div_lag,'divamt']))/len(divData.dropna(subset = [div_lag]))
    divData_quar = divData.query('freq == "123"')
    table2_quar_div_prob[str(i + 1)] = len(divData_quar.dropna(subset = [div_lag,'divamt']))/len(divData_quar.dropna(subset = [div_lag]))


# In[107]:


table2_panelA = pd.DataFrame(pd.Series(table2_mean),columns = ['Mean return'])
table2_panelA['Standard Deviation'] = table2_std.values()
table2_panelA['All dividends'] = table2_all_div_prob.values()
table2_panelA['quar dividends'] = table2_quar_div_prob.values()


# In[108]:


table2_panelA#  ## quartly dividends 的return 有些偏高，因该是120，121，122都归类为123的原因，后面可以再调整下。


# #### 2.2 creat table2 PanelB

# In[109]:


# assign signal
divData["signal"] = np.where((divData.div_lag3.notna() | divData.div_lag6.notna() |
                                 divData.div_lag9.notna() | divData.div_lag12.notna()), "L",
                                np.where(divData.div_lag1.notna() | divData.div_lag2.notna() |
                                         divData.div_lag4.notna() | divData.div_lag5.notna() |
                                         divData.div_lag7.notna() | divData.div_lag8.notna() |
                                         divData.div_lag10.notna() | divData.div_lag11.notna(), "S", ""))

portfolio1 = pd.DataFrame(divData.groupby(["date", "signal"])['ret'].mean())


# In[110]:


probs = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]


# In[111]:


table2PanelB_part1 = portfolio1.groupby('signal')['ret'].describe(percentiles = probs).loc[['L','S']].drop(
    columns = ['count','max','min']).apply(lambda x:x*100)


# In[112]:


table2PanelB_part1


# In[113]:


table2PanelB_part2 = pd.DataFrame(NoDivData.groupby('date')['ret'].mean()).describe(percentiles = probs).T.drop(columns = ['count','min','max']).apply(lambda x:x*100)


# In[114]:


table2PanelB = pd.concat([table2PanelB_part1,table2PanelB_part2])


# In[115]:


table2PanelB['index'] = ['div_predM','div_NoPredM','NoDiv']


# In[116]:


table2PanelB.set_index('index',inplace = True)


# In[142]:


def weighted_average(dataframe):
    return (dataframe.ret * dataframe.mcap_lag1).sum() / dataframe.mcap_lag1.sum()


# In[143]:


divData.columns


# In[144]:


port1 = divData.groupby(["date", "signal"]).apply(lambda x: pd.Series({"vwret": weighted_average(x),
                                                                                   "ewret": x.ret.mean()}))


# In[145]:


port1


# In[146]:


port1.reset_index(inplace=True)
# calculate the total return of equal weight and value weight if we long and short each month
port1 = port1.query('signal in(["L","S"])').groupby("date").apply(
                          lambda x: pd.Series({"ewret": sum(np.where(x.signal == "L", x.ewret, x.ewret*-1)),
                                               "vwret": sum(np.where(x.signal == "L", x.vwret, x.vwret*-1))}))


# In[195]:


long1_short2 = port1[['ewret']].describe(percentiles = probs).T.drop(columns = ['count','max','min']).apply(lambda x:x*100)


# In[ ]:


long1_short2.insert(loc = 2)


# In[181]:


port_long = pd.DataFrame(divData.query('signal == "L"').groupby('date')['ret'].mean())


# In[184]:


port_short = pd.DataFrame(NoDivData.groupby('date')['ret'].mean())


# In[191]:


portfolio2 = port_long.join(port_short,lsuffix = '_long',rsuffix = '_short') >> mutate(monthly_ret = X.ret_long - X.ret_short)


# In[194]:


long1_short3 = portfolio2[['monthly_ret']].describe(percentiles = probs).T.drop(columns = ['count','max','min']).apply(lambda x:x*100)


# In[208]:


table2PanelB


# In[204]:


long1_short2['index'] = 'long1_short2'


# In[210]:


long1_short2.set_index('index',inplace = True)


# In[212]:


long1_short2


# In[213]:


long1_short3['index'] = 'long1_short3'


# In[214]:


long1_short3.set_index('index',inplace = True)


# In[215]:


long1_short3


# In[217]:


table2PanelB = pd.concat([table2PanelB,long1_short2,long1_short3])


# In[219]:


table2PanelB.insert(loc= 2,column = 'sharp_ratio',value = (table2PanelB['mean']-0.297)/table2PanelB['std'])


# In[220]:


table2PanelB


# In[222]:


portfolio2['cumulative_ret'] = np.cumprod(1+portfolio2['monthly_ret'])


# In[227]:


port1['cumulative_ret_ewret'] = np.cumprod(1+port1['ewret'])


# In[228]:


port1.tail()


# In[224]:


portfolio2.tail()


# In[ ]:




