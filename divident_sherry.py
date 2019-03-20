import os
from dfply import *
import pandas as pd
import datetime
os.chdir(r"c:\Users\ChonWai\Desktop\quantitive_trading\final_assignment")


def end_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    return next_month - datetime.timedelta(days=next_month.day)


# formatting of msf
msf = pd.read_csv("msf.csv")
msf["date"] = pd.to_datetime(msf["date"], format='%Y%m%d')
msf["date"] = [end_of_month(date) for date in msf["date"]]
msf.columns = [i.lower() for i in msf.columns]
msf["month"] = [date.month for date in msf["date"]]
msf["year"] = [date.year for date in msf["date"]]
# formatting of mse
mse = pd.read_csv("mse.csv")
mse.columns = [i.lower() for i in mse.columns]
mse["exdt"] = pd.to_datetime(mse["exdt"], format='%Y%m%d')
mse["dclrdt"] = pd.to_datetime(mse["dclrdt"], format='%Y%m%d')
mse["paydt"] = pd.to_datetime(mse["paydt"], format='%Y%m%d')
mse = mse >> mask(X.divamt == X.divamt) ## dropna subset == divamt
mse["date"] = [end_of_month(date) for date in mse["exdt"]]
# merge mse and msf
data = pd.merge(msf, mse, how="outer", on=["permno", "date"])
data["mcap"] = data["prc"] * data["shrout"]
# fill na with distcd if the stock has distcd before so that we can identify stock that pay dividend
# and stock that don't pay
data["distcd"] = data.groupby("permno")["distcd"].transform(lambda x: x.fillna(method="ffill"))
# take out the first two digit of the dist code which represent what kind of dividend is pay during the distrubtion
data["divtype"] = [i if i != i else str(i)[2] for i in data["distcd"]]
# take out the thrid digit of the dist code which represent the how often a company pay dividend
data["disttype"] = [i if i != i else str(i)[:2] for i in data["distcd"]]
data = data >> arrange(X.permno, X.date)
data["prc_lag"] = data.groupby("permno")["prc"].shift(1)
# data set with stock that pay div and stock that don;t pay (divtype = nan)
data = data.query('date <= "2011-12-31" &'
                  'shrcd in([10,11]) &'
                  'hexcd in([1,2,3]) &'
                  'prc_lag >= 5 &'
                  'ret not in(["B","C"]) &'
                  '(disttype == "12" | disttype != disttype) &'
                  '(divtype in(["0","1","3","4","5"])| divtype != divtype)')
data['turnover'] = data['vol']/data['shrout']
data['spread_2'] = data['ask'] -data['bid']
regular_div = data.query('divtype == divtype')
table1_first_part = regular_div[["mcap", "spread_2", "permno","turnover"]].describe()
table1_first_part = table1_first_part.transpose()
table1_second_part = pd.DataFrame([[len(regular_div), "", "", "", "", "", "", ""],
                                  [len(set(regular_div.permno)), "", "", "", "", "", "", ""]],
                                  columns=list(table1_first_part))
table1_second_part = table1_second_part.rename(index={0: "Number of Firms months",
                                                      1: "Number of Firms"})
table1 = pd.concat((table1_first_part, table1_second_part), axis=0)
table1 = table1.rename(index={0: "Number of Firm Month", 1: "Number of Firm"},
                       columns={"count": "N"})
data['freq'] = data.distcd.apply(lambda x:str(x)[:3])
divData = data.dropna(subset = ['freq'])
## bid_ask spread
divData_spread = Data.dropna(subset = ['spread'])
divData_spread.groupby('permno')['spread'].mean()
divData_spread['spread']











