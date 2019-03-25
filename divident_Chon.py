import os
from dfply import *
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import feather as ft
import statsmodels.api as sm
os.chdir(r"c:\Users\ChonWai\Desktop\quantitive_trading\final_assignment")


def end_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    return next_month - datetime.timedelta(days=next_month.day)


def weighted_average(dataframe):
    return (dataframe.ret * dataframe.mcap_lag).sum() / dataframe.mcap_lag.sum()


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
mse = mse >> mask(X.divamt == X.divamt)
mse["date"] = [end_of_month(date) for date in mse["exdt"]]
# import Fama French 4 factors and 5 factors dataset
ff4 = ft.read_dataframe("ff_four_factor.feather")
ff4 = ff4.iloc[1:]
ff4["dt"] = pd.to_datetime(ff4["dt"])
ff4 = ff4.rename(columns={"dt": "date"})
ff4["date"] = ff4.agg({"date": [end_of_month]})
ff4_monthly = ff4.groupby(["date"]).apply(lambda x: pd.Series({"mkt_rf": (x.mkt_rf+1).prod()-1,
                                                               "smb": (x.SMB+1).prod()-1,
                                                               "hml": (x.HML+1).prod()-1,
                                                               "rf": (x.RF+1).prod()-1,
                                                               "mom": (x.Mom+1).prod()-1}))
ff4_monthly["ewff"] = (ff4_monthly.mkt_rf + ff4_monthly.smb + ff4_monthly.hml + ff4_monthly.mom)/4
ff4_monthly.reset_index(inplace=True)
ff4_monthly = ff4_monthly.assign(cumulative_mkt_rf=np.cumprod(1 + ff4_monthly.mkt_rf),
                                 cumulative_smb=np.cumprod(1 + ff4_monthly.smb),
                                 cumulative_hml=np.cumprod(1 + ff4_monthly.hml),
                                 cumulative_rf=np.cumprod(1 + ff4_monthly.rf),
                                 cumulative_mom=np.cumprod(1 + ff4_monthly.mom),
                                 cumulative_ewff=np.cumprod(1 + ff4_monthly.ewff))
ff5 = ft.read_dataframe("ff5.feather")
ff5["dt"] = pd.to_datetime(ff5["dt"])
ff5 = ff5.rename(columns={"dt": "date"})
ff5["date"] = ff5.agg({"date": [end_of_month]})
ff5_monthly = ff5.groupby(["date"]).apply(lambda x: pd.Series({"mkt_rf": (x.mkt_rf+1).prod()-1,
                                                               "smb": (x.smb+1).prod()-1,
                                                               "hml": (x.hml+1).prod()-1,
                                                               "rf": (x.rf+1).prod()-1,
                                                               "rmw": (x.rmw+1).prod()-1,
                                                               "cma": (x.cma+1).prod()-1}))
ff5_monthly["ewff"] = (ff5_monthly.mkt_rf + ff5_monthly.smb + ff5_monthly.hml + ff5_monthly.rmw + ff5_monthly.cma)/5
ff5_monthly.reset_index(inplace=True)
ff5_monthly = ff5_monthly.assign(cumulative_mkt_rf=np.cumprod(1 + ff5_monthly.mkt_rf),
                                 cumulative_smb=np.cumprod(1 + ff5_monthly.smb),
                                 cumulative_hml=np.cumprod(1 + ff5_monthly.hml),
                                 cumulative_rf=np.cumprod(1 + ff5_monthly.rf),
                                 cumulative_rmw=np.cumprod(1 + ff5_monthly.rmw),
                                 cumulative_cma=np.cumprod(1 + ff5_monthly.cma),
                                 cumulative_ewff=np.cumprod(1 + ff5_monthly.ewff))
# merge mse and msf
data = pd.merge(msf, mse, how="left", on=["permno", "date"])
data["mcap"] = data["prc"] * data["shrout"]
data = data >> arrange(X.permno, X.date)
# fill na with distcd if the stock has distcd before so that we can identify stock that pay dividend
# and stock that don't pay
data["distcd"] = data.groupby(["permno"])["distcd"].transform(lambda x: x.fillna(method="ffill"))
# take out the first two digit of the dist code which represent what kind of dividend is pay during the distrubtion
data["divtype"] = [i if i != i else str(i)[2] for i in data["distcd"]]
# take out the thrid digit of the dist code which represent the how often a company pay dividend
data["disttype"] = [i if i != i else str(i)[:2] for i in data["distcd"]]

# create lag variable

data = data.assign(div_lag3=data.groupby(["permno"])["divamt"].shift(3),
                   div_lag6=data.groupby(["permno"])["divamt"].shift(6),
                   div_lag12=data.groupby(["permno"])["divamt"].shift(12),
                   div_lag1=data.groupby(["permno"])["divamt"].shift(1),
                   div_lag4=data.groupby(["permno"])["divamt"].shift(4),
                   div_lag7=data.groupby(["permno"])["divamt"].shift(7),
                   div_lag10=data.groupby(["permno"])["divamt"].shift(10),
                   div_lag13=data.groupby(["permno"])["divamt"].shift(13),
                   prc_lag=data.groupby("permno")["prc"].shift(1),
                   mcap_lag=data.groupby(["permno"])["mcap"].shift(1))

data = data >> arrange(X.permno, X.month)
# assign a variable which indicated whether it has pay dividend in the pass 12 months
data["distcd_lag"] = data.groupby(["permno", "month"])["distcd"].shift(1)
# data set with stock that pay div and stock that don;t pay (divtype = nan)
data = data.query('date <= "2011-12-31" &'
                  'shrcd in([10,11]) &'
                  'hexcd in([1,2,3]) &'
                  'prc_lag >= 5 &'
                  'mcap_lag == mcap_lag &'
                  'ret == ret &'
                  'mcap_lag > 0 &'
                  'ret not in(["B","C"])')

data['turnover'] = data['vol']/data['shrout']
data['spread_2'] = data['ask'] - data['bid']
data["ret"] = data.ret.apply(float)

# table 1
"""
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
"""
# create a portfilio which strategy is long-short within company
# filter out stock that did not give dividend
portfilio1_data = data.query('disttype == disttype & '
                             'date > "1927-12-31" &'
                             'disttype == "12"  & '
                             'divtype in(["0","1","3","4","5"]) &'
                             'distcd_lag == distcd_lag')

portfilio1 = portfilio1_data
# assign signal
portfilio1["signal"] = np.where(((portfilio1.div_lag3.notna()) & (portfilio1.divtype.isin(["0", "1", "3"]))) |
                                ((portfilio1.div_lag6.notna()) & (portfilio1.divtype == "4")) |
                                ((portfilio1.div_lag12.notna()) & (portfilio1.divtype == "5")), "L",
                                np.where(((portfilio1.div_lag3.isna()) & (portfilio1.divtype.isin(["0", "1", "3"]))) |
                                         ((portfilio1.div_lag6.isna()) & (portfilio1.divtype == "4")) |
                                         ((portfilio1.div_lag12.isna()) & (portfilio1.divtype == "5")),
                                         "S", ""))
portfilio1 = portfilio1.groupby(["date", "signal"]).apply(lambda x: pd.Series({"vwret": weighted_average(x),
                                                                               "ewret": x.ret.mean()}))
portfilio1.reset_index(inplace=True)
portfilio1 = portfilio1.query('signal in(["L","S"])').groupby("date").apply(
                          lambda x : pd.Series({"ewret": sum(np.where(x.signal == "L", x.ewret, x.ewret*-0.5)),
                                                "vwret": sum(np.where(x.signal == "L", x.vwret, x.vwret*-0.5))}))
portfilio1 = portfilio1.assign(cumulative_ewret=np.cumprod(1 + portfilio1.ewret),
                               cumulative_vwret=np.cumprod(1 + portfilio1.vwret))
# plot graph of equal weight and value weight cumulative return
x = portfilio1.index
y_ewret = np.log2(portfilio1.cumulative_ewret)
y_vwret = np.log2(portfilio1.cumulative_vwret)
plt.plot(x, y_ewret, color='red')
plt.plot(x, y_vwret, color="blue")
plt.legend(["Equal Weight Cumulative Return", "Value Weight Cumulative Return"])
plt.show()
# run a regression againist Fama French 4 factors
portfilio1_ff4 = pd.merge(portfilio1, ff4_monthly, on=["date"], how="inner")
model1_ewret_ff4 = sm.OLS.from_formula('ewret~mkt_rf + smb + hml + mom', data=portfilio1_ff4).fit()
print(model1_ewret_ff4.summary())
model1_vwret_ff4 = sm.OLS.from_formula('vwret~mkt_rf + smb + hml + mom', data=portfilio1_ff4).fit()
print(model1_vwret_ff4.summary())
# plot graph between two strategies plus benchmark


# run a regression againist Fama French 5 factors
portfilio1_ff5 = pd.merge(portfilio1, ff5_monthly, on=["date"], how="inner")
model1_ewret_ff5 = sm.OLS.from_formula('ewret~mkt_rf + smb + hml + rmw + cma', data=portfilio1_ff5).fit()
print(model1_ewret_ff5.summary())
model1_vwret_ff5 = sm.OLS.from_formula('vwret~mkt_rf + smb + hml + rmw + cma', data=portfilio1_ff5).fit()
print(model1_ewret_ff5.summary())

# create a portfilio which strategy is long_short between companies
portfilio2_data = data.query('date > "1927-12-31"')

portfilio2 = portfilio2_data
portfilio2["signal"] = np.where(((portfilio2.div_lag3.notna()) & (portfilio2.divtype.isin(["0", "1", "3"]))) |
                                ((portfilio2.div_lag6.notna()) & (portfilio2.divtype == "4")) |
                                ((portfilio2.div_lag12.notna()) & (portfilio2.divtype == "5")), "L",
                                "S")

portfilio2 = portfilio2.groupby(["date", "signal"]).apply(lambda x: pd.Series({"vwret": weighted_average(x),
                                                                               "ewret": x.ret.mean()}))
portfilio2.reset_index(inplace=True)
portfilio2 = portfilio2.query('signal in(["L","S"])').groupby("date").apply(
                          lambda x : pd.Series({"ewret": sum(np.where(x.signal == "L", x.ewret, x.ewret*-0.5)),
                                                "vwret": sum(np.where(x.signal == "L", x.vwret, x.vwret*-0.5))}))
portfilio2 = portfilio1.assign(cumulative_ewret=np.cumprod(1 + portfilio2.ewret),
                               cumulative_vwret=np.cumprod(1 + portfilio2.vwret))
# plot graph of equal weight and value weight cumulative return
x = portfilio2.index
y_ewret = np.log2(portfilio2.cumulative_ewret)
y_ewret = np.log2(portfilio2.cumulative_vwret)
plt.plot(x, y_ewret, color='red')
plt.plot(x, y_ewret, color="blue")
plt.legend(["Equal Weight Cumulative Return", "Value Weight Cumulative Return"])
plt.show()


# create portfilio which strategy short month after predicted dividend
portfilio3_data = data.query('disttype == disttype & '
                             'date > "1927-12-31" &'
                             'disttype == "12"  & '
                             'divtype in(["0","1","3","4","5"])')

portfilio3 = portfilio3_data
# assign signal
portfilio3["signal"] = np.where(((portfilio3.div_lag3.notna()) & (portfilio3.divtype.isin(["0", "1", "3"]))) |
                                ((portfilio3.div_lag6.notna()) & (portfilio3.divtype == "4")) |
                                ((portfilio3.div_lag12.notna()) & (portfilio3.divtype == "5")), "L",
                                np.where(((portfilio3.div_lag4.notna()) & (portfilio3.divtype.isin(["0", "1", "3"]))) |
                                         ((portfilio3.div_lag7.notna()) & (portfilio3.divtype == "4")) |
                                         ((portfilio3.div_lag13.notna()) & (portfilio3.divtype == "5")),
                                         "S", ""))
portfilio3 = portfilio3.groupby(["date", "signal"]).apply(lambda x: pd.Series({"vwret": weighted_average(x),
                                                                               "ewret": x.ret.mean()}))
portfilio3.reset_index(inplace=True)
portfilio3 = portfilio3.query('signal in(["L","S"])').groupby("date").apply(
                          lambda x : pd.Series({"ewret": sum(np.where(x.signal == "L", x.ewret, x.ewret*-0.5)),
                                                "vwret": sum(np.where(x.signal == "L", x.vwret, x.vwret*-0.5))}))
portfilio3 = portfilio3.assign(cumulative_ewret=np.cumprod(1 + portfilio3.ewret),
                               cumulative_vwret=np.cumprod(1 + portfilio3.vwret))
# plot graph of equal weight and value weight cumulative return
x = portfilio3.index
y1 = np.log2(portfilio3.cumulative_ewret)
y2 = np.log2(portfilio3.cumulative_vwret)
plt.plot(x, y1, color='red')
plt.plot(x, y2, color="blue")
plt.legend(["Equal Weight Cumulative Return", "Value Weight Cumulative Return"])
plt.show()




