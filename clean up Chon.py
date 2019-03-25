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


def regression_and_plot(portfilio):
    """
     :param portfilio: portfilio that need to be run the regression and plot graph
     :return: run the regression for the input portfilio on Fama French 4 factors and 5 factors on equal weight return
     and value weight return , then plot the comparsion return comparsion graph as well as the graph compare Fama French
     4 and 5 factors equal weight cumulative return as well as smb and hml factor cumulative return, then return a tuble
     which contain the regression paramaters for each regression with the following order: equal weight return
     on Fama French 4 factors, value weight return on Fama French 4 factors, equal weight return on Fama French 5 factors
     and lastly value weight return on Fama French 5 factors
     """
    # run regression on Fama French four factors
    portfilio_ff4 = pd.merge(portfilio, ff4_monthly, on=["date"], how="inner")
    # equal weight return against Fama French four factors
    model_ewret_ff4 = sm.OLS.from_formula('ewret~mkt_rf + smb + hml + mom', data=portfilio_ff4).fit()
    print(model_ewret_ff4.summary())
    ewret_ff4_params = model_ewret_ff4.params
    # value weight return against Fama French four factors
    model_vwret_ff4 = sm.OLS.from_formula('vwret~mkt_rf + smb + hml + mom', data=portfilio_ff4).fit()
    print(model_vwret_ff4.summary())
    vwret_ff4_params = model_vwret_ff4.params
    # run regression on Fama French five factors
    portfilio_1963 = portfilio.query('date > "1963-07-01"')
    portfilio_1963 = portfilio_1963.assign(cumulative_ewret=np.cumprod(1 + portfilio_1963.ewret),
                                           cumulative_vwret=np.cumprod(1 + portfilio_1963.vwret))
    portfilio_ff5 = pd.merge(portfilio_1963, ff5_monthly, on=["date"], how="inner")
    # equal weight return against Fama French five factors
    model_ewret_ff5 = sm.OLS.from_formula('ewret~mkt_rf + smb + hml + rmw + cma', data=portfilio_ff5).fit()
    print(model_ewret_ff5.summary())
    ewret_ff5_params = model_ewret_ff5.params
    # value weight return against Fama French five factors
    model_vwret_ff5 = sm.OLS.from_formula('vwret~mkt_rf + smb + hml + rmw + cma', data=portfilio_ff5).fit()
    print(model_vwret_ff5.summary())
    vwret_ff5_params = model_vwret_ff5.params
    # Comparsion of equal weight and value weight cummulative return
    x = portfilio.index
    y_ewret = portfilio.cumulative_ewret
    y_vwret = portfilio.cumulative_vwret
    plt.plot(x, y_ewret, color='red')
    plt.plot(x, y_vwret, color="blue")
    plt.legend(["Equal Weight Cumulative Return", "Value Weight Cumulative Return"])
    plt.ylabel("Cumulative Return", fontdict={"fontweight": "bold"})
    plt.xlabel("Date", fontdict={"fontweight": "bold"})
    plt.title("Comparsion of Equal Weight and Value Weight Cummulative Return",
              fontdict={"fontstyle": "italic", "fontweight": "bold"})
    plt.show()
    plt.figure()
    # Comparsion of equal weight and value weight cummulative return plus benchmark which is equal weighted
    # Fama French four factors
    x = portfilio_ff4.date
    y_ewret = portfilio_ff4.cumulative_ewret
    y_vwret = portfilio_ff4.cumulative_vwret
    y_ewff = portfilio_ff4.cumulative_ewff
    plt.plot(x, y_ewret, color='red')
    plt.plot(x, y_vwret, color="blue")
    plt.plot(x, y_ewff, color="gold")
    plt.legend(["Equal Weight Cumulative Return", "Value Weight Cumulative Return", "Benchmark"])
    plt.ylabel("Cumulative Return", fontdict={"fontweight": "bold"})
    plt.xlabel("Date", fontdict={"fontweight": "bold"})
    plt.title("Comparsion of Equal Weight and Value Weight Cummulative Return Plus FF4 Benchmark",
              fontdict={"fontstyle": "italic", "fontweight": "bold"})
    plt.show()
    plt.figure()
    # Comparsion of equal weight and value weight cummulative return plus benchmark which is equal weighted
    # Fama French four factors
    x = portfilio_ff4.date
    y_ewret = portfilio_ff4.cumulative_ewret
    y_vwret = portfilio_ff4.cumulative_vwret
    y_smb = portfilio_ff4.cumulative_smb
    y_hml = portfilio_ff4.cumulative_hml
    plt.plot(x, y_ewret, color='red')
    plt.plot(x, y_vwret, color="blue")
    plt.plot(x, y_smb, color="gold")
    plt.plot(x, y_hml, color="black")
    plt.legend(["Equal Weight Cumulative Return", "Value Weight Cumulative Return", "SMB", "HML"])
    plt.ylabel("Cumulative Return", fontdict={"fontweight": "bold"})
    plt.xlabel("Date", fontdict={"fontweight": "bold"})
    plt.title("Comparsion of Equal Weight and Value Weight Cummulative Return plus SML and HML Factors ",
              fontdict={"fontstyle": "italic", "fontweight": "bold"})
    plt.show()
    plt.figure()
    # Comparsion of equal weight and value weight cummulative return plus benchmark which is equal weighted
    # Fama French five factors
    x = portfilio_ff5.date
    y_ewret = portfilio_ff5.cumulative_ewret
    y_vwret = portfilio_ff5.cumulative_vwret
    y_ewff = portfilio_ff5.cumulative_ewff
    plt.plot(x, y_ewret, color='red')
    plt.plot(x, y_vwret, color="blue")
    plt.plot(x, y_ewff, color="gold")
    plt.legend(["Equal Weight Cumulative Return", "Value Weight Cumulative Return", "Benchmark"])
    plt.ylabel("Cumulative Return", fontdict={"fontweight": "bold"})
    plt.xlabel("Date", fontdict={"fontweight": "bold"})
    plt.title("Comparsion of Equal Weight and Value Weight Cummulative Return Plus FF5 Benchmark",
              fontdict={"fontstyle": "italic", "fontweight": "bold"})
    plt.show()
    return ewret_ff4_params, vwret_ff4_params, ewret_ff5_params, vwret_ff5_params


if __name__ == '__main__':
    # sherry code should insert right below this

    # import Fama French 4 factors
    ff4 = ft.read_dataframe("ff_four_factor.feather")
    ff4 = ff4.iloc[1:]
    ff4["dt"] = pd.to_datetime(ff4["dt"])
    ff4 = ff4.rename(columns={"dt": "date"})
    ff4["date"] = ff4.agg({"date": [end_of_month]})
    # convert the daily file to monthly file by cumulating return on each day base on thier year and month
    ff4_monthly = ff4.groupby(["date"]).apply(lambda x: pd.Series({"mkt_rf": (x.mkt_rf+1).prod()-1,
                                                                   "smb": (x.SMB+1).prod()-1,
                                                                   "hml": (x.HML+1).prod()-1,
                                                                   "rf": (x.RF+1).prod()-1,
                                                                   "mom": (x.Mom+1).prod()-1}))

    ff4_monthly["ewff"] = (ff4_monthly.mkt_rf + ff4_monthly.smb + ff4_monthly.hml + ff4_monthly.mom)/4
    ff4_monthly.reset_index(inplace=True)
    # convert the daily file to monthly file by cumulating return on each day base on thier year and month
    ff4_monthly = ff4_monthly.assign(cumulative_mkt_rf=np.cumprod(1 + ff4_monthly.mkt_rf),
                                     cumulative_smb=np.cumprod(1 + ff4_monthly.smb),
                                     cumulative_hml=np.cumprod(1 + ff4_monthly.hml),
                                     cumulative_rf=np.cumprod(1 + ff4_monthly.rf),
                                     cumulative_mom=np.cumprod(1 + ff4_monthly.mom),
                                     cumulative_ewff=np.cumprod(1 + ff4_monthly.ewff))
    # import Fama French 5 factors dataset
    ff5 = ft.read_dataframe("ff5.feather")
    ff5["dt"] = pd.to_datetime(ff5["dt"])
    ff5 = ff5.rename(columns={"dt": "date"})
    ff5["date"] = ff5.agg({"date": [end_of_month]})
    # convert the daily file to monthly file by cumulating return on each day base on thier year and month
    ff5_monthly = ff5.groupby(["date"]).apply(lambda x: pd.Series({"mkt_rf": (x.mkt_rf+1).prod()-1,
                                                                   "smb": (x.smb+1).prod()-1,
                                                                   "hml": (x.hml+1).prod()-1,
                                                                   "rf": (x.rf+1).prod()-1,
                                                                   "rmw": (x.rmw+1).prod()-1,
                                                                   "cma": (x.cma+1).prod()-1}))
    # create a benchmark which equal weight all five factors of Fama French 5 factos
    ff5_monthly["ewff"] = (ff5_monthly.mkt_rf + ff5_monthly.smb + ff5_monthly.hml + ff5_monthly.rmw + ff5_monthly.cma)/5
    ff5_monthly.reset_index(inplace=True)
    # create a cumulative return over time for all factos plus benchmark
    ff5_monthly = ff5_monthly.assign(cumulative_mkt_rf=np.cumprod(1 + ff5_monthly.mkt_rf),
                                     cumulative_smb=np.cumprod(1 + ff5_monthly.smb),
                                     cumulative_hml=np.cumprod(1 + ff5_monthly.hml),
                                     cumulative_rf=np.cumprod(1 + ff5_monthly.rf),
                                     cumulative_rmw=np.cumprod(1 + ff5_monthly.rmw),
                                     cumulative_cma=np.cumprod(1 + ff5_monthly.cma),
                                     cumulative_ewff=np.cumprod(1 + ff5_monthly.ewff))
    # this data should be modify after sherry code import and the code until the next comment might need additional
    # adjustment after sherry code implement in this one
    data = pd.read_csv("dataUse.csv")
    data["date"] = pd.to_datetime(data["date"])
    # take out the first two digit of the dist code which represent what kind of dividend is pay during the distrubtion
    data["divtype"] = [i if i != i else str(i)[2] for i in data["freq"]]
    # take out the thrid digit of the dist code which represent the how often a company pay dividend
    data["disttype"] = [i if i != i else str(i)[:2] for i in data["freq"]]
    # create a portfilio which strategy is long-short within company
    data = data.assign(prc_lag=data.groupby("permno")["prc"].shift(1),
                       mcap_lag=data.groupby(["permno"])["mcap"].shift(1))
    # filter out stock that did not give dividend
    portfilio1_data = data.query('freq == freq & '
                                 'disttype == "12" &'
                                 'divtype in(["0","1","3","4","5"])')
    portfilio1 = portfilio1_data
    # assign signal
    portfilio1["signal"] = np.where((portfilio1.div_lag3.notna() | portfilio1.div_lag6.notna() |
                                     portfilio1.div_lag9.notna() | portfilio1.div_lag12.notna()), "L",
                                    np.where(portfilio1.div_lag1.notna() | portfilio1.div_lag2.notna() |
                                             portfilio1.div_lag4.notna() | portfilio1.div_lag5.notna() |
                                             portfilio1.div_lag7.notna() | portfilio1.div_lag8.notna() |
                                             portfilio1.div_lag10.notna() | portfilio1.div_lag11.notna(), "S", ""))
    # calculate the equal weight mean and value weight mean return base on each month and signal
    portfilio1 = portfilio1.groupby(["date", "signal"]).apply(lambda x: pd.Series({"vwret": weighted_average(x),
                                                                                   "ewret": x.ret.mean()}))
    portfilio1.reset_index(inplace=True)
    # calculate the total return of equal weight and value weight if we long and short each month
    portfilio1 = portfilio1.query('signal in(["L","S"])').groupby("date").apply(
                              lambda x: pd.Series({"ewret": sum(np.where(x.signal == "L", x.ewret, x.ewret*-1)),
                                                   "vwret": sum(np.where(x.signal == "L", x.vwret, x.vwret*-1))}))
    # calculate the cumulative return over time for equal weight and value weight
    portfilio1 = portfilio1.assign(cumulative_ewret=np.cumprod(1 + portfilio1.ewret),
                                   cumulative_vwret=np.cumprod(1 + portfilio1.vwret))
    # plot comparsion graphs and run regression against Fama French four factos and five factors on portfilio1
    # and return a tuple which contains all regression paramaters
    regression_result1 = regression_and_plot(portfilio1)
    # create a portfilio which strategy is long_short between companies
    # using the orignal data as the portfilio2 data since we are shorting everything that is not long
    portfilio2_data = data
    portfilio2 = portfilio2_data
    # assign signal
    portfilio2["signal"] = np.where((portfilio2.div_lag3.notna() | portfilio2.div_lag6.notna() |
                                     portfilio2.div_lag9.notna() | portfilio2.div_lag12.notna()), "L", "S")
    # calculate the equal weight mean and value weight mean return base on each month and signal
    portfilio2 = portfilio2.groupby(["date", "signal"]).apply(lambda x: pd.Series({"vwret": weighted_average(x),
                                                                                   "ewret": x.ret.mean()}))
    portfilio2.reset_index(inplace=True)
    # calculate the total return of equal weight and value weight if we long and short each month
    portfilio2 = portfilio2.query('signal in(["L","S"])').groupby("date").apply(
                              lambda x : pd.Series({"ewret": sum(np.where(x.signal == "L", x.ewret, x.ewret*-1)),
                                                    "vwret": sum(np.where(x.signal == "L", x.vwret, x.vwret*-1))}))
    # calculate the cumulative return over time for equal weight and value weight
    portfilio2 = portfilio2.assign(cumulative_ewret=np.cumprod(1 + portfilio2.ewret),
                                   cumulative_vwret=np.cumprod(1 + portfilio2.vwret))
    # plot comparsion graphs and run regression against Fama French four factos and five factors on portfilio2
    # and return a tuple which contains all regression paramaters
    regression_result2 = regression_and_plot(portfilio2)
    # create portfilio which strategy short month after predicted dividend
    # fiter out stock that do not pay dividend
    portfilio3_data = data.query('disttype == disttype & '
                                 'disttype == "12"  & '
                                 'divtype in(["0","1","3","4","5"])')
    portfilio3 = portfilio3_data
    # assign signal
    portfilio3["signal"] = np.where((portfilio3.div_lag3.notna() | portfilio3.div_lag6.notna() |
                                     portfilio3.div_lag9.notna() | portfilio3.div_lag12.notna()), "L",
                                    np.where((portfilio3.div_lag4.notna() | portfilio3.div_lag1.notna() |
                                              portfilio3.div_lag7.notna() | portfilio3.div_lag10.notna()), "S", ""))
    # calculate the equal weight mean and value weight mean return base on each month and signal
    portfilio3 = portfilio3.groupby(["date", "signal"]).apply(lambda x: pd.Series({"vwret": weighted_average(x),
                                                                                   "ewret": x.ret.mean()}))
    portfilio3.reset_index(inplace=True)
    # calculate the total return of equal weight and value weight if we long and short each month
    portfilio3 = portfilio3.query('signal in(["L","S"])').groupby("date").apply(
                              lambda x : pd.Series({"ewret": sum(np.where(x.signal == "L", x.ewret, x.ewret*-1)),
                                                    "vwret": sum(np.where(x.signal == "L", x.vwret, x.vwret*-1))}))
    # calculate the cumulative return over time for equal weight and value weight
    portfilio3 = portfilio3.assign(cumulative_ewret=np.cumprod(1 + portfilio3.ewret),
                                   cumulative_vwret=np.cumprod(1 + portfilio3.vwret))
    # plot comparsion graphs and run regression against Fama French four factos and five factors on portfilio1
    # and return a tuple which contains all regression paramaters
    regression_result3 = regression_and_plot(portfilio3)






