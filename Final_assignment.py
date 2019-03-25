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


def long_short_regression(long_short_portfilio):
    long_short_portfilio_ff4 = pd.merge(long_short_portfilio, ff4_monthly, on=["date"], how="inner")
    # equal weight return against CAPM which is mkt_rf factor if we just long
    model_long_ewret_capm = sm.OLS.from_formula('ewret_long~mkt_rf ', data=long_short_portfilio_ff4).fit()
    print(model_long_ewret_capm.summary())
    # equal weight return against CAPM which is mkt_rf factor if we just short
    model_short_ewret_capm = sm.OLS.from_formula('ewret_short~mkt_rf ', data=long_short_portfilio_ff4).fit()
    print(model_short_ewret_capm.summary())
    # value weight return against CAPM which is mkt_rf factor if we just long
    model_long_vwret_capm = sm.OLS.from_formula('vwret_long~mkt_rf ', data=long_short_portfilio_ff4).fit()
    print(model_long_vwret_capm.summary())
    # value weight return against CAPM which is mkt_rf factor if we just short
    model_short_vwret_capm = sm.OLS.from_formula('vwret_short~mkt_rf ', data=long_short_portfilio_ff4).fit()
    print(model_short_vwret_capm.summary())

    # equal weight return againist Fama French three factors if we just long
    model_long_ewret_ff3 = sm.OLS.from_formula('ewret_long~mkt_rf + smb + hml ', data=long_short_portfilio_ff4).fit()
    print(model_long_ewret_ff3.summary())
    # equal weight return againist Fama French three factors if we just short
    model_short_ewret_ff3 = sm.OLS.from_formula('ewret_short~mkt_rf + smb + hml ', data=long_short_portfilio_ff4).fit()
    print(model_short_ewret_ff3.summary())
    # value weight return againist Fama French three factors if we just long
    model_long_vwret_ff3 = sm.OLS.from_formula('vwret_long~mkt_rf + smb + hml ', data=long_short_portfilio_ff4).fit()
    print(model_long_vwret_ff3.summary())
    # equal weight return againist Fama French three factors if we just short
    model_short_vwret_ff3 = sm.OLS.from_formula('vwret_short~mkt_rf + smb + hml ', data=long_short_portfilio_ff4).fit()
    print(model_short_vwret_ff3.summary())

    # equal weight return against Fama French four factos if we just long
    model_long_ewret_ff4 = sm.OLS.from_formula('ewret_long~mkt_rf + smb + hml + mom ',
                                               data=long_short_portfilio_ff4).fit()
    print(model_long_ewret_ff4.summary())
    # equal weight return against Fama French four factos if we just short
    model_short_ewret_ff4 = sm.OLS.from_formula('ewret_short~mkt_rf + smb + hml + mom ',
                                                data=long_short_portfilio_ff4).fit()
    print(model_short_ewret_ff4.summary())
    # vqual weight return against Fama French four factos if we just long
    model_long_vwret_ff4 = sm.OLS.from_formula('vwret_long~mkt_rf + smb + hml + mom ',
                                               data=long_short_portfilio_ff4).fit()
    print(model_long_vwret_ff4.summary())
    # vqual weight return against Fama French four factos if we just short
    model_short_vwret_ff4 = sm.OLS.from_formula('vwret_short~mkt_rf + smb + hml + mom ',
                                                data=long_short_portfilio_ff4).fit()
    print(model_short_vwret_ff4.summary())

    long_short_portfilio_ff4_liq = pd.merge(long_short_portfilio_ff4, liq, on=["date"], how="inner")
    # equal weight return against Fama French four factos if we just long
    model_long_ewret_ff4_liq = sm.OLS.from_formula('ewret_long~mkt_rf + smb + hml + mom + ps_innov ',
                                                   data=long_short_portfilio_ff4_liq).fit()
    print(model_long_ewret_ff4_liq.summary())
    # equal weight return against Fama French four factos if we just short
    model_short_ewret_ff4_liq = sm.OLS.from_formula('ewret_short~mkt_rf + smb + hml + mom + ps_innov ',
                                                    data=long_short_portfilio_ff4_liq).fit()
    print(model_short_ewret_ff4_liq.summary())
    # vqual weight return against Fama French four factos if we just long
    model_long_vwret_ff4_liq = sm.OLS.from_formula('vwret_long~mkt_rf + smb + hml + mom + ps_innov',
                                                   data=long_short_portfilio_ff4_liq).fit()
    print(model_long_vwret_ff4_liq.summary())
    # vqual weight return against Fama French four factos if we just short
    model_short_vwret_ff4_liq = sm.OLS.from_formula('vwret_short~mkt_rf + smb + hml + mom + ps_innov ',
                                                    data=long_short_portfilio_ff4_liq).fit()
    print(model_short_vwret_ff4_liq.summary())


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


def tabel1_tabel2(dataUse):
    divData = dataUse.query('freq in(["120","121","123","124","125"])')
    # creat table1
    # 1.1  creat table1PanelA
    table1PanelApart1 = divData[['mcap', 'turnover', 'spread_2', 'Div_Yield']].describe().T
    table1PanelApart2 = divData.query('bm <500 & bm >0')[['bm']].describe().T
    table1PanelA = pd.concat([table1PanelApart1, table1PanelApart2])
    N_firms = pd.DataFrame({'count': len(set(divData['permno']))}, index=['N_firms'])
    table1PanelA = pd.concat([table1PanelA, N_firms])
    print(table1PanelA)
    # 1.2 Creat table1PanelB
    NoDivData = dataUse.query('freq not in(["120","121","123","124","125"])')
    table1PanelBpart1 = NoDivData[['mcap', 'turnover', 'spread_2']].describe().T
    table1PanelBpart2 = NoDivData.query('bm <1000&bm >0')[['bm']].describe().T
    N_firms2 = pd.DataFrame({'count': len(set(NoDivData.permno))}, index=['N_firms'])
    table1PanelB = pd.concat([table1PanelBpart1, table1PanelBpart2, N_firms2])
    print(table1PanelB)
    # Panel C column 1: Percent of firm-months with dividend in the last year
    pannel3_data = dataUse
    pannel3_data['divType'] = pannel3_data.distcd.apply(lambda x: str(x)[:3])
    DivFreq = pd.DataFrame(pannel3_data.groupby(['year', 'divType'])['divType'].count())
    DivFreq['percent'] = DivFreq.divType / DivFreq.groupby(level=0).divType.transform(np.sum)
    DivFreq.columns = ['count', 'percent']
    DivFreq = DivFreq.reset_index()
    pannel3_data['divType'] = pannel3_data.groupby('permno')['divType'].fillna(method='ffill')
    DivFreq = pd.DataFrame(DivFreq.groupby('divType')['percent'].mean())
    div_freq = ['120', '122', '123', '124', '125', '126', '127']
    Any_freq = DivFreq.loc[div_freq,].sum()
    div_freq_index = {'monthly': '122', 'quarterly': '123', 'semiAnnualy': '124', 'Annualy': '125'}
    table1PanelC_part1 = DivFreq.loc[['122', '123', '124', '125', '121'],]
    table1PanelC_part1.loc['Any_freq'] = {'percent': Any_freq.tolist()[0]}
    # Panel C-Column 2: Percent of dividend observations
    divData['divType'] = divData.distcd.apply(lambda x: str(x)[:3])
    DivFreq2 = pd.DataFrame(divData.groupby('divType')['divType'].count())
    DivFreq2.columns = ['count']
    DivFreq2['percent'] = DivFreq2['count'] / DivFreq2['count'].sum()
    table1PanelC_part2 = DivFreq2[['percent']].loc[['122', '123', '124', '125', '121'],]
    table1PanelC = pd.merge(table1PanelC_part1, table1PanelC_part2, left_index=True, right_index=True, how='outer')
    table1PanelC.rename(
        index={'121': 'unknow_freq', '122': 'monthly', '123': 'quarterly', '124': 'semiAnnualy', '125': 'Annualy'},
        inplace=True)
    table1PanelC = round(table1PanelC.apply(lambda x: x * 100), 2)
    print(table1PanelC)
    # create Table2
    # create Table2 PanelA
    table2_mean = {}
    table2_std = {}
    table2_all_div_prob = {}
    table2_quar_div_prob = {}
    index_start = divData.columns.tolist().index('div_lag1')
    index_end = divData.columns.tolist().index('div_lag12')
    div_lag_list = divData.columns[index_start:index_end + 1].tolist()
    for i, div_lag in enumerate(div_lag_list):
        table2_mean[str(i + 1)] = divData.dropna(subset=[div_lag])['ret'].mean() * 100
        table2_std[str(i + 1)] = divData.dropna(subset=[div_lag])['ret'].std() * 100
        table2_all_div_prob[str(i + 1)] = len(divData.dropna(subset=[div_lag, 'divamt'])) / len(
            divData.dropna(subset=[div_lag]))
        divData_quar = divData.query('freq == "123"')
        table2_quar_div_prob[str(i + 1)] = len(divData_quar.dropna(subset=[div_lag, 'divamt'])) / len(
            divData_quar.dropna(subset=[div_lag]))
    table2_panelA = pd.DataFrame(pd.Series(table2_mean), columns=['Mean return'])
    table2_panelA['Standard Deviation'] = table2_std.values()
    table2_panelA['All dividends'] = table2_all_div_prob.values()
    table2_panelA['quar dividends'] = table2_quar_div_prob.values()
    print(table2_panelA)
    # 2.2 creat table2 PanelB
    # assign signal
    divData["signal"] = np.where((divData.div_lag3.notna() | divData.div_lag6.notna() |
                                  divData.div_lag9.notna() | divData.div_lag12.notna()), "L",
                                 np.where(divData.div_lag1.notna() | divData.div_lag2.notna() |
                                          divData.div_lag4.notna() | divData.div_lag5.notna() |
                                          divData.div_lag7.notna() | divData.div_lag8.notna() |
                                          divData.div_lag10.notna() | divData.div_lag11.notna(), "S", ""))
    portfolio1 = pd.DataFrame(divData.groupby(["date", "signal"])['ret'].mean())
    probs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    table2PanelB_part1 = portfolio1.groupby('signal')['ret'].describe(percentiles=probs).loc[['L', 'S']].drop(
        columns=['count', 'max', 'min']).apply(lambda x: x * 100)
    table2PanelB_part2 = pd.DataFrame(NoDivData.groupby('date')['ret'].mean()). \
        describe(percentiles=probs).T.drop(columns=['count', 'min', 'max']).apply(lambda x: x * 100)
    table2PanelB = pd.concat([table2PanelB_part1, table2PanelB_part2])
    table2PanelB['index'] = ['div_predM', 'div_NoPredM', 'NoDiv']
    table2PanelB.set_index('index', inplace=True)
    port1 = divData.groupby(["date", "signal"]).apply(lambda x: pd.Series({"vwret": weighted_average(x),
                                                                           "ewret": x.ret.mean()}))
    port1.reset_index(inplace=True)
    # calculate the total return of equal weight and value weight if we long and short each month
    port1 = port1.query('signal in(["L","S"])').groupby("date").apply(
        lambda x: pd.Series({"ewret": sum(np.where(x.signal == "L", x.ewret, x.ewret * -1)),
                             "vwret": sum(np.where(x.signal == "L", x.vwret, x.vwret * -1))}))
    long1_short2 = port1[['ewret']].describe(percentiles=probs).T.drop(
        columns=['count', 'max', 'min']).apply(lambda x: x * 100)
    port_long = pd.DataFrame(divData.query('signal == "L"').groupby('date')['ret'].mean())
    port_short = pd.DataFrame(NoDivData.groupby('date')['ret'].mean())
    portfolio2 = port_long.join(port_short, lsuffix='_long', rsuffix='_short') >> mutate(
        monthly_ret=X.ret_long - X.ret_short)
    long1_short3 = portfolio2[['monthly_ret']].describe(percentiles=probs).T.drop(columns=['count', 'max', 'min']). \
        apply(lambda x: x * 100)
    long1_short2['index'] = 'long1_short2'
    long1_short2.set_index('index', inplace=True)
    long1_short3['index'] = 'long1_short3'
    long1_short3.set_index('index', inplace=True)
    table2PanelB = pd.concat([table2PanelB, long1_short2, long1_short3])
    table2PanelB.insert(loc=2, column='sharp_ratio', value=(table2PanelB['mean'] - 0.297) / table2PanelB['std'])
    portfolio2['cumulative_ret'] = np.cumprod(1 + portfolio2['monthly_ret'])
    print(table2PanelB)


if __name__ == '__main__':
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
    # import the liquidity data set which is the Pastor_Stambaugh_liquidity from crsp but did not cover all of our
    # portfilio investment horizon
    liq = pd.read_csv("Pastor_Stambaugh_liquidity.csv")
    liq.columns = [i.lower() for i in liq.columns]
    liq["date"] = liq.date.apply(lambda x: str(x))
    liq["date"] = pd.to_datetime(liq["date"])
    # import the msf dataset and formatting columns
    msf = pd.read_csv("msf.csv")
    msf["date"] = pd.to_datetime(msf["date"], format='%Y%m%d')
    msf["date"] = [end_of_month(date) for date in msf["date"]]
    msf.columns = [i.lower() for i in msf.columns]
    msf["month"] = [date.month for date in msf["date"]]
    msf["year"] = msf.date.apply(lambda x: x.year)
    # import the msf dataset and formatting columns
    mse = pd.read_csv("mse.csv")
    mse.columns = [i.lower() for i in mse.columns]
    mse["exdt"] = pd.to_datetime(mse["exdt"], format='%Y%m%d')
    mse["dclrdt"] = pd.to_datetime(mse["dclrdt"], format='%Y%m%d')
    mse["paydt"] = pd.to_datetime(mse["paydt"], format='%Y%m%d')
    # filter out data which have a na value for dividend columns
    mse = mse >> mask(X.divamt == X.divamt)
    mse['date'] = mse['date'] = mse.exdt.agg(end_of_month)
    mse['freq'] = mse.distcd.apply(lambda x: str(x)[:3])
    mse['freq'] = mse.freq.astype('int64')
    mse['freq'].loc[mse['freq'] <= 123] = 123
    mseUse = pd.DataFrame(mse.groupby(['permno', 'date', 'distcd', 'freq'])['divamt'].sum()).reset_index()
    # merging msf and mse
    data = pd.merge(msf, mseUse, how='left', on=['permno', 'date'])
    data.pop('bidlo')
    data.pop('askhi')
    data["mcap"] = data["prc"] * data["shrout"]
    # fill na with distcd if the stock has distcd before so that we can identify stock that pay dividend
    data = data >> arrange(X.permno, X.date)
    data['freq'] = data.groupby('permno')['freq'].fillna(method='ffill')
    data['distcd'] = data.groupby('permno')['distcd'].fillna(method='ffill')
    data = data.drop_duplicates(subset=['permno', 'date'])
    # creating lag variables
    data.sort_values(['permno', 'date'], inplace=True)
    data['prc_lag1'] = data.groupby('permno')['prc'].shift(1)
    data['mcap_lag1'] = data.groupby('permno')['mcap'].shift(1)
    data['freq_lag1'] = data.groupby('permno')['freq'].shift(1)
    data['turnover'] = data['vol'] / data['shrout']
    data['spread_2'] = data['ask'] - data['bid']
    data['Div_Yield'] = data['divamt'] / data['prc']
    for i in range(12):
        div_lag = 'div_lag' + str(i + 1)
        data[div_lag] = data.groupby('permno')['divamt'].shift(i + 1)
    # import gvkey permno linktable
    linkTable = ft.read_dataframe('monthly_gvkey_permno_link.feather')
    linkTable['date'] = pd.to_datetime(linkTable['date'])
    linkTable['year'] = linkTable.date.apply(lambda x: x.year)
    linkTable.drop_duplicates(subset=['gvkey', 'permno', 'year'], inplace=True)
    linkTable['N_permno'] = linkTable.groupby(['year', 'gvkey'])['permno'].transform('count')
    # import the book market data set
    BMdata = pd.read_csv('BMdata.csv')
    BMdata['datadate'] = pd.to_datetime(BMdata.datadate, format='%Y%m%d')
    BMdata = BMdata[['gvkey', 'datadate', 'bkvlps']]
    BMdata.rename(columns={'datadate': 'date'}, inplace=True)
    BMdata['year'] = BMdata.date.apply(lambda x: x.year)
    # merge the BM data to linktable data set
    BMdata_permno = BMdata.merge(linkTable, on=['gvkey', 'year'], how='left', suffixes=['_BM', '_link'])
    BMdata_permno.dropna(subset=['bkvlps'], inplace=True)
    BMdata_permno['permno'] = BMdata_permno.groupby('gvkey')['permno'].fillna(method='ffill')
    BMdata_permno.drop_duplicates(subset=['permno', 'date_BM', 'bkvlps'], inplace=True)
    BMdata_permno.dropna(subset=['permno'], inplace=True)
    BMdata_permno['N_permno'] = BMdata_permno.groupby(['permno', 'date_BM'])['permno'].transform('count')
    BMuse = pd.DataFrame(BMdata_permno.groupby(['permno', 'date_BM'])['bkvlps'].mean()).reset_index()
    BMdata_permno.rename(columns={'date_BM': 'date'}, inplace=True)
    # merging the BM to msf and mse
    data = pd.merge(data, BMdata_permno.drop(columns=['year']), on=['permno', 'date'], how='left')
    data = data >> arrange(X.permno, X.date)
    data['bkvlps'] = data.groupby('permno')['bkvlps'].fillna(method='ffill')
    data['bm'] = data['bkvlps'] / data['prc']
    # change the name to data later on
    data = data.query('date <= "2011-12-31" &'
                      'shrcd in([10,11]) &'
                      'hexcd in([1,2,3]) &'
                      'prc_lag1 >= 5 &'
                      'ret not in(["B","C"])')
    data['ret'] = data.ret.astype('float')
    data["date"] = pd.to_datetime(data["date"])
    # take out the first two digit of the dist code which represent what kind of dividend is pay during the distrubtion
    data["divtype"] = [i if i != i else str(i)[2] for i in data["freq"]]
    # take out the thrid digit of the dist code which represent the how often a company pay dividend
    data["disttype"] = [i if i != i else str(i)[:2] for i in data["freq"]]
    data = data.rename(columns={"mcap_lag1": "mcap_lag", "prc_lag1": "prc_lag"})
    data = data.query('mcap_lag == mcap_lag & ret == ret')

    # print table 1 and table 2
    tabel1_tabel2(data)

    # create a portfilio which strategy is long-short within company
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
    # calculate the value weight and equal weight return if we long or short only
    portfilio1_long_or_short_only = portfilio1.query('signal in(["L","S"])').\
        groupby("date").apply(lambda x: pd.Series({"ewret_long": sum(np.where(x.signal == "L", x.ewret, 0)),
                                                   "vwret_long": sum(np.where(x.signal == "L", x.vwret, 0)),
                                                   "ewret_short": sum(np.where(x.signal == "S", x.ewret, 0)),
                                                   "vwret_short": sum(np.where(x.signal == "S", x.vwret, 0))}))
    portfilio1_long_or_short_only = portfilio1_long_or_short_only.replace(0, np.nan)
    # calculate the cumulative return for both equal wieght and value weight for both long and short only
    portfilio1_long_or_short_only = portfilio1_long_or_short_only.assign(
        cumulative_ewret_long=np.cumprod(1 + portfilio1_long_or_short_only.ewret_long),
        cumulative_vwret_long=np.cumprod(1 + portfilio1_long_or_short_only.vwret_long),
        cumulative_ewret_short=np.cumprod(1 + portfilio1_long_or_short_only.ewret_short),
        cumulative_vwret_short=np.cumprod(1 + portfilio1_long_or_short_only.vwret_short))
    # run various regression on long only and short only
    long_short_regression(portfilio1_long_or_short_only)
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
    # calculate the value weight and equal weight return if we long or short only
    portfilio2_long_or_short_only = portfilio2.query('signal in(["L","S"])'). \
        groupby("date").apply(lambda x: pd.Series({"ewret_long": sum(np.where(x.signal == "L", x.ewret, 0)),
                                                   "vwret_long": sum(np.where(x.signal == "L", x.vwret, 0)),
                                                   "ewret_short": sum(np.where(x.signal == "S", x.ewret, 0)),
                                                   "vwret_short": sum(np.where(x.signal == "S", x.vwret, 0))}))
    portfilio2_long_or_short_only = portfilio2_long_or_short_only.replace(0, np.nan)
    # calculate the cumulative return for both equal wieght and value weight for both long and short only
    portfilio2_long_or_short_only = portfilio2_long_or_short_only.assign(
        cumulative_ewret_long=np.cumprod(1 + portfilio2_long_or_short_only.ewret_long),
        cumulative_vwret_long=np.cumprod(1 + portfilio2_long_or_short_only.vwret_long),
        cumulative_ewret_short=np.cumprod(1 + portfilio2_long_or_short_only.ewret_short),
        cumulative_vwret_short=np.cumprod(1 + portfilio2_long_or_short_only.vwret_short))
    # run various regression on long only and short only
    long_short_regression(portfilio2_long_or_short_only)
    # calculate the total return of equal weight and value weight if we long and short each month
    portfilio2 = portfilio2.query('signal in(["L","S"])').groupby("date").apply(
                              lambda x: pd.Series({"ewret": sum(np.where(x.signal == "L", x.ewret, x.ewret*-1)),
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
    # calculate the value weight and equal weight return if we long or short only
    portfilio3_long_or_short_only = portfilio3.query('signal in(["L","S"])'). \
        groupby("date").apply(lambda x: pd.Series({"ewret_long": sum(np.where(x.signal == "L", x.ewret, 0)),
                                                   "vwret_long": sum(np.where(x.signal == "L", x.vwret, 0)),
                                                   "ewret_short": sum(np.where(x.signal == "S", x.ewret, 0)),
                                                   "vwret_short": sum(np.where(x.signal == "S", x.vwret, 0))}))
    portfilio3_long_or_short_only = portfilio3_long_or_short_only.replace(0, np.nan)
    # calculate the cumulative return for both equal wieght and value weight for both long and short only
    portfilio3_long_or_short_only = portfilio3_long_or_short_only.assign(
        cumulative_ewret_long=np.cumprod(1 + portfilio3_long_or_short_only.ewret_long),
        cumulative_vwret_long=np.cumprod(1 + portfilio3_long_or_short_only.vwret_long),
        cumulative_ewret_short=np.cumprod(1 + portfilio3_long_or_short_only.ewret_short),
        cumulative_vwret_short=np.cumprod(1 + portfilio3_long_or_short_only.vwret_short))
    # run various regression on long only and short only
    long_short_regression(portfilio3_long_or_short_only)
    # calculate the total return of equal weight and value weight if we long and short each month
    portfilio3 = portfilio3.query('signal in(["L","S"])').groupby("date").apply(
                              lambda x: pd.Series({"ewret": sum(np.where(x.signal == "L", x.ewret, x.ewret*-1)),
                                                   "vwret": sum(np.where(x.signal == "L", x.vwret, x.vwret*-1))}))
    # calculate the cumulative return over time for equal weight and value weight
    portfilio3 = portfilio3.assign(cumulative_ewret=np.cumprod(1 + portfilio3.ewret),
                                   cumulative_vwret=np.cumprod(1 + portfilio3.vwret))
    # plot comparsion graphs and run regression against Fama French four factos and five factors on portfilio1
    # and return a tuple which contains all regression paramaters
    regression_result3 = regression_and_plot(portfilio3)





