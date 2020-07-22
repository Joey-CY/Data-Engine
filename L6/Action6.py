# coding=utf-8
import pandas as pd


def rule1():
    from fbpropeht import Prophet
    train = pd.read_csv("./train.csv")
    train["Datetime"] = pd.to_datetime(train.Datetime, format="%d-%m-%Y %H:%M")
    train.index = train.Datetime
    train.drop(["ID", "Datetime"], axis=1, inplace=True)
    daily_train = train.resample("D").sum()
    daily_train["ds"] = daily_train.index
    daily_train["y"] = daily_train.Count
    daily_train.drop(["Count"], axis=1, inplace=True)
    m = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
    m.fit(daily_train)
    future = m.make_future_dataframe(periods=213)
    forecast = m.predict(future)
    m.plot(forecast)
    m.plot_components(forecast)


def rule2():
    from statsmodels.tsa.arima_model import ARIMA
    from itertools import product
    import statsmodels.api as sm
    from datetime import datetime, timedelta
    import calendar
    import matplotlib.pyplot as plt

    df = pd.read_csv("./train.csv")
    df.Datetime = pd.to_datetime(df.Datetime)
    df.index = df.Datetime
    df["y"] = df.Count
    df.drop(["ID", "Datetime"], axis=1, inplace=True)
    df_month = df.resample("M").count()
    df_month.drop(["Count"], axis=1, inplace=True)
    print(df_month)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    ps = range(0, 5)
    qs = range(0, 5)
    ds = range(1, 2)
    parameters = product(ps, qs, ds)
    parameters_list = list(parameters)
    results = []
    best_aic = float("inf")
    for param in parameters_list:
        try:
            model = sm.tsa.statespace.SARIMAX(df_month.y, order=(param[0], param[1], param[2]),
                                              enforce_stationarity=False, enforce_invertibility=False).fit()
        except ValueError:
            print('ValueError:', param)
            continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
    print('Best Model: ', best_model.summary())

    df_month2 = df_month[['y']]
    future_month = 3
    last_month = pd.to_datetime(df_month2.index[len(df_month2) - 1])
    date_list = []
    for i in range(future_month):
        # 计算下个月有多少天
        year = last_month.year
        month = last_month.month
        if month == 12:
            month = 1
            year = year + 1
        else:
            month = month + 1
        next_month_days = calendar.monthrange(year, month)[1]
        # print(next_month_days)
        last_month = last_month + timedelta(days=next_month_days)
        date_list.append(last_month)
    print('date_list=', date_list)

    # 添加未来要预测的3个月
    future = pd.DataFrame(index=date_list, columns=df_month.columns)
    df_month2 = pd.concat([df_month2, future])

    # get_prediction得到的是区间，使用predicted_mean
    df_month2['forecast'] = best_model.get_prediction(start=0, end=len(df_month2)).predicted_mean

    # 沪市指数预测结果显示
    plt.figure(figsize=(30, 7))
    df_month2.y.plot(label='实际指数')
    df_month2.forecast.plot(color='r', ls='--', label='预测指数')
    plt.legend()
    plt.title('沪市指数（月）')
    plt.xlabel('时间')
    plt.ylabel('指数')
    plt.show()


rule2()
