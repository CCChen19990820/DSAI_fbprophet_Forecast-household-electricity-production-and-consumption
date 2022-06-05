
# basic
import numpy as np
import pandas as pd

# visual
import matplotlib.pyplot as plt

#time
import datetime as datetime

#Prophet
#from fbprophet import Prophet
from fbprophet import Prophet

from sklearn import metrics
import math

# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return


if __name__ == "__main__":
    print('start')
    args = config()

    model = Prophet()
    data = [["2018-01-01 00:00:00", "buy", 2.5, 3],
            ["2018-01-01 01:00:00", "sell", 3, 5]]

    con = pd.read_csv(args.consumption)
    gen = pd.read_csv(args.generation)
    bid = pd.read_csv(args.bidresult)
    #output(args.output, data)

    # ============================================
    # consumption prediction
    # ============================================
    temp = con
    temp = temp.rename(columns={'time': 'ds', 'consumption': 'y'})
    temp['ds'] = pd.to_datetime(temp['ds'], format="%Y-%m-%d %H:%M:%S")
    model.fit(temp)
    # 建構預測集
    future = model.make_future_dataframe(periods=24, freq='h') #forecasting for 1 year from now.
    # 進行預測
    forecast = model.predict(future)
    #print(forecast[['ds','yhat']].tail(24))
    #figure=model.plot(forecast)

    # ============================================
    # generation prediction
    # ============================================
    model2 = Prophet()
    temp2 = gen
    temp2 = temp2.rename(columns={'time': 'ds', 'generation': 'y'})
    temp2['ds'] = pd.to_datetime(temp2['ds'], format="%Y-%m-%d %H:%M:%S")
    model2.fit(temp2)
    # 建構預測集
    future2 = model2.make_future_dataframe(periods=24, freq='h') #forecasting for 1 year from now.
    # 進行預測
    forecast2 = model2.predict(future2)
    #print(forecast2[['ds','yhat']].tail(24))
    #figure=model.plot(forecast)

    # ============================================
    # concat consumption and generation prediction
    # ============================================
    prediction = forecast[['ds','yhat']].tail(24)
    prediction['gen_pred'] = forecast2['yhat'].tail(24)
    prediction = prediction.rename(columns={'yhat': 'con_pred', 'ds': 'time', 'gen_pred' : 'gen_pred'})
    prediction['modify'] = prediction['gen_pred'].where(prediction['gen_pred'] > 0, 0)
    prediction['residual'] = prediction['modify'] - prediction['con_pred']
    prediction['buy'] = prediction['con_pred'] - prediction['modify']
    prediction['buy_int'] = prediction['buy'].astype(int)


    print(prediction)

    data = []
    for index, row in prediction.iterrows():
        # format = '%Y-%m-%d %I:%M:%S %p'
        # my_date = datetime.datetime.strptime(str(row['time']), format)
        if row['buy_int'] > 0:
            #print('buy')
            # data.append([str(row['time']), 'buy', 2, math.ceil(row['buy_int']/2)])
            # data.append([str(row['time']), 'buy', 2.5, math.floor(row['buy_int']/2)])
            data.append([str(row['time']), 'buy', 2.5, round(row['buy'], 2)])
        elif row['buy_int'] < 0:
            #print('sell')
            # data.append([str(row['time']), 'sell', 2.5, math.ceil((row['buy_int']* -1)/2)])
            # data.append([str(row['time']), 'sell', 2, math.floor((row['buy_int']* -1)/2)])
            data.append([str(row['time']), 'sell', 2.5, round(row['buy']* -1, 2)])
        else:
            continue
    # print(data)
    output(args.output, data)