
# basic
from cProfile import label
import numpy as np
import pandas as pd

# visual
import matplotlib.pyplot as plt

#time
import datetime as datetime
from datetime import timedelta
#Prophet
#from fbprophet import Prophet
# from fbprophet import Prophet

from sklearn import metrics
import math
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="C:/Users/Jeff/Downloads/DSAI-HW3LSTM/DSAI-HW3-2021-master/sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="C:/Users/Jeff/Downloads/DSAI-HW3LSTM/DSAI-HW3-2021-master/sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="C:/Users/Jeff/Downloads/DSAI-HW3LSTM/DSAI-HW3-2021-master/sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")
    return parser.parse_args()

def output(path, data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return

def create_dataset(dataset,look_back=1):
    dataX,dataY = [],[]
    for i in range(look_back,len(dataset)):
        a = dataset[i-look_back:i,]
        dataX.append(a)
        dataY.append(dataset[i,0])
    #print(len(dataY))
    return np.array(dataX), np.array(dataY)

if __name__ == "__main__":
    print('start')
    args = config()

    # model = Prophet()
    con = pd.read_csv(args.consumption)
    gen = pd.read_csv(args.generation)
    bid = pd.read_csv(args.bidresult)
    #output(args.output, data)

    # ============================================
    # add a hour
    # ============================================
    con['time'] = pd.to_datetime(con['time'], format="%Y-%m-%d %H:%M:%S")
    print(con['time'][len(con['time'])-1])
    date_format = "%Y-%m-%d %H:%M:%S"
    dtObj = con['time'][len(con['time'])-1]
    future_date = dtObj + timedelta(hours=1)
    print('Future Date: ', future_date)
    
    
    # ============================================
    # preprocessing
    # ============================================
    # temp = pd.DataFrame(temp)
    # scaler = MinMaxScaler(feature_range=(0,1))
    # scaled = scaler.fit_transform(temp)
    # print(scaled)
    temp = con['consumption']
    temp = pd.DataFrame(temp)
    train_size = 120
    test_size = 48
    train = temp.iloc[:120, ].values
    test = temp.iloc[120:, ].values

    scaler = MinMaxScaler(feature_range=(0,1))
    trainN = scaler.fit_transform(train)
    testN = scaler.transform(test)
    #print(len(train),len(test))

    look_back = 24
    trainX, trainY = create_dataset(trainN,look_back)
    testX, testY = create_dataset(test,look_back)
    #print(testX,testY)
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # ============================================
    # LSTM model train
    # ============================================
    model = Sequential()
    model.add(LSTM(100,input_shape=(trainX.shape[1],trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    model.summary()
    history = model.fit(trainX,trainY,epochs=100)

    # ============================================
    # consumption prediction
    # ============================================

    start = test[0:24]
    abc = np.concatenate( (test[0:24], [[0]] ) )
    ccc,bbb = create_dataset(abc,look_back)
    #print(abc)
    for i in range(0,23):
        #abc = np.concatenate( (abc, [test[25+i]] ) )
        predict = model.predict(ccc)
        predicted_power = scaler.inverse_transform(predict)
        print(predicted_power[len(predict)-1])
        start = np.concatenate( (start, [predicted_power[len(predict)-1]] ) )
        abc = np.concatenate( (start, [[0]] ) )
        ccc,bbb = create_dataset(abc,look_back)
    # print(testX)
    # predict = model.predict(testX)
    print(start)

    # predicted_power = scaler.inverse_transform(predict)
    # print('predict',predicted_power)

    # ============================================
    # Visualising the results
    # ============================================
    plt.plot(test[24:], color = 'red', label = 'Real ')  # 紅線表示真實股價
    plt.plot(predicted_power, color = 'blue', label = 'Predicted')  # 藍線表示預測股價
    plt.title('power Prediction')
    plt.xlabel('Time')
    plt.ylabel('power consumptions')
    plt.legend()
    plt.show()

    # # ============================================
    # # consumption prediction
    # # ============================================
    # temp = con
    # temp = temp.rename(columns={'time': 'ds', 'consumption': 'y'})
    # temp['ds'] = pd.to_datetime(temp['ds'], format="%Y-%m-%d %H:%M:%S")
    # model.fit(temp)
    # # 建構預測集
    # future = model.make_future_dataframe(periods=24, freq='h') #forecasting for 1 year from now.
    # # 進行預測
    # forecast = model.predict(future)
    # #print(forecast[['ds','yhat']].tail(24))
    # #figure=model.plot(forecast)

    # # ============================================
    # # generation prediction
    # # ============================================
    # model2 = Prophet()
    # temp2 = gen
    # temp2 = temp2.rename(columns={'time': 'ds', 'generation': 'y'})
    # temp2['ds'] = pd.to_datetime(temp2['ds'], format="%Y-%m-%d %H:%M:%S")
    # model2.fit(temp2)
    # # 建構預測集
    # future2 = model2.make_future_dataframe(periods=24, freq='h') #forecasting for 1 year from now.
    # # 進行預測
    # forecast2 = model2.predict(future2)
    # #print(forecast2[['ds','yhat']].tail(24))
    # #figure=model.plot(forecast)

    # # ============================================
    # # concat consumption and generation prediction
    # # ============================================
    # prediction = forecast[['ds','yhat']].tail(24)
    # prediction['gen_pred'] = forecast2['yhat'].tail(24)
    # prediction = prediction.rename(columns={'yhat': 'con_pred', 'ds': 'time', 'gen_pred' : 'gen_pred'})
    # prediction['modify'] = prediction['gen_pred'].where(prediction['gen_pred'] > 0, 0)
    # prediction['residual'] = prediction['modify'] - prediction['con_pred']
    # prediction['buy'] = prediction['con_pred'] - prediction['modify']
    # prediction['buy_int'] = prediction['buy'].astype(int)


    # print(prediction)

    # data = []
    # for index, row in prediction.iterrows():
    #     # format = '%Y-%m-%d %I:%M:%S %p'
    #     # my_date = datetime.datetime.strptime(str(row['time']), format)
    #     if row['buy_int'] > 0:
    #         #print('buy')
    #         # data.append([str(row['time']), 'buy', 2, math.ceil(row['buy_int']/2)])
    #         # data.append([str(row['time']), 'buy', 2.5, math.floor(row['buy_int']/2)])
    #         data.append([str(row['time']), 'buy', 2.5, round(row['buy'], 2)])
    #     elif row['buy_int'] < 0:
    #         #print('sell')
    #         # data.append([str(row['time']), 'sell', 2.5, math.ceil((row['buy_int']* -1)/2)])
    #         # data.append([str(row['time']), 'sell', 2, math.floor((row['buy_int']* -1)/2)])
    #         data.append([str(row['time']), 'sell', 2.5, round(row['buy']* -1, 2)])
    #     else:
    #         continue
    # # print(data)
    # output(args.output, data)