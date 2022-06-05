
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

# from sklearn import metrics
import math
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from sklearn.preprocessing import MinMaxScaler

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
    # print(con['time'][len(con['time'])-1])
    date_format = "%Y-%m-%d %H:%M:%S"
    dtObj = con['time'][len(con['time'])-1]

    time_list = []
    for i in range(1,25):
        future_date = dtObj + timedelta(hours=i)
        print('Future Date: ', future_date)
        time_list.append(future_date)
    print(time_list)
    
    # ============================================
    # preprocessing
    # ============================================
    avg_con = [0] * 24
    for i in range(len(con)):
        avg_con[i%24]  += con['consumption'][i]
    #print(avg_con)
    avg_con = [round(i / 7,2) for i in avg_con]
    print(avg_con)

    avg_gen = [0] * 24
    for i in range(len(con)):
        avg_gen[i%24]  += gen['generation'][i]
    #print(avg_gen)
    avg_gen = [round(i / 7,2) for i in avg_gen]
    print(avg_gen)


    # ============================================
    # consumption prediction
    # ============================================
    data = []
    for i in range(0,24):
        if avg_con[i] > avg_gen[i]:
            buy_mount = avg_con[i] - avg_gen[i]
            data.append([str(time_list[i]), 'buy', 2, round(buy_mount,2)])
        elif avg_con[i] < avg_gen[i]:
            sell_mount = avg_gen[i] - avg_con[i]
            data.append([str(time_list[i]), 'sell', 2.5, round(sell_mount,2)])
    print(data)
    output(args.output, data)
    