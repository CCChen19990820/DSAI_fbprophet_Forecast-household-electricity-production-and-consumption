# DSAI-HW3-_Forecast-household-electricity-production-and-consumption

題目流程如下圖所示:
![螢幕擷取畫面 2022-06-05 182926](https://user-images.githubusercontent.com/48405514/172046264-adf99aa2-a01d-4e57-9a03-cffbf135e2e3.png)

程式會讀取家庭7天的comsumption.csv以及generation.csv檔案並給出未來一天的家庭用電量以及產電量的預測，並將預測出的結果做出未來一天每個小時要買還是賣的決策，平台媒合後所得到的結果會存在bidresult.csv當中，可輔助做下一次的預測。而目的是要盡可能的花費最少的電費，電費的計算方式如下圖:
![截圖 2022-05-10 下午11 35 09](https://user-images.githubusercontent.com/48405514/172046424-9d3c1e87-e334-4b4f-bdd7-3a1ee25ffcb2.png)

##預測方法
本組總共採用三種方法做預測，分別使用了fbprophet模型、LSTM模型、以及7天均值三種方法，方法敘述如下列:
1.
