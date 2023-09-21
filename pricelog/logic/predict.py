import numpy as np
import pandas as pd
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from datetime import datetime
from statsmodels.tsa.stattools import adfuller

np.random.seed(42)

database = "db.sqlite3"
conn = sqlite3.connect(database, check_same_thread=False)

# id: 2
def estate_predict(product_id):
    df1 = pd.read_sql(f"SELECT * FROM pricelog_estatelog WHERE product_id = {product_id}", conn)

    df1['month'] = df1['month'].apply(lambda x : str(x).zfill(2))
    df1['day'] = df1['day'].apply(lambda x : str(x).zfill(2))

    cols = ['year', 'month', 'day']
    df1['date'] =df1[cols].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
    
    df2 = df1[df1['area']==76.79]

    df3 = df2.copy()
    df3['date'] = pd.to_datetime(df3.date, format='%Y-%m-%d')
    df3 = df3.groupby('date')['price'].mean().reset_index()

    df4 = df3.copy()
    df4.index = df4.date
    df4 = df4.drop(columns="date")
    
    for i in range(1, len(df4) - 1):
        current_price = df4['price'].iloc[i]
        previous_price = df4['price'].iloc[i - 1]
        next_price = df4['price'].iloc[i + 1]
        
        if abs(current_price - previous_price) >= 50000 and abs(current_price - next_price) >= 50000:
            avg_price = (current_price + previous_price) /2
            df4.at[df4.index[i], 'price'] = avg_price
    
    train = df4['price'][:int(0.8*len(df4))]
    test = df4['price'][int(0.8*len(df4)):]
    
    
    model2 = pm.auto_arima(train, d=1, seasonal=False, trace=True)       
    
    def forcast_one_step():
        fc, conf = model2.predict(n_periods=1, return_conf_int=True)
        return fc.tolist()[0], np.asarray(conf).tolist()[0]
    
    # 값들을 담을 빈 리스트를 생성
    y_pred = []
    pred_upper = []
    pred_lower = []

    # for문으로 예측 및 모델 업데이트 반복 (1일씩)
    forecast_period = 12   # ?일을 나타내는 일 수
    for _ in range(forecast_period):
        fc, conf = forcast_one_step()
        y_pred.append(fc)
        pred_upper.append(conf[1])
        pred_lower.append(conf[0])
        
        # 모델 업데이트
        model2.update(fc)

    forecast_index = pd.date_range(start=test.index[-1], periods=forecast_period + 1, freq='M')
    ymd = []
    for i in forecast_index.tolist():
        ymd.append(i.strftime('%Y-%m-%d'))
 

    # 최종 12개월 예측 결과 출력
    data = []
    for i in range(12):
        temp = {}
        temp['date'] = ymd[i]
        temp["predict"] = y_pred[i]
        temp["upper"] = pred_upper[i]
        temp["lower"] = pred_lower[i]
        data.append(temp)
  
    return data

# 
def luxury_predict(product_id):
    df = pd.read_sql(f"SELECT * FROM pricelog_luxurylog WHERE luxury_id = {product_id}", conn)
    
    #'Date' 열을 datetime형으로 변환하기
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df.ymd, format='%Y-%m-%d')

    df2 = df2[['date', 'price']]

    #날짜를 인덱스로 변환하기
    #인덱스 설정 후 drop
    df3 = df2.copy()
    df3.index = df3.date
    df3 = df3.drop(columns="date")
    
    adfuller(df3, autolag='AIC')
    
    # index를 period로 변환해주어야 warning 뜨지 않음
    df3_copy = df3.copy()
    df3_copy.index = pd.DatetimeIndex(df3_copy.index).to_period('D')

    # 예측을 시작할 위치(이후 차분을 적용하기 때문에 맞추어주었음
    start_idx = df3_copy.index[1]

    #train data: 80%, test data: 20%
    train = df3['price'][:int(0.8*len(df3))] #80%
    test = df3['price'][int(0.8*len(df3)):] #나머지 20%
    
    #최적화된 ARIMA 모델 분석
    #최적화 파라미터로 ARIMA 하기
    #모델 파라미터 최적화 (p=1, d=1, q=3)
    model2 = ARIMA(df3_copy, order=(0,1,0))
    
    # fit
    model2 = pm.auto_arima(train, d=1, seasonal=False, trace=True)
    
    # ARIMA 모델 생성과 학습
    def forcast_one_step():
        fc, conf = model2.predict(n_periods=1, return_conf_int=True)
        return fc.tolist()[0], np.asarray(conf).tolist()[0]

    y_pred = []
    pred_upper = []
    pred_lower = []
    
    # for문으로 예측 및 모델 업데이트 반복 (1일씩)
    forecast_period = 12   # ?일을 나타내는 일 수
    for _ in range(forecast_period):
        fc, conf = forcast_one_step()
        y_pred.append(fc)
        pred_upper.append(conf[1])
        pred_lower.append(conf[0])
        
        # 모델 업데이트
        model2.update(fc)

    forecast_index = pd.date_range(start=test.index[-1], periods=forecast_period + 1, freq='M')
    forecast_df = pd.DataFrame({'Predicted': y_pred, 'Upper': pred_upper, 'Lower': pred_lower}, index=forecast_index[1:])
    
    ymd = []
    for i in forecast_index.tolist():
        ymd.append(i.strftime('%Y-%m-%d'))

    # 최종 12개월 예측 결과 출력
    data = []
    cnt = 0
    for idx, ser in forecast_df.iterrows():
        temp = {}
        temp['date'] = ymd[cnt]
        temp["predict"] = ser["Predicted"] / 10000
        temp["upper"] = ser["Upper"] / 10000
        temp["lower"] = ser["Lower"] / 10000
        data.append(temp)
        cnt += 1

    return data

    

def music_predict(product_id):
    
    return product_id

def estate_cloud():
    df = pd.read_sql(f"SELECT contents, pubdate FROM pricelog_estate_text", conn)
    print(df)


def luxury_cloud():
    df = pd.read_sql(f"SELECT contents, pubdate FROM pricelog_luxury_text", conn)
    print(df)



    