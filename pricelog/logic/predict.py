from .ai_asset import *

import numpy as np
import pandas as pd
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller


import pandas as pd
import numpy as np

# word cloud 관련
from konlpy.tag import Kkma        ; kkma = Kkma()
from konlpy.tag import Hannanum    ; hannanum = Hannanum()
from konlpy.tag import Okt         ; t = Okt()     # 구 트위터
from konlpy.tag import *
import nltk

from wordcloud import WordCloud, STOPWORDS

import re

np.random.seed(42)
database = "db.sqlite3"
conn = sqlite3.connect(database, check_same_thread=False)

def get_best_area(df):
    # 가장 거래수가 많은 평수를 찾는 로직
    area = df['area'].value_counts().keys()[0]
    if (area > 70 and area < 90):
        area = int(area)
    else: 
        # 가장 많은 거래가 일어나는 평수
        area = 84
    
    area_list = []
    for i in df['area'].value_counts().keys():
        temp = int(i)
        if temp == area:
            area_list.append(i)
            
    df2 = pd.DataFrame()
    
    for al in area_list:
        temp = df[df['area']== al]
        df2 = pd.concat([df2, temp]) 
    return df2


def estate_predict(product_id):
    df1 = pd.read_sql(f"SELECT * FROM pricelog_estatelog WHERE product_id = {product_id}", conn)
    df1['month'] = df1['month'].apply(lambda x : str(x).zfill(2))
    df1['day'] = df1['day'].apply(lambda x : str(x).zfill(2))

    cols = ['year', 'month', 'day']
    df1['date'] =df1[cols].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)

    df2 = get_best_area(df1)

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
    # 가사가 포함되어 있는 댓글 삭제
    music1_df = pd.read_sql(f"SELECT * FROM pricelog_musiccommentlog WHERE music_id = {product_id}", conn)
    delete_keyword = music_delete_words_list()[int(product_id) - 21]
    music1_df = music1_df[music1_df['comment'].fillna('').apply(lambda x: not any(keyword in x for keyword in delete_keyword))]

    # 인덱스 다시 세팅
    music1_df.reset_index(drop=True, inplace=True)
    
    music1_df['comment'] = music1_df['comment'].astype(str) # 'comment' 열의 데이터를 문자열로 변환

    music1_df['comment'].map(lambda x: re.sub('[,\.!?]', '', x))

    # 한글이 아닌 문자 제거
    # re.sub(정규표현식, new_text, old_text)
    music1_df["comment"] = music1_df["comment"].apply( lambda x : re.sub("[^가-힣]", " ", x) )
    music1_df["comment"]


    title_list = music1_df.comment.values.tolist()
    title_text = ''
    for each_line in title_list:
        title_text = title_text + each_line + '\n'

    tokens_ko = t.morphs(title_text)
    
    ko = nltk.Text(tokens_ko)
    
    #불용어 처리
    stop_words = music_stop_words_list()
    tokens_ko = [each_word for each_word in tokens_ko
           if each_word not in stop_words]

    ko = nltk.Text(tokens_ko)
    ko.vocab().most_common(70)
    
    common_words = ko.vocab().most_common(70)

    # 리스트를 데이터 프레임으로 변환
    df_common_words = pd.DataFrame(common_words, columns=['word', 'frequency'])

    answer = df_common_words.to_dict(orient='records')
    
    # 데이터프레임을 json 파일로 저장

    return answer


def estate_cloud():
    news = pd.read_sql(f"SELECT contents, pubdate FROM pricelog_estatetext", conn)
    # 중복된 기사 내용은 1개만 남기고 삭제
    news = news.drop_duplicates(subset='contents', keep='first')

    # 인덱스 다시 세팅
    news.reset_index(drop=True, inplace=True)
    
    # Remove punctuation(구두점 제거)
    news['contents'] = news['contents'].astype(str) # 'comment' 열의 데이터를 문자열로 변환

    news['contents'].map(lambda x: re.sub('[,\.!?]', '', x))

    # 한글이 아닌 문자 제거
    # re.sub(정규표현식, new_text, old_text)
    news["contents"] = news["contents"].apply( lambda x : re.sub("[^가-힣]", " ", x) )
    
    #형태소 분석을 수행하기 위해 '\n' 으로 구분하여 한 문자열에 담음
    title_list = news.contents.values.tolist()
    title_text = ''
    for each_line in title_list:
        title_text = title_text + each_line + '\n'
        
    #형태소 분석
    tokens_ko = t.morphs(title_text)
    
    ko = nltk.Text(tokens_ko)
    
    
    #불용어 제거된 텍스트 데이터 생성 및 출력
    stop_words = estate_stop_words_list()
    
    tokens_ko = [each_word for each_word in tokens_ko
            if each_word not in stop_words]

    ko = nltk.Text(tokens_ko)
    
    #워드 클라우드를 서버랑 연결하기 위해 따로 저장

    common_words = ko.vocab().most_common(10)

    # 리스트를 데이터 프레임으로 변환
    df_common_words = pd.DataFrame(common_words, columns=['word', 'frequency'])

    # 데이터프레임을 json 파일로 저장
    result = df_common_words.to_dict(orient='records')
    
    return result


def luxury_cloud():
    news= pd.read_sql(f"SELECT contents, pubdate FROM pricelog_luxurytext", conn)
    news = news.drop_duplicates(subset='contents', keep='first')

    # 인덱스 다시 세팅
    news.reset_index(drop=True, inplace=True)
    
    # Remove punctuation(구두점 제거)
    news['contents'] = news['contents'].astype(str) # 'comment' 열의 데이터를 문자열로 변환

    news['contents'].map(lambda x: re.sub('[,\.!?]', '', x))

    # 한글이 아닌 문자 제거
    # re.sub(정규표현식, new_text, old_text)
    news["contents"] = news["contents"].apply( lambda x : re.sub("[^가-힣]", " ", x) )
    
    #형태소 분석을 수행하기 위해 '\n' 으로 구분하여 한 문자열에 담음
    title_list = news.contents.values.tolist()
    title_text = ''
    for each_line in title_list:
        title_text = title_text + each_line + '\n'
        
    #형태소 분석
    tokens_ko = t.morphs(title_text)
    
    ko = nltk.Text(tokens_ko)
    
    
    #불용어 제거된 텍스트 데이터 생성 및 출력
    stop_words = luxury_stop_words_list()
    
    tokens_ko = [each_word for each_word in tokens_ko
            if each_word not in stop_words]

    ko = nltk.Text(tokens_ko)
    
    #워드 클라우드를 서버랑 연결하기 위해 따로 저장

    common_words = ko.vocab().most_common(10)

    # 리스트를 데이터 프레임으로 변환
    df_common_words = pd.DataFrame(common_words, columns=['word', 'frequency'])

    # 데이터프레임을 json 파일로 저장
    result = df_common_words.to_dict(orient='records')
    
    return result

