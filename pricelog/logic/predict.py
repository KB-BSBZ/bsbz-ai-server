import numpy as np
import pandas as pd
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from datetime import datetime
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
    # 가사가 포함되어 있는 댓글 삭제
    music1_df = pd.read_sql(f"SELECT * FROM pricelog_musiccommentlog WHERE music_id = {product_id}", conn)
    print("music1_df")
    print(music1_df)
    delete_keyword = ['후회하고 있어요', '후회하고있어요']
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
    stop_words = ['바보', '명수', '이석훈', '지아', '박명수', '곡', '사람', '가수', '노래', '이승기', '승기', '응급실', '이', '\n', '가', '에', '진짜', '도', '을', '잘', '너무',
                '를', '님', '는', '의', '내', '은', '들', '때', '한', '야', '번', '보', '부를', '그때',
                '너', '합니다', '씨', '만', '에서', '더', '로', '다', '안', '와', '으로', '거', '하고', '아', '요', '정말', '하는',
                '많이', '그', '나를', '저', '고', '년', '지금', '네', '그냥', '해', '춘', '게', '개', '형', '지' '듣고', '날', '이렇게', '왜', '때문', '입니다', '못', '아직도', '난', 
                '역시', '다시', '것', '이런', '줄', '우리', '아니야', '까지', '이다', '인데', '대', '알', '부터', '라', '적', '보고', '하', '팬', '고음', '개', '분', 
                '마', '할', '그렇게', '몰라', '들어', '원', '넘', '꽃', '없는데', '하나', '언제', '래', '된', '과', '가진', '인', '참', '수', '계속', '정도', '급', 
                '하게', '아니', '부른건', '좀', '어떻게', '중', '근데', '있어요', '임', '없이', '밖에', '하는데', '앞', '이제', '보다', '부르는', '해주세요', '면', '맨', '제', 
                '요즘', '랑', '늘', '뭐', '얼마나', '젠', '니까', '없는', '한다', '라고', '이지', '한번', '이노', '해서', '모든', '같은', '나', '항상', '제발', '지', 
                '듣고', '꼭', '밖엔', '이네', '\n     ', '떠나가지', '또', '라도', '두고', '부르고', '실', '이건', '들어도', '아무', '봐', '\n  ', '있는', '지나도', '가지', 
                '돼', '없어', '쉽게', '들으면', '\n    ', '끝', '살', '서', '함', '전', '한테', '했는데', '마다', '니', '알았어', '찾길', '했나', '처럼', '부른', '박', '불러', 
                '같아요', '엔', '속', '다투던', '내게', '해줘서', '힘들었던', '편이', '준', '멋대로', '이랑', '그리고', '좋겠다','건', '무슨', '이야', '갑니다', '보지', '듯',  
                '뭔가', '마세요', '다른', '분들', '듣기', '저렇게', '해요', '같이', '오', '두', '에도', '인가', '에게', '듣는', '나오는', '그동안', '언제나', '하면서', '권',
                '버리지', '여기', '예요', '\n   \n', '처음', '조차', '말', '일', '좋은','월', '금방', '온', '부분', '들으러', '들을', '락', '걸', '목', '오랜', '왔습니다', '댓글',
                '불', '딱', '어', '하지', '없니', '알았는데', '부', '갑자기', '위', '이고', '널', '메리', '울', '위해', '옹', '어제', '보내주신', '않을', '걸고', 
                '다시는', '겁니다나', '잡고', '리미', '메', '지켜', '훈', '석', '모두', '낼게요', '넌', '듯이', '보란', '없다면', '한다는', '따윈', '버릴게', 
                '보낼', '서로', '놓지', '미랑', '않을게', '뿐', '하며', '오늘', '제일', '없다면아마', '남', '채정안', '이승철', '형님', '원래', '남자', '여자', '볼', 
                '만날', '훨씬', '음', '미가', '해볼게', '니니', '이곡', '했었지니', '부르면', '몇', '하지는', '해볼게나', '갓', '뻔했어나', '볼거야', '좋지만', '냐', 
                '내사', '먼저', '이나', '완전', '따라', '좋', '비교', '자', '꺼', '없다', '자체', '라서', '\n   ', '여', '런가', '누구', '같은데', '태어날거야', 
                '누가', '어떤', '미', '하하', '하네요', '이형', '부르니', '그렇고', '이라', '그래도', '쿤', '하면', '지켜줄게요', '에요', '소', '죠', '데', '라는',
                '잡고다시', '태어날', '솔직히', '애', '보면', '보니', '둘다', '년전', '애', '아니라', '뭔', '존나', '자꾸', '나도', '시', '뿐이야다시', '같음', '엄청', '나온', 
                '멜', '로망스', '김상민', '버려진', '채', '슬피', '멜', '로망스', '김상민', '버려진', '채', '슬피', '자리', '우는', '혼자', '젖어', '들던', '손'
                '슈가', '오는', '서있는', '봤니', '들꺼야', '해봤니', '땐', '조금', '이분', '둘', '유','상민', '기', '해야', '작은', '보내', '이루겠지', '조차도', 
                '했었어', '어울린다고', '그럴', '지내다', '떠난다고', '해준다면', '이유', '품', '기대어', '꿈속', '흘리겠지', '보내고', '아파', '내겐', '그런', 
                '년대', '쉬즈', '같아', '곤', '억', '같다', '아주', '되면', '감으며', '깨어', '김경호', '남겨지는게', '잠못', '편할것', '잠못', '션', '잊을수', '년도', 
                '레이', '두려울텐데', '김민석', '성', '\n ' ,'조', '긴', '그래', '옆', '초', '만나', '난', '후', '문', '왔어', '봐도', '저런', '가서', '하지만', '같', 
                '알수', '스', '그대로', '거야', '본', '가장', '하네', '보며', '부른다', '거의', '해도', '재', '가요', '에는', '할거라는', '있을것', 
                '떠난다는', '시몬', '손', '없네', '린다', '예', '아직', '같아아주', '만나긴', '우와', '벌써', '아님', '모르는', '어도', '모습', '듣다가', '난후', 
                '정신', '나갔었나', '살때', '봤는데', '김탁구', '전이', '학년', '여구', '맨날', '떠나가', '였는데', '멍', '갔었나', '당시', '막', '냄', '만이', 
                '들으니까', '들으니', '봤던', '봤었는데', '똥꾸', '쉬', '챙겨', '예전', '어릴', '했었는데', '대웅', '들었는데' ,'파', '땜', '웅' ,'엇', '나유', '주님', 
                '첫', '언', '는데', '보던', '겁나', '즐겨', '돌아가고싶다', '나가서', '자기', '제빵', '왕', '크', '미호', '제목', '해줘', '흘러내려', '화', '바', 
                '곳', '되는', '확', '눈', '몰랐는데', '빠르다', '듣던', '싶다', '여서', '왔는데', '용형' ,'초깨비', '봤다', '나간', '반', '이라는', '계차', '만을', 
                '승', '앙', '명', '그거', '\n    \n', '특히', '타고', '좋고', '동', '들었던', '봤을', '현', '햇', '유주', '에일리', '가끔', '씩', '쯤', '가도', 
                '아닌', '오고', '남아', '겪어', '스런', '걸려', '있어', '안고', '누군가', '줘', '남기는', '버리면', '보는', '어디', '일까', '지는', '언니', '김이나', 
                '라니', '일리', '젤', '있을', '나우', '이정훈', '바로', '옴', '듣게', '자주', '인가요', '들으면서', '이해', '여전히', '듣는데', '몇번', '\n                \n', 
                '가사도', '신', '\n          \n', '절', '보면서', '\n                  \n', '뒤', '이즈', '게임', '노랜데', '아는', '가는', '창정', '임창정', 
                '한잔', '소주', '나야', '주', '여보', '여보세요', '미나', '창수', '마지막', '거기', '지내니', '에게로', '양', '가슴', '먹고', '하니', '불러오라고', 
                '지만', '하는게', '나오면', '아치', '혹시', '외쳤어', '건가', '이지만', '마시고', '이미', '차', '있는데', '낸', '캬', '누', '댓', '\n      ', '됨', 
                '건지', '그래서', '이란', '억뷰', '이형은', '그저', '왜케', '영', '한편', '올', '인거', '그녀', '임재범', '엄정화', '재범', '했던', '머리', '손지창', 
                '누님', '전혀', '지나면', '있다', '우리나라', '등', '했다', '물론', '얼', '한국', '마이클', '볼튼', '사자', '대한민국', '집', '하던', '호랑이', '비주', 
                '어느', '스킨', '왼쪽', '쇠고리', '티', '이라고', '본인', '유튜브', '일본', '아저씨', '덱스', '난다', '며', '계', '무', '영상', '김장훈', '장훈', 
                '숲튽훈', '숲', '나와', '김연우', '만에', '단', '같다면', '그대', '세종', '맘', '중간', '튽훈', '부르네',' 비워', '시작', '하는거', '보러', '세상', 
                '왤케', '나처르', '아악', '타', '된다', '좋아', '이후', '당신', '에서도', '튽', '력', '만의', '자신', '비워', '웃기', '체코', '결국', '수준', '하루', 
                '움', '까', '아니면', '아아', '놈', '퍼지면', '김', '빼면', '같네', '개인', '부르는게', '까는', '이소라', '둔', '박상태', '부름', '세계', '감', '하기', 
                '도대체', '되', '부르는거', '많은', '될', '있고', '주는', '하시는', '보소', '이문세', '대로', '함께', '좋냐', '씹', '숲툱훈', '했으면', '부르지', '좆', 
                '돌려', '인지', '김현식', '듣다', '에서는', '됩니다', '맛', '거미', '그랬어', '부릅니다', '조정석', '헤헤', '될껄', '김흥국', '옥수수', 
                '비긴', '이라도', '어게인', '껄', '휘성', '스파이더', '티비', '한예슬', '호호', '유희열', '박효신', '믿고', '구', '랫', '뺄', '든', '사실', '대체', 
                '열', '란', '간', '했어', '흑', '헤', '되고', '같아서', '만드는', '말고', '친구', '보는', '티비', '쾅쾅', '유나', '영상', '변기', '왔다', '유천', '하루',
                '최유나', '구독', '봉', '보미', '박유천', '있을까', '보여', '헐', '내주세요', '용', '젤', '노영진', '우연히', '로꼬', '신세경', '우현', '싶어',
                '솔', '어느새', '이니까', '이에요', '얘', '부르', '멀리', '\n     \n', '세', '덕분', '멤버', '바바티', '되게', '버젼', '매', '은하', '정해져있는', '아니고', 
                '우연이', '넘나', '째', '마디', '사', '스테파니', '내게오나', '향', '쾌걸', '속보', '민희', '황인', '욱', '강민희', '생각나서', '워', '장관', 
                '하고있어요', '왔어요', '방송', '니야', '윤찬',' 보기', '할수', '괜', '들은', '있다면', '한채영', '무조건', '였구나', '안치홍', '부른거', 
                '떠나', '보고온', '테디', '않는', '버린', '허나', '모르고', '볼줄', '나와서', '갈', '성춘향', '안보', '재희', '순간', '모', '몽룡', '응', '않고', '두번째', 
                '진', '없던', 
                
                '좋아요', '좋네요', '좋음', '좋아하는', '감사합니다', '호미', '미는', '미보', '좋네', '그게', '직접', '좋은데', '니깐', '겉', '생활', '많고', 
                '있습니다', '첨', '훨', '좋아서', '불렀던', '한다고', '하다', '있음', '얘기', '문과', '보급', '살짝', '꾼', '승철', '연예인', '진행', '가라', 
                '괜히', '김민근', '이자', '큰', '맞다', '수옹', '지내고', '천', '넘게', '찍어', '불러서', '좋아하는데', '보기', '너무나', '나오고', '있었는데', 
                '터', '가가', '나온거', '노래네요', '부르는데', '부른게', '오지', '은근', '좋다', 
                '좋다', '폰트', '좋네요', '글씨체', '가면', '좋음', '출석', '보기', '검색', '마카롱', '배', '발', '좋네', '자세히', '간주', '시대', 
                '따', '하다가', '이번', '들어요', '시발', '열심히', '미니', '돌아가고', '안나', '있네요', '회', '슈가', '있나', '인줄', '다운', '오신', '드립', '학번', 
                '어플', '성인', '인증', '출첵', '하세요', '음악', '회수', '노래방', '엄마', '좋아하는', '감사합니다', '선생님', '신화', '갔습니다', '분명', '는걸', '있나요', '들어온', '하자', '않아', '찾아', '좋아요'
                '좋다', '좋아요', '슈가', '어깨', '노래방', '좋음', '감사합니다', '민석', '좋네요', '좋은데', '좋아하는', '좋네', '정동환', '환', '수가', '본방', 
                '케이윌', '높은', '갠', '망', '부르기', '키로', '신청', '해주신', '청곡', '어쩜', '엄마', '부르는데', '발', '뉴', '대는', '안변', '박진영', 
                '\n           \n', '있는거', '들어왔는데', '에선', '랄', '감사해요', '있어서', '광고', '땅', '따로', '얼른', '있네', '조아요', '좋습니다', '좋아용', 
                '머', '입', '당', '이리', '앜', '생기', '차이', '갬', '봐요', '나나', '게이', '\n  \n', '\n \n', '걍', '으로도', '드라마', '구미호', '초딩', '신민아', '초딩', 
                '재밌게', '좋아요', '엄마', '좋네', '좋아', '유튜브', '그땐', '좋아했는데', '돋는다', '계', '좋다', '영상', '태권도', '제적', '만큼', 
                '좋네요', '알투비트', '듣네', '난다', '안나', '라니', '쯤', '아이', '지만', '이라니', '아빠', '살인', '뎅', '채널', '광희', '먹어', '배그', 
                '알고리즘', '에가', '재밌어서', '뭘', '찾았다', '몰래', '파트', '한국인', '이름', '오래', '잔', '상하이', '창', '만능', '이대로', '별']
        
    tokens_ko = [each_word for each_word in tokens_ko
           if each_word not in stop_words]

    ko = nltk.Text(tokens_ko)
    ko.vocab().most_common(70)
    
    common_words = ko.vocab().most_common(70)

    # 리스트를 데이터 프레임으로 변환
    df_common_words = pd.DataFrame(common_words, columns=['word', 'frequency'])
    print("df_common_words")
    print(df_common_words)

    answer = df_common_words.to_dict(orient='records')
    
    # 데이터프레임을 json 파일로 저장

    return answer

def estate_cloud():
    df = pd.read_sql(f"SELECT contents, pubdate FROM pricelog_estate_text", conn)
    print(df)


def luxury_cloud():
    df = pd.read_sql(f"SELECT contents, pubdate FROM pricelog_luxury_text", conn)
    print(df)



    