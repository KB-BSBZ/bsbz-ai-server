from django.shortcuts import render
from rest_framework.decorators import api_view
# from serializer import TradeLogSerializer
from django.shortcuts import get_list_or_404, render, get_object_or_404
from django.db.models.functions import Concat
from django.db.models import F, Value, CharField


from rest_framework.response import Response
from .models import EstateLog, LuxuryLog, MusicLog
from .serializer import LuxuryLogSerializer, EstateLogSerializer, MusicLogSerializer, LogSerializer
from .logic.predict import estate_predict, music_predict, luxury_predict, luxury_cloud, estate_cloud
import urllib
import json
import random

# Create your views here.
@api_view(['GET',])
def show_estate_log(request, product_id):
    products = EstateLog.objects.filter(product_id = product_id).annotate(ymd = Concat('year', Value('-'), 'month', Value('-'), 'day', output_field=CharField(max_length=20))).values()
    serializer = LogSerializer(products, many = True)
    return Response(serializer.data)

@api_view(['GET',])
def show_luxury_log(request, product_id):
    products = LuxuryLog.objects.filter(luxury_id = product_id)
    serializer = LogSerializer(products, many = True)
    return Response(serializer.data)

@api_view(['GET',])
def show_music_log(request, product_id):
    products = MusicLog.objects.filter(music_id = product_id).annotate(price = F('price_close')).values()
    serializer = LogSerializer(products, many = True)
    return Response(serializer.data)

@api_view(['GET',])
def show_estate_predict(request, product_id):
    data = estate_predict(product_id)
    return Response(data)

@api_view(['GET',])
def show_luxury_predict(request, product_id):
    data = luxury_predict(product_id)
    return Response(data)

@api_view(['GET',])
def show_music_predict(request, product_id):
    data = music_predict(product_id)
    return Response(data)


@api_view(['GET',])
def show_news(request):
    result = []
    client_id = "VgDYFbG_QqQM1DMRL7VQ"
    client_secret = "on5OWBwcMm"
    
    query_list = ["부동산 시장", "명품시계 시장", "명품가방 시장", "뮤직카우"]
    
    sampleList = []
    for i in range(40):
        sampleList.append(i)
    
    for q in query_list:
        encText = urllib.parse.quote(q)
        url = "https://openapi.naver.com/v1/search/news.json?query=" + encText + "&" +"sort=sim" # JSON 결과
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)
        response = urllib.request.urlopen(request)
        response_body = response.read()
        data = json.loads(response_body.decode('utf-8-sig'))
        for d in data["items"]:
            result.append(d)
    
    sampleNum = random.sample(sampleList, 5)
    result_sampled = []
    for s in sampleNum:
        result_sampled.append(result[s])
        
    result = []
    
    cnt = 0
    for rs in result_sampled:
        rs["id"] = cnt
        rs["title"] = rs["title"].replace('<b>', '')
        rs["title"] = rs["title"].replace('</b>', '')
        rs['title'] = rs['title'].replace('&quot;', '')
        rs["description"] = rs["description"].replace('<b>', '')
        rs["description"] = rs["description"].replace('</b>', '')
        rs['description'] = rs['description'].replace('&quot;', '')
        cnt += 1

    return Response(result_sampled)

@api_view(['GET',])
def show_music_cloud_word(request, product_id):
    
    return Response("")

@api_view(['GET',])
def show_estate_cloud_word(request):
    df = estate_cloud()
    return Response("")

@api_view(['GET',])
def show_luxury_cloud_word(request):
    df = luxury_cloud()
    return Response("")