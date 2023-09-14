from django.shortcuts import render
from rest_framework.decorators import api_view
# from serializer import TradeLogSerializer
from django.shortcuts import get_list_or_404, render, get_object_or_404
from rest_framework.response import Response
from .models import EstateLog, LuxuryLog, MusicLog
from .serializer import LuxuryLogSerializer, EstateLogSerializer
from .logic.predict import estate_predict
import urllib
import json
import random

# Create your views here.
@api_view(['GET',])
def show_all_log(request, product_id, product_type):
    if product_type=='1':
        products = EstateLog.objects.filter(product_id = product_id)
        serializer = EstateLogSerializer(products, many = True)
        return Response(serializer.data)
    
    if product_type=='2':
        products = LuxuryLog.objects.filter(luxury_id = product_id)
        serializer = LuxuryLogSerializer(products, many = True)
        return Response(serializer.data)
    
    if product_type=='3':
        pass
        
    return Response({'message' : 'error'})

@api_view(['GET',])
def show_estate_predict(request, product_id):
    print("왔다네 왔다네 여기까지 왔다네")
    data = estate_predict(product_id)
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
        print(q)
        print(data["items"])
        for d in data["items"]:
            print(d)
            result.append(d)
    
    sampleNum = random.sample(sampleList, 5)
    result_sampled = []
    for s in sampleNum:
        result_sampled.append(result[s])
        
    
    
    return Response(result_sampled)