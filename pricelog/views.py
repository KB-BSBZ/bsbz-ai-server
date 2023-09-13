from django.shortcuts import render
from rest_framework.decorators import api_view
# from serializer import TradeLogSerializer
from django.shortcuts import get_list_or_404, render, get_object_or_404
from rest_framework.response import Response
from .models import EstateLog, LuxuryLog, MusicLog
from .serializer import LuxuryLogSerializer


# Create your views here.
@api_view(['GET',])
def show_all_log(request, product_id, product_type):
    print(product_type)
    print(type(product_type))
    if product_type=='2':
        products = LuxuryLog.objects.filter(luxury_id = product_id)
        print("testestest")
        print(type(products))
        serializer = LuxuryLogSerializer(products, many = True)
        return Response(serializer.data)
    return Response({'message' : 'error'})

@api_view(['GET'])
def show_predict(request, product_id):
    print("왔다네 왔다네 여기까지 왔다네")
    print(product_id)
    return Response({"id" : product_id})