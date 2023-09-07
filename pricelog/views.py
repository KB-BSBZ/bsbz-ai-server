from django.shortcuts import render
from rest_framework.decorators import api_view
# from serializer import TradeLogSerializer
from django.shortcuts import get_list_or_404, render, get_object_or_404
from rest_framework.response import Response


# Create your views here.
@api_view(['GET'])
def show_all_log(request, product_id):
    print("왔다네 왔다네 여기까지 왔다네")
    print(product_id)
    return Response({"id" : 1})

@api_view(['GET'])
def show_predict(request, product_id):
    print("왔다네 왔다네 여기까지 왔다네")
    print(product_id)
    return Response({"id" : product_id})