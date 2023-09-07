from django.shortcuts import render
from rest_framework import api_view
from serializer import TradeLogSerializer
from django.shortcuts import get_list_or_404, render, get_object_or_404


# Create your views here.
@api_view('GET')
def show_all_log():
    pass