"""analysis URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('estate_log/<product_id>/', views.show_estate_log, name='show_estate_log'),
    path('luxury_log/<product_id>/', views.show_luxury_log, name='show_luxury_log'),
    path('music_log/<product_id>/', views.show_music_log, name='show_music_log'),
    path('estate_predict/<product_id>/', views.show_estate_predict, name='show_estate_predict'),
    path('luxury_predict/<product_id>/', views.show_luxury_predict, name='show_luxury_predict'),
    path('music_predict/<product_id>/', views.show_music_predict, name='show_music_predict'),
    path('news/', views.show_news, name="show_news"),
    path('estate_cloud_word/', views.show_estate_cloud_word, name="show_estate_cloud_word"),
    path('luxury_cloud_word/', views.show_luxury_cloud_word, name="show_luxury_cloud_word"),
]
