from rest_framework import serializers
from models import PriceLog

class PriceLogShowSerializer(serializers.ModelSerializer):

    class Meta:
        model = 'PriceLog'
        field = '__all__'

class PriceLogPredictSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = 'PriceLog'
        field = '__all__'
        