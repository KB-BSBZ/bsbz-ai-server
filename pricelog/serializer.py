from rest_framework import serializers
from .models import LuxuryLog, EstateLog

class EstateLogSerializer(serializers.ModelSerializer):

    class Meta:
        model = EstateLog
        exclude = ('id', 'product_id')

class PriceLogPredictSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = 'PriceLog'
        field = '__all__'

class LuxuryLogSerializer(serializers.ModelSerializer):

    class Meta:
        model = LuxuryLog
        exclude = ('id', 'luxury_id')        