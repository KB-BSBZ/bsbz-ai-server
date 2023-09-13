from rest_framework import serializers
from .models import LuxuryLog

class PriceLogSerializer(serializers.ModelSerializer):

    class Meta:
        model = 'PriceLog'
        field = '__all__'

class PriceLogPredictSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = 'PriceLog'
        field = '__all__'

class LuxuryLogSerializer(serializers.ModelSerializer):

    class Meta:
        model = LuxuryLog
        exclude = ('id', 'luxury_id')        