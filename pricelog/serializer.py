from rest_framework import serializers
from .models import LuxuryLog, EstateLog, MusicLog

class EstateLogSerializer(serializers.ModelSerializer):

    class Meta:
        model = EstateLog
        exclude = ('id', 'product_id')

class MusicLogSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = MusicLog
        exclude = ('id' , 'music_id')
        
        
class LuxuryLogSerializer(serializers.ModelSerializer): 
    
    class Meta:
        model = LuxuryLog
        exclude = ('id', 'luxury_id')        
        
        
class LogSerializer(serializers.Serializer):
    ymd  = serializers.CharField(max_length = 20)
    price = serializers.IntegerField()
    
        