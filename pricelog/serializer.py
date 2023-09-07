from rest_framework import serializers

class TradeLogSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = ''
        field = '__all__'
        