from django.db import models

# Create your models here.
class EstateLog(models.Model):
    product_id = models.IntegerField()
    price = models.IntegerField() # 거래금액
    year = models.IntegerField() # 년
    month = models.IntegerField() # 월
    day = models.IntegerField() # 일
    area = models.IntegerField() # 면적
    
    
class LuxuryLog(models.Model):
    luxury_id = models.IntegerField()
    ymd = models.CharField(max_length=30) # 년월일
    price = models.IntegerField() # 거래금액

    
class MusicLog(models.Model):
    music_id = models.IntegerField()
    ymd = models.CharField(max_length=30) # 년월일
    price_high = models.IntegerField()
    price_low = models.IntegerField()
    price_close = models.IntegerField()
    pct_price_change = models.IntegerField()
    cnt_units_traded = models.IntegerField() 
    

    
    
    
