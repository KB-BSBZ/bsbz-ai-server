from django.db import models

# Create your models here.
class EstateLog(models.Model):
    product_id = models.IntegerField() # 부동산 아이디
    price = models.IntegerField() # 거래금액
    year = models.IntegerField() # 년
    month = models.IntegerField() # 월
    day = models.IntegerField() # 일
    area = models.FloatField() # 면적
    
    
class LuxuryLog(models.Model):
    luxury_id = models.IntegerField() # 부동산 아이디
    ymd = models.CharField(max_length=30) # 년월일
    price = models.IntegerField() # 거래금액

    
class MusicLog(models.Model):
    music_id = models.IntegerField() # 음악 아이디
    ymd = models.CharField(max_length=30) # 년월일
    price_high = models.IntegerField() # 판매 최고가
    price_low = models.IntegerField() # 판매 최저가
    price_close = models.IntegerField() # 종가
    pct_price_change = models.FloatField() # 가격 변동률
    cnt_units_traded = models.IntegerField() # 거래 횟수
    
    
class MusicCommentLog(models.Model):
    comment = models.TextField()
    music_id = models.IntegerField()
    
    
class EstateText():
    contents = models.TextField()
    pubdate = models.DateField()
    
    
class LuxuryText():
    contents = models.TextField()
    pubdate = models.DateField()
    