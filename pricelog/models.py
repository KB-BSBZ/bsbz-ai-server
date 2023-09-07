from django.db import models

# Create your models here.
class PriceLog(models.Model):
    product_id = models.IntegerField()
    price = models.IntegerField() # 거래금액
    
    ymd = models.CharField((""), max_length=50)