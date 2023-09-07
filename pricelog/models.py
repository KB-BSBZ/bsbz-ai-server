from django.db import models

# Create your models here.
class trade_log(models.Model):
    product_id = models.IntegerField()
    ymd = models.DateField()
    price = models.IntegerField()
    