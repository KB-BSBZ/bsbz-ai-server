# Generated by Django 3.2.7 on 2023-09-12 07:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='EstateLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product_id', models.IntegerField()),
                ('price', models.IntegerField()),
                ('year', models.IntegerField()),
                ('month', models.IntegerField()),
                ('day', models.IntegerField()),
                ('area', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='LuxuryLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('luxury_id', models.IntegerField()),
                ('ymd', models.CharField(max_length=30)),
                ('price', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='MusicLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('music_id', models.IntegerField()),
                ('ymd', models.CharField(max_length=30)),
                ('price_high', models.IntegerField()),
                ('price_low', models.IntegerField()),
                ('price_close', models.IntegerField()),
                ('pct_price_change', models.FloatField()),
                ('cnt_units_traded', models.IntegerField()),
            ],
        ),
    ]