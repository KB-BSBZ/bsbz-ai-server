# Generated by Django 3.2.7 on 2023-09-22 07:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pricelog', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='MusicCommentLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('comment', models.TextField()),
                ('music_id', models.IntegerField()),
            ],
        ),
    ]