# Generated by Django 4.1.4 on 2022-12-14 07:32

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0003_querytopredict'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Person',
        ),
    ]