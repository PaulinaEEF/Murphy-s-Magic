from rest_framework import serializers
from api.models import QueryToPredict

class QuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = QueryToPredict
        fields = ['img', 'title']