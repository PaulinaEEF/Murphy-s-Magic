from django.urls import path
from api.views import api_get_query_view, api_create_query_view, api_delete_query_view


app_name = 'Prediction'

urlpatterns = [
    path('<slug>/deleteQuery', api_delete_query_view, name="deleteQuery"),
    path('<slug>/', api_get_query_view, name="getQuery"),
    path('createQuery', api_create_query_view, name="createQuery"),

]