from django.contrib import admin

# Register your models here.

from api.models import QueryToPredict

class QueryAdmin(admin.ModelAdmin):
    readonly_fields = ['img_preview']
    list_display = ["title", "img_preview"]

admin.site.register(QueryToPredict, QueryAdmin)