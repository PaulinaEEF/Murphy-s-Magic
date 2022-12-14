from django.db import models
from django.utils.safestring import mark_safe

# Create your models here.
class QueryToPredict(models.Model):
    img = models.ImageField(upload_to="media/images")
    title = models.CharField(unique=True, null=True, max_length=100)

    def img_preview(self): #new
        return mark_safe('<img src="%s" style="width: 200px; height:200px;" />' % self.img.url)
