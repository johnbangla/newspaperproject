

# Create your models here.
from django.db import models

# Create your models here.
class News(models.Model):
    id = models.AutoField(primary_key=True)
    news = models.CharField(max_length=30, unique=True)


    def __str__(self):
	    return self.name