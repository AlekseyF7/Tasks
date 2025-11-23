from django.urls import path

from . import views

urlpatterns = [
    path('1/', views.get_luke_skywalker),

]
