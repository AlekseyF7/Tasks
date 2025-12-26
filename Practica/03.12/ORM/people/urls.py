from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.get_people_list, name='people_list'),
    path('character/<int:pk>/', views.get_character_info, name='character'),
    path('starship/<int:pk>/', views.get_ship_info, name='starship'),

]