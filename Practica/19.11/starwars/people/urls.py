from django.urls import path
from . import views

urlpatterns = [
    path('character/<int:id>', views.get_character, name='character'),
    path('all_characters/', views.get_all_character, name='all_characters'),

]