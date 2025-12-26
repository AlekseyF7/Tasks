from django.contrib import admin

# Register your models here.
from django.contrib import admin

# Register your models here.
from .models import Starship, Charactter




# Register your models here.
class CharacterAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'name',
        'height',
        'mass',
        'hair_color',
        'skin_color',
        'eye_color',
        'birth_year',
        'gender',
        'homeworld',
    )
    list_editable = (
        'height',
        'mass'
    )
    search_fields = (
        'id',
        'name'
    )

    list_filter = ('hair_color',)
    list_display_links = ('name',)




admin.site.register(Starship)
admin.site.register(Charactter, CharacterAdmin)