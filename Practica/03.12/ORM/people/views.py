from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from people.models import Charactter, Starship


# Create your views here.
def get_people_list(request):
    template_name = 'people.html'
    people_list = Charactter.objects.all()
    print(people_list[0].pk)

    context = {'people_list': people_list}
    return render(request, template_name, context)


def get_character_info(request, pk):
    template_name = 'character.html'
    data = Charactter.objects.get(pk=pk)
    ships = data.starships.all()
    context = {'character': data, 'starships': ships}
    return render(request, template_name, context)


def get_ship_info(request, pk):
    template_name = 'ship.html'
    data = Starship.objects.get(pk=pk)
    context = {'starship': data}
    return render(request, template_name, context)
