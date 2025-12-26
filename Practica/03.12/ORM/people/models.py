from django.db import models
# from starships.models import Starship

# Create your models here.
class Charactter(models.Model):
    name = models.TextField()
    height = models.IntegerField()
    mass = models.IntegerField()
    hair_color = models.TextField()
    skin_color = models.TextField()
    eye_color = models.TextField()
    birth_year = models.IntegerField()
    gender = models.TextField()
    homeworld = models.TextField()

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Персонаж'
        verbose_name_plural = 'Персонажи'


# Create your models here.
class Starship(models.Model):
    name = models.TextField()
    model = models.TextField()
    manufacturer = models.TextField()
    cost_in_credits = models.IntegerField()
    length = models.IntegerField()
    max_atmosphering_speed = models.IntegerField()
    crew = models.IntegerField()
    passengers = models.IntegerField()
    cargo_capacity = models.IntegerField()
    consumablesv = models.TextField()
    hyperdrive_rating = models.IntegerField()
    MGLT = models.IntegerField()
    starship_class = models.TextField()
    pilots = models.ManyToManyField(
        Charachter,
        related_name='starships')


    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Корабль'
        verbose_name_plural = 'Корабли'