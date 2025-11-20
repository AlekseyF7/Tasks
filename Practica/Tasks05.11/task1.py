import requests
from numpy.ma.core import harden_mask

# Версия 1
# ==================================================
# response = requests.get('https://api.giphy.com/v1/gifts/random')
# data = response.json()
# print(data)
# ==================================================

# Версия 2
# отправляет запрос
def send_request(url):
    response = requests.get(url) #отправляет запрос
    if response.status_code == 200: #проеряем код ответа
        return response.json() # возвращаем словарь
    return False # не возвращаем ничего


def main():
    url = 'https://swapi.dev/api/people/1'
    data = send_request(url)
    name = data['name']
    height = data['height']
    mass = data['mass']
    hair_color = data['hair_color']

    print(f'Имя: {name}\nРост: {height}\nВес: {mass}\nЦвет волос: {hair_color}')

if __name__ == '__main__':
    main()