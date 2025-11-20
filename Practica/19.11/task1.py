import requests

# TASK 1
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