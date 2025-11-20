import requests
import json

# TASK 2
def send_request(url):
    response = requests.get(url) #отправляет запрос
    if response.status_code == 200: #проеряем код ответа
        return response.json() # возвращаем словарь
    return False # не возвращаем ничего

def save_data(data):
    with open('data.json', mode='w', encoding='utf-8') as file:
        json.dump(data, file)


def update_data(data):
    # итоговый словарь
    # для элемента в списке дата
        # итоговый словарь[индекс] = элемент
    # вернуть данные
    result = {}
    for index, element in enumerate(data):
        result[index] = element
    return result

def main():
    url = "https://swapi.dev/api/people/1"
    # список
    characters = []
    # выполните 82 раза
    for i in range(1, 83):
        # данные = запрос по api человека с id = итерация цикла
        url = f"https://swapi.dev/api/people/{i}"
        data = send_request(url)
        # проверяем что строка вернулась не пустота
        if data:  # если данные не None
            # добавляем в список персонажей
            characters.append(data)

    # Обновляем данные в словарь
    updated = update_data(characters)
    save_data(updated)


if __name__ == '__main__':
    main()