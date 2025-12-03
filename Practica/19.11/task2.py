import requests
import json

# TASK 2
# функция отправки запроса
def send_request(url:str):
    response = requests.get(url) #отправляет запрос
    if response.status_code == 200: #проеряем код ответа
        return response.json() # возвращаем словарь
    return None # не возвращаем ничего

def save_data(data):
    with open('data.json', mode='w', encoding='utf-8') as file:
        json.dump(data, file)


def update_data(data):
    character_dict = {}
    for index, el in enumerate(data):
        character_dict[str(index)] = el
    return character_dict

def main():
    main_url = "https://swapi.dev/api/people/"
    character_list = list()
    for i in range(1, 83):
        url = main_url + str(i)
        data = send_request(url)
        if data:
            character_list.append(data)

    updated_data = update_data(character_list)
    save_data(updated_data)


if __name__ == '__main__':
    main()