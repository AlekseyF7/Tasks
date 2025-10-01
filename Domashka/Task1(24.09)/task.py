text = input('Введите вашу строку: ').lower()
if len(text)==0:
    print("пустая строка")
elif len(text)==1:
    print(f"{text}:1")
elif len(text)==2:
    el = []
    for s in text:
        el += s
    if el[0] == el[1]:
        print(f"{text}:2")
    else:
        print(f"{el[0]}:1")
        print(f"{el[1]}:1")

else:
    char_count = {}
    for char in text:
        if char in char_count.keys():
            char_count[char] +=1
        else:
            char_count[char]=1

    char_list = []
    for char, count in char_count.items():
        char_list.append((char, count))

    for i in range(len(char_list)):
        for j in range(i+1, len(char_list)):
            if char_list[i][1] < char_list[j][1]:
                char_list[i], char_list[j] = char_list[j], char_list[i]

    for i in range(3):
        char, count = char_list[i]
        print(f"{i+1}){char}:{count}")