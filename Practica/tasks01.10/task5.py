#задание конвертер чисел в римские и обратно
def arabic_to_roman(n: int) -> str:
    if not (1 <= n <= 3999):
        raise ValueError("Число должно быть от 1 до 3999")
    values = [
        (1000, 'M'),
        (900, 'CM'),
        (500, 'D'),
        (400, 'CD'),
        (100, 'C'),
        (90, 'XC'),
        (50, 'L'),
        (40, 'XL'),
        (10, 'X'),
        (9, 'IX'),
        (5, 'V'),
        (4, 'IV'),
        (1, 'I')]

    roman = ''
    for value, numeral in values:
        while n >= value:
            roman += numeral
            n -= value
    return roman

def roman_to_arabic(s: str) -> int:
    roman_map = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000}
    if not all(c in roman_map for c in s):
        raise ValueError("Недопустимые символы в римском числе")
    total = 0
    prev_value = 0
    for char in reversed(s):
        value = roman_map[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    #Проверка корректности:
    if arabic_to_roman(total) != s:
        raise ValueError("Некорректное римское число")
    return total

def is_arabic(s: str) -> bool:
    #Проверяет корректность числа
    if not s.isdigit():
        return False
    num = int(s)
    return 1 <= num <= 3999


def is_roman(s: str) -> bool:
    #Проверяет коректность римского числа
    return all(c in "IVXLCDM" for c in s.upper())

def main():
    print("Перевод чисел: арабские <-> римские")
    print("Введите число или 'выход' для завершения.\n")

    while True:
        user_input = input("Ваш ввод: ").strip()
        if user_input.lower() in ('выход', 'exit', 'quit', 'q'):
            print("Это конец :(")
            break
        if not user_input:
            print("Пустой ввод! Попробуйте снова.\n")
            continue
        try:
            if is_arabic(user_input):
                num = int(user_input)
                roman = arabic_to_roman(num)
                print(f"-> Римское: {roman}\n")
            elif is_roman(user_input):
                roman_str = user_input.upper()
                arabic = roman_to_arabic(roman_str)
                print(f"-> Арабское: {arabic}\n")
            else:
                print(
                    "Не удалось распознать ввод. Убедитесь, что это число от 1 до 3999 или корректное римское число (I, V, X, L, C, D, M).\n")
        except ValueError as e:
            print(f"Ошибка: {e}\n")
        except Exception as e:
            print(f"Неожиданная ошибка: {e}\n")

if __name__ == "__main__":
    # Можно запустить и тесты, и интерактивный режим
    roman_tests = ["IV", "IX", "XLII", "XCIX", "MMXXIII"]
    print("Тестовые примеры:")
    for r in roman_tests:
        a = roman_to_arabic(r)
        back = arabic_to_roman(a)
        print(f"{r} → {a} → {back}")
    print("-" * 30 + "\n")
    main()