"""
Скрипт для запуска предсказаний на тестовом наборе
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qa_model import predict_with_gigachat_improved, parse_options
import pandas as pd

def main():
    """Основная функция для запуска предсказаний на тестовом наборе"""
    
    # Загрузка токена
    try:
        token = open('authGigaChat.txt').read().strip()
    except FileNotFoundError:
        print("Ошибка: файл authGigaChat.txt не найден!")
        print("Создайте файл authGigaChat.txt с токеном GigaChat")
        return
    
    # Пути к файлам
    train_path = '/Users/aleksey/Downloads/hw-3-questions-and-answering/train.csv'
    test_path = '/Users/aleksey/Downloads/hw-3-questions-and-answering/test.csv'
    
    # Загрузка данных
    print("Загрузка данных...")
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Ошибка при загрузке файлов: {e}")
        print("Убедитесь, что файлы train.csv и test.csv находятся в правильной директории")
        return
    
    print(f"Обучающий набор: {len(df_train)} записей")
    print(f"Тестовый набор: {len(df_test)} записей")
    
    # Предсказания для тестового набора
    print("\n" + "="*60)
    print("Начало предсказаний для тестового набора...")
    print("="*60 + "\n")
    
    df_results = predict_with_gigachat_improved(
        df_test,
        token=token,
        df_train=df_train,
        max_retries=3,
        delay_between_requests=0.1,
        timeout=60,
        batch_size=50,
        delay_between_batches=2.0,
        use_few_shot=True
    )
    
    # Проверка результатов
    print("\n" + "="*60)
    print("Статистика предсказаний:")
    print(f"Всего предсказаний: {len(df_results)}")
    print(f"Успешных предсказаний (0-3): {(df_results['prediction'].between(0, 3)).sum()}")
    print(f"Неудачных предсказаний (-1): {(df_results['prediction'] == -1).sum()}")
    
    # Распределение предсказаний
    print("\nРаспределение предсказаний:")
    print(df_results['prediction'].value_counts().sort_index())
    print("="*60 + "\n")
    
    # Сохранение результатов
    output_file = 'submit.csv'
    df_results.to_csv(output_file, index=False)
    print(f"✓ Результаты сохранены в {output_file}")
    print(f"✓ Формат: id, prediction")
    print(f"✓ Готово к отправке!")

if __name__ == "__main__":
    main()

