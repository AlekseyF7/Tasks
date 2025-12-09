#!/usr/bin/env python3
"""
Финальный скрипт для создания submit.csv
Запуск: python create_submit.py
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_model import predict_with_gigachat_improved, get_default_prediction_by_category
import pandas as pd

def main():
    """Основная функция для создания submit.csv"""
    
    print("="*70)
    print("Модель для предсказания правильных ответов")
    print("="*70 + "\n")
    
    # Загрузка токена
    token_file = 'authGigaChat.txt'
    if not os.path.exists(token_file):
        print(f"❌ Ошибка: файл {token_file} не найден!")
        print(f"Создайте файл {token_file} с токеном GigaChat API")
        return
    
    try:
        token = open(token_file).read().strip()
        if not token:
            print(f"❌ Ошибка: файл {token_file} пуст!")
            return
    except Exception as e:
        print(f"❌ Ошибка при чтении {token_file}: {e}")
        return
    
    # Пути к файлам данных
    train_path = '/Users/aleksey/Downloads/hw-3-questions-and-answering/train.csv'
    test_path = '/Users/aleksey/Downloads/hw-3-questions-and-answering/test.csv'
    
    # Проверка существования файлов
    if not os.path.exists(train_path):
        print(f"❌ Ошибка: файл {train_path} не найден!")
        return
    
    if not os.path.exists(test_path):
        print(f"❌ Ошибка: файл {test_path} не найден!")
        return
    
    # Загрузка данных
    print("📂 Загрузка данных...")
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except Exception as e:
        print(f"❌ Ошибка при загрузке файлов: {e}")
        return
    
    print(f"✓ Обучающий набор: {len(df_train)} записей")
    print(f"✓ Тестовый набор: {len(df_test)} записей")
    
    # Статистика по обучающему набору
    print(f"\n📊 Статистика обучающего набора:")
    answer_dist = df_train['ответ'].value_counts().sort_index()
    print(f"   Распределение ответов:")
    for ans, count in answer_dist.items():
        print(f"     {ans}: {count} ({100*count/len(df_train):.1f}%)")
    most_common = df_train['ответ'].mode()[0]
    print(f"   Наиболее частый ответ: {most_common}")
    
    # Предсказания для тестового набора
    print("\n" + "="*70)
    print("🚀 Начало предсказаний для тестового набора...")
    print("="*70 + "\n")
    
    try:
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
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем!")
        print("Частичные результаты будут сохранены...")
        return
    except Exception as e:
        print(f"\n❌ Критическая ошибка при выполнении предсказаний: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Постобработка: заменяем -1 на предсказания на основе категории
    failed_count = (df_results['prediction'] == -1).sum()
    if failed_count > 0:
        print(f"\n🔧 Замена {failed_count} неудачных предсказаний на основе статистики категорий...")
        for idx, row in df_results[df_results['prediction'] == -1].iterrows():
            test_row = df_test[df_test['id'] == row['id']]
            category = test_row['категория'].values[0] if len(test_row) > 0 else None
            df_results.loc[idx, 'prediction'] = get_default_prediction_by_category(df_train, category)
    
    # Проверка результатов
    print("\n" + "="*70)
    print("📈 Статистика предсказаний:")
    print("="*70)
    print(f"   Всего предсказаний: {len(df_results)}")
    valid_predictions = df_results['prediction'].between(0, 3)
    print(f"   Валидных предсказаний (0-3): {valid_predictions.sum()}")
    print(f"   Невалидных предсказаний: {(~valid_predictions).sum()}")
    
    print("\n   Распределение предсказаний:")
    pred_dist = df_results['prediction'].value_counts().sort_index()
    for pred, count in pred_dist.items():
        print(f"     {pred}: {count} ({100*count/len(df_results):.1f}%)")
    
    # Сохранение результатов
    output_file = 'submit.csv'
    df_results.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print(f"✅ Результаты сохранены в {output_file}")
    print("="*70)
    print(f"   Формат: id, prediction")
    print(f"   Размер файла: {os.path.getsize(output_file)} байт")
    print(f"   Готово к отправке!")
    print("="*70)

if __name__ == "__main__":
    main()

