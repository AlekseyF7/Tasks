#!/usr/bin/env python3
"""
Скрипт для тестирования улучшений на небольшой выборке
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_model_advanced import predict_with_gigachat_improved, get_default_prediction_by_category
import pandas as pd

def main():
    """Тестирование улучшений на выборке из обучающего набора"""
    
    print("="*70)
    print("Тестирование улучшений модели")
    print("="*70 + "\n")
    
    # Загрузка токена
    token_file = 'authGigaChat.txt'
    try:
        token = open(token_file).read().strip()
    except:
        print(f"❌ Ошибка: файл {token_file} не найден!")
        return
    
    # Загрузка данных
    train_path = '/Users/aleksey/Downloads/hw-3-questions-and-answering/train.csv'
    try:
        df_train = pd.read_csv(train_path)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return
    
    # Берем выборку для тестирования (последние 20 записей, которые не использовались в few-shot)
    test_sample = df_train.tail(20).copy()
    
    print(f"📊 Тестирование на выборке из {len(test_sample)} вопросов\n")
    
    # Тест 1: Стандартный режим
    print("="*70)
    print("Тест 1: Стандартный режим (улучшенный промпт)")
    print("="*70)
    results1 = predict_with_gigachat_improved(
        test_sample,
        token=token,
        df_train=df_train.head(len(df_train) - 20),  # Исключаем тестовую выборку
        delay_between_requests=0.1,
        batch_size=10,
        delay_between_batches=1.0,
        use_few_shot=True,
        use_ensemble=False
    )
    
    df_merged1 = pd.merge(test_sample[['id', 'ответ']], results1, on='id')
    df_merged1['is_correct'] = (df_merged1['ответ'] == df_merged1['prediction']).astype(int)
    acc1 = df_merged1['is_correct'].mean()
    print(f"\n✓ Accuracy: {100*acc1:.1f}% ({df_merged1['is_correct'].sum()}/{len(df_merged1)})\n")
    
    # Тест 2: С ансамблированием (на меньшей выборке для скорости)
    print("="*70)
    print("Тест 2: С ансамблированием (3 предсказания)")
    print("="*70)
    test_sample_small = test_sample.head(5)  # Меньшая выборка для скорости
    results2 = predict_with_gigachat_improved(
        test_sample_small,
        token=token,
        df_train=df_train.head(len(df_train) - 20),
        delay_between_requests=0.1,
        batch_size=5,
        delay_between_batches=1.0,
        use_few_shot=True,
        use_ensemble=True,
        n_ensemble=3
    )
    
    df_merged2 = pd.merge(test_sample_small[['id', 'ответ']], results2, on='id')
    df_merged2['is_correct'] = (df_merged2['ответ'] == df_merged2['prediction']).astype(int)
    acc2 = df_merged2['is_correct'].mean()
    print(f"\n✓ Accuracy: {100*acc2:.1f}% ({df_merged2['is_correct'].sum()}/{len(df_merged2)})\n")
    
    # Итоговое сравнение
    print("="*70)
    print("Сравнение результатов:")
    print("="*70)
    print(f"Стандартный режим:     {100*acc1:.1f}%")
    print(f"С ансамблированием:    {100*acc2:.1f}%")
    print(f"Улучшение:             {100*(acc2-acc1):.1f}%")
    print("="*70)
    
    if acc2 > acc1:
        print("\n✅ Ансамблирование показывает улучшение!")
        print("Рекомендуется использовать режим с ансамблированием для финального запуска.")
    else:
        print("\n💡 Стандартный режим показывает хорошие результаты.")
        print("Можно использовать стандартный режим для баланса скорости и точности.")

if __name__ == "__main__":
    main()

