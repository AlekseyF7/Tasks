import re
import time
import json
import pandas as pd
from tqdm import tqdm
from gigachat import GigaChat
from gigachat.models import Chat, MessagesRole
import ast

# Глобальная переменная для кэширования статистики по категориям
_category_stats_cache = None

def parse_options(options_str):
    """Парсит строку с вариантами ответов в список"""
    try:
        # Пробуем распарсить как JSON
        if isinstance(options_str, str):
            # Заменяем одинарные кавычки на двойные для JSON
            options_str = options_str.replace("'", '"')
            options = json.loads(options_str)
        else:
            options = options_str
        return options if isinstance(options, list) else []
    except:
        try:
            # Пробуем через ast.literal_eval
            options = ast.literal_eval(options_str)
            return options if isinstance(options, list) else []
        except:
            return []

def create_few_shot_examples(df_train, n_examples=5):
    """Создает few-shot примеры из обучающего набора"""
    examples = []
    
    # Фильтруем категории (исключаем "-")
    valid_categories = df_train[df_train['категория'] != '-']['категория'].unique()
    
    if len(valid_categories) > 0:
        # Берем примеры из разных категорий
        examples_per_category = max(1, n_examples // min(len(valid_categories), 5))
        
        for category in valid_categories[:min(5, len(valid_categories))]:
            cat_df = df_train[df_train['категория'] == category]
            if len(cat_df) > 0:
                sample = cat_df.sample(min(examples_per_category, len(cat_df)), random_state=42)
                examples.extend(sample.to_dict('records'))
    
    # Если не набрали достаточно, добавляем случайные (включая без категории)
    if len(examples) < n_examples:
        remaining = n_examples - len(examples)
        additional = df_train.sample(min(remaining, len(df_train)), random_state=42)
        examples.extend(additional.to_dict('records'))
    
    # Убираем дубликаты по id
    seen_ids = set()
    unique_examples = []
    for ex in examples:
        if ex['id'] not in seen_ids:
            seen_ids.add(ex['id'])
            unique_examples.append(ex)
        if len(unique_examples) >= n_examples:
            break
    
    return unique_examples[:n_examples]

def get_default_prediction_by_category(df_train, category):
    """Возвращает наиболее частый ответ для данной категории из обучающего набора"""
    global _category_stats_cache
    
    if df_train is None:
        return 1
    
    # Используем кэш для ускорения
    if _category_stats_cache is None:
        _category_stats_cache = {}
        for cat in df_train['категория'].unique():
            cat_df = df_train[df_train['категория'] == cat]
            if len(cat_df) > 0:
                mode_values = cat_df['ответ'].mode()
                _category_stats_cache[cat] = int(mode_values[0]) if len(mode_values) > 0 else 1
    
    if category and category != '-':
        if category in _category_stats_cache:
            return _category_stats_cache[category]
        # Если категория не найдена в кэше, вычисляем
        cat_df = df_train[df_train['категория'] == category]
        if len(cat_df) > 0:
            mode_values = cat_df['ответ'].mode()
            result = int(mode_values[0]) if len(mode_values) > 0 else 1
            _category_stats_cache[category] = result
            return result
    
    # Если категории нет или она не найдена, возвращаем наиболее частый ответ в целом
    mode_values = df_train['ответ'].mode()
    return int(mode_values[0]) if len(mode_values) > 0 else 1

def format_example(question, options, answer_idx, category=None):
    """Форматирует один пример для промпта"""
    options_list = parse_options(options)
    options_text = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options_list)])
    
    example_text = f"Вопрос: {question}\n"
    if category and category != '-':
        example_text += f"Категория: {category}\n"
    example_text += f"Варианты ответов:\n{options_text}\n"
    example_text += f"Правильный ответ: {answer_idx}\n"
    
    return example_text

def qa_message_template_improved(question, answers, category=None, few_shot_examples=None):
    """Улучшенный промпт с few-shot примерами и категорией"""
    
    msg = """Ты эксперт по решению вопросов с множественным выбором. Твоя задача - выбрать правильный вариант ответа.

ВАЖНО: 
- Ответ должен быть ТОЛЬКО числом: 0, 1, 2 или 3
- Не пиши никаких объяснений, только число
- Число соответствует индексу правильного варианта (начиная с 0)

"""
    
    # Добавляем few-shot примеры
    if few_shot_examples:
        msg += "Примеры правильных решений:\n\n"
        for idx, ex in enumerate(few_shot_examples):
            ex_options = parse_options(ex['варианты'])
            ex_text = format_example(
                ex['вопрос'], 
                ex['варианты'], 
                int(ex['ответ']),
                ex.get('категория', None)
            )
            msg += f"Пример {idx + 1}:\n{ex_text}\n"
        msg += "\n" + "-"*60 + "\n\n"
    
    # Текущий вопрос
    options_list = parse_options(answers)
    msg += "Вопрос: " + question + "\n\n"
    
    if category and category != '-':
        msg += f"Категория: {category}\n\n"
    
    msg += "Варианты ответов:\n"
    for i, opt in enumerate(options_list):
        msg += f"{i}. {opt}\n"
    
    msg += "\nВыбери номер правильного варианта. Ответь ТОЛЬКО числом (0, 1, 2 или 3):\n"
    
    return msg

def extract_answer(response_text):
    """Извлекает индекс ответа из текста ответа модели (улучшенная версия)"""
    # Убираем пробелы и приводим к нижнему регистру для поиска
    text = response_text.strip().lower()
    
    # Удаляем лишние символы, но оставляем цифры
    text_clean = re.sub(r'[^\d\s]', ' ', text)
    
    # Пробуем найти число от 0 до 3 в начале строки (наиболее вероятно)
    # Сначала ищем в первых 50 символах
    text_start = text_clean[:50].strip()
    
    # Ищем первое число от 0 до 3
    for num in ['0', '1', '2', '3']:
        # Ищем число как отдельное слово
        pattern = r'\b' + num + r'\b'
        match = re.search(pattern, text_start)
        if match:
            return int(num)
    
    # Если не нашли в начале, ищем во всем тексте
    for num in ['0', '1', '2', '3']:
        pattern = r'\b' + num + r'\b'
        match = re.search(pattern, text_clean)
        if match:
            return int(num)
    
    # Альтернативный подход: ищем все числа и берем первое в диапазоне 0-3
    numbers = re.findall(r'\d+', text_clean)
    for num_str in numbers:
        num = int(num_str)
        if 0 <= num <= 3:
            return num
    
    # Если в тексте есть слова "первый", "второй" и т.д.
    word_to_num = {
        'первый': 0, 'первая': 0, 'первое': 0, 'первым': 0,
        'второй': 1, 'вторая': 1, 'второе': 1, 'вторым': 1,
        'третий': 2, 'третья': 2, 'третье': 2, 'третьим': 2,
        'четвертый': 3, 'четвертая': 3, 'четвертое': 3, 'четвертым': 3,
        'четвёртый': 3, 'четвёртая': 3, 'четвёртое': 3, 'четвёртым': 3
    }
    
    for word, num in word_to_num.items():
        if word in text:
            return num
    
    # Если ничего не найдено, возвращаем -1
    return -1

def predict_with_gigachat_improved(
    df,
    token,
    df_train=None,
    max_retries=3,
    delay_between_requests=0.1,
    timeout=60,
    batch_size=50,
    delay_between_batches=2.0,
    use_few_shot=True
):
    """
    Улучшенная версия функции предсказания с few-shot learning
    
    Параметры:
    - df: DataFrame с тестовыми данными (должен содержать колонки 'id', 'вопрос', 'варианты', 'категория')
    - token: токен для GigaChat API
    - df_train: DataFrame с обучающими данными (для few-shot и статистики по категориям)
    - остальные параметры: настройки обработки запросов
    """
    """
    Улучшенная версия функции предсказания с few-shot learning
    """
    results = []
    
    # Подготавливаем few-shot примеры
    few_shot_examples = None
    if use_few_shot and df_train is not None:
        few_shot_examples = create_few_shot_examples(df_train, n_examples=5)
        print(f"Подготовлено {len(few_shot_examples)} few-shot примеров")
    
    with GigaChat(credentials=token, verify_ssl_certs=False, timeout=timeout) as giga:
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing QA")):
            question = row['вопрос']
            answers = row['варианты']
            category = row.get('категория', None)
            record_id = row['id']
            
            prediction = -1
            
            for attempt in range(max_retries):
                try:
                    msg = qa_message_template_improved(
                        question, 
                        answers, 
                        category=category,
                        few_shot_examples=few_shot_examples
                    )
                    
                    response = giga.chat(msg)
                    response_text = response.choices[0].message.content.strip()
                    
                    prediction = extract_answer(response_text)
                    
                    # Валидация: если получили -1, пробуем еще раз с другим промптом
                    if prediction == -1 and attempt < max_retries - 1:
                        # Упрощенный промпт для повторной попытки
                        options_list = parse_options(answers)
                        simple_msg = f"""Вопрос: {question}
Варианты:
0. {options_list[0] if len(options_list) > 0 else ''}
1. {options_list[1] if len(options_list) > 1 else ''}
2. {options_list[2] if len(options_list) > 2 else ''}
3. {options_list[3] if len(options_list) > 3 else ''}

Выбери номер правильного варианта (0, 1, 2 или 3). Ответь ТОЛЬКО числом:"""
                        try:
                            response = giga.chat(simple_msg)
                            response_text = response.choices[0].message.content.strip()
                            prediction = extract_answer(response_text)
                        except:
                            pass
                    
                    if prediction != -1:
                        break  # Успешный запрос
                        
                except Exception as e:
                    print(f"\nОшибка при обработке id={record_id}, попытка {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1 + attempt)
                    prediction = -1
            
            # Если все попытки неудачны, используем эвристику на основе категории
            if prediction == -1:
                # Используем наиболее частый ответ для данной категории из обучающего набора
                category = row.get('категория', None)
                if df_train is not None:
                    prediction = get_default_prediction_by_category(df_train, category)
                else:
                    prediction = 1  # Дефолтное значение
                print(f"\nПредупреждение: не удалось извлечь ответ для id={record_id}, используется предсказание на основе категории={prediction}")
            
            results.append({'id': record_id, 'prediction': prediction})
            
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)
            
            # Дополнительная задержка после каждого батча
            if (i + 1) % batch_size == 0 and (i + 1) < len(df):
                print(f"\nОбработано {i + 1} запросов, пауза {delay_between_batches} сек...")
                time.sleep(delay_between_batches)
    
    return pd.DataFrame(results)

def main():
    """Основная функция для запуска предсказаний"""
    
    # Загрузка токена
    try:
        token = open('authGigaChat.txt').read().strip()
    except FileNotFoundError:
        print("Ошибка: файл authGigaChat.txt не найден!")
        return
    
    # Загрузка данных
    print("Загрузка данных...")
    train_path = '/Users/aleksey/Downloads/hw-3-questions-and-answering/train.csv'
    test_path = '/Users/aleksey/Downloads/hw-3-questions-and-answering/test.csv'
    
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Ошибка при загрузке файлов: {e}")
        return
    
    print(f"Обучающий набор: {len(df_train)} записей")
    print(f"Тестовый набор: {len(df_test)} записей")
    
    # Статистика по обучающему набору
    print(f"\nСтатистика обучающего набора:")
    print(f"Распределение ответов: {df_train['ответ'].value_counts().sort_index().to_dict()}")
    print(f"Наиболее частый ответ: {df_train['ответ'].mode()[0]}")
    
    # Проверка точности на обучающем наборе (опционально)
    print("\n" + "="*50)
    print("Проверка точности на обучающем наборе (первые 20 записей)...")
    df_train_sample = df_train.head(20)
    results_train = predict_with_gigachat_improved(
        df_train_sample,
        token=token,
        df_train=df_train,
        delay_between_requests=0.1,
        batch_size=10,
        delay_between_batches=1.0,
        use_few_shot=True
    )
    
    # Вычисление accuracy
    df_merged = pd.merge(
        df_train_sample[['id', 'ответ']], 
        results_train, 
        on='id'
    )
    df_merged['is_correct'] = (df_merged['ответ'] == df_merged['prediction']).astype(int)
    acc = df_merged['is_correct'].mean()
    print(f"\nAccuracy на выборке из обучающего набора: {100*acc:.1f}%")
    print("="*50 + "\n")
    
    # Предсказания для тестового набора
    print("Начало предсказаний для тестового набора...")
    df_results = predict_with_gigachat_improved(
        df_test,
        token=token,
        df_train=df_train,
        delay_between_requests=0.1,
        batch_size=50,
        delay_between_batches=2.0,
        use_few_shot=True
    )
    
    # Постобработка: заменяем -1 на предсказания на основе категории
    failed_count = (df_results['prediction'] == -1).sum()
    if failed_count > 0:
        print(f"\nЗамена {failed_count} неудачных предсказаний на основе статистики категорий...")
        for idx, row in df_results[df_results['prediction'] == -1].iterrows():
            test_row = df_test[df_test['id'] == row['id']]
            category = test_row['категория'].values[0] if len(test_row) > 0 else None
            df_results.loc[idx, 'prediction'] = get_default_prediction_by_category(df_train, category)
    
    # Сохранение результатов
    output_file = 'submit.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nРезультаты сохранены в {output_file}")
    print(f"Всего предсказаний: {len(df_results)}")
    print(f"Успешных предсказаний (0-3): {(df_results['prediction'].between(0, 3)).sum()}")

if __name__ == "__main__":
    main()

