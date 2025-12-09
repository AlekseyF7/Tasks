import re
import time
import json
import pandas as pd
from tqdm import tqdm
from gigachat import GigaChat
from gigachat.models import Chat, MessagesRole
import ast
from collections import Counter

# Глобальная переменная для кэширования статистики по категориям
_category_stats_cache = None

def parse_options(options_str):
    """Парсит строку с вариантами ответов в список"""
    try:
        if isinstance(options_str, str):
            options_str = options_str.replace("'", '"')
            options = json.loads(options_str)
        else:
            options = options_str
        return options if isinstance(options, list) else []
    except:
        try:
            options = ast.literal_eval(options_str)
            return options if isinstance(options, list) else []
        except:
            return []

def create_category_specific_examples(df_train, category, n_examples=3):
    """Создает few-shot примеры из той же категории"""
    if category and category != '-':
        cat_df = df_train[df_train['категория'] == category]
        if len(cat_df) >= n_examples:
            return cat_df.sample(n_examples, random_state=42).to_dict('records')
        elif len(cat_df) > 0:
            return cat_df.to_dict('records')
    
    # Если категории нет или примеров недостаточно, берем общие
    return df_train.sample(min(n_examples, len(df_train)), random_state=42).to_dict('records')

def create_few_shot_examples(df_train, category=None, n_examples=7):
    """Создает few-shot примеры с приоритетом на категорию"""
    examples = []
    
    # Если есть категория, берем примеры из нее
    if category and category != '-':
        cat_examples = create_category_specific_examples(df_train, category, n_examples=min(3, n_examples))
        examples.extend(cat_examples)
    
    # Дополняем примерами из других категорий
    remaining = n_examples - len(examples)
    if remaining > 0:
        valid_categories = df_train[df_train['категория'] != '-']['категория'].unique()
        if len(valid_categories) > 0:
            examples_per_category = max(1, remaining // min(len(valid_categories), 3))
            for cat in valid_categories[:3]:
                if cat != category:
                    cat_df = df_train[df_train['категория'] == cat]
                    if len(cat_df) > 0:
                        sample = cat_df.sample(min(examples_per_category, len(cat_df)), random_state=42)
                        examples.extend(sample.to_dict('records'))
                        if len(examples) >= n_examples:
                            break
    
    # Если все еще не хватает, добавляем случайные
    if len(examples) < n_examples:
        remaining = n_examples - len(examples)
        additional = df_train.sample(min(remaining, len(df_train)), random_state=42)
        examples.extend(additional.to_dict('records'))
    
    # Убираем дубликаты
    seen_ids = set()
    unique_examples = []
    for ex in examples:
        if ex['id'] not in seen_ids:
            seen_ids.add(ex['id'])
            unique_examples.append(ex)
        if len(unique_examples) >= n_examples:
            break
    
    return unique_examples[:n_examples]

def format_example_with_reasoning(question, options, answer_idx, category=None):
    """Форматирует пример с объяснением"""
    options_list = parse_options(options)
    options_text = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options_list)])
    
    example_text = f"Вопрос: {question}\n"
    if category and category != '-':
        example_text += f"Категория: {category}\n"
    example_text += f"Варианты ответов:\n{options_text}\n"
    example_text += f"Правильный ответ: {answer_idx}\n"
    
    return example_text

def qa_message_template_advanced(question, answers, category=None, few_shot_examples=None, use_reasoning=True):
    """Продвинутый промпт с chain-of-thought reasoning"""
    
    if use_reasoning:
        msg = """Ты эксперт по решению вопросов с множественным выбором. Решай задачи пошагово:

1. Внимательно прочитай вопрос
2. Проанализируй каждый вариант ответа
3. Исключи явно неправильные варианты
4. Выбери наиболее правильный ответ
5. Ответь ТОЛЬКО числом (0, 1, 2 или 3) без дополнительного текста

"""
    else:
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
            ex_text = format_example_with_reasoning(
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
    
    if use_reasoning:
        msg += "\nПроанализируй вопрос и варианты, затем выбери правильный ответ. Ответь ТОЛЬКО числом (0, 1, 2 или 3):\n"
    else:
        msg += "\nВыбери номер правильного варианта. Ответь ТОЛЬКО числом (0, 1, 2 или 3):\n"
    
    return msg

def extract_answer(response_text):
    """Извлекает индекс ответа из текста ответа модели (улучшенная версия)"""
    text = response_text.strip().lower()
    text_clean = re.sub(r'[^\d\s]', ' ', text)
    text_start = text_clean[:50].strip()
    
    # Ищем первое число от 0 до 3
    for num in ['0', '1', '2', '3']:
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
    
    return -1

def get_default_prediction_by_category(df_train, category):
    """Возвращает наиболее частый ответ для данной категории из обучающего набора"""
    global _category_stats_cache
    
    if df_train is None:
        return 1
    
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
        cat_df = df_train[df_train['категория'] == category]
        if len(cat_df) > 0:
            mode_values = cat_df['ответ'].mode()
            result = int(mode_values[0]) if len(mode_values) > 0 else 1
            _category_stats_cache[category] = result
            return result
    
    mode_values = df_train['ответ'].mode()
    return int(mode_values[0]) if len(mode_values) > 0 else 1

def predict_with_ensemble(
    df,
    token,
    df_train=None,
    n_ensemble=3,
    max_retries=3,
    delay_between_requests=0.1,
    timeout=60,
    batch_size=50,
    delay_between_batches=2.0,
    use_few_shot=True,
    use_reasoning=True
):
    """
    Улучшенная версия с ансамблированием (несколько предсказаний и голосование)
    """
    results = []
    
    with GigaChat(credentials=token, verify_ssl_certs=False, timeout=timeout) as giga:
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing QA (Ensemble)")):
            question = row['вопрос']
            answers = row['варианты']
            category = row.get('категория', None)
            record_id = row['id']
            
            predictions = []
            
            # Делаем несколько предсказаний (ансамблирование)
            for ensemble_idx in range(n_ensemble):
                # Подготавливаем few-shot примеры для каждого предсказания
                few_shot_examples = None
                if use_few_shot and df_train is not None:
                    few_shot_examples = create_few_shot_examples(
                        df_train, 
                        category=category, 
                        n_examples=7 if ensemble_idx == 0 else 5  # Первое предсказание с большим количеством примеров
                    )
                
                prediction = -1
                
                for attempt in range(max_retries):
                    try:
                        msg = qa_message_template_advanced(
                            question, 
                            answers, 
                            category=category,
                            few_shot_examples=few_shot_examples,
                            use_reasoning=use_reasoning and ensemble_idx == 0  # Reasoning только для первого
                        )
                        
                        response = giga.chat(msg)
                        response_text = response.choices[0].message.content.strip()
                        prediction = extract_answer(response_text)
                        
                        if prediction != -1:
                            break
                            
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"\nОшибка при обработке id={record_id}, ensemble={ensemble_idx}: {e}")
                        time.sleep(0.5 + attempt)
                
                if prediction != -1:
                    predictions.append(prediction)
                
                # Небольшая задержка между предсказаниями в ансамбле
                if ensemble_idx < n_ensemble - 1:
                    time.sleep(0.1)
            
            # Голосование: берем наиболее частый ответ
            if len(predictions) > 0:
                final_prediction = Counter(predictions).most_common(1)[0][0]
            else:
                # Если все предсказания неудачны, используем статистику по категории
                final_prediction = get_default_prediction_by_category(df_train, category)
            
            results.append({'id': record_id, 'prediction': final_prediction})
            
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)
            
            # Дополнительная задержка после каждого батча
            if (i + 1) % batch_size == 0 and (i + 1) < len(df):
                print(f"\nОбработано {i + 1} запросов, пауза {delay_between_batches} сек...")
                time.sleep(delay_between_batches)
    
    return pd.DataFrame(results)

def predict_with_gigachat_improved(
    df,
    token,
    df_train=None,
    max_retries=3,
    delay_between_requests=0.1,
    timeout=60,
    batch_size=50,
    delay_between_batches=2.0,
    use_few_shot=True,
    use_ensemble=False,
    n_ensemble=3
):
    """
    Улучшенная версия функции предсказания с опцией ансамблирования
    """
    if use_ensemble:
        return predict_with_ensemble(
            df, token, df_train, n_ensemble=n_ensemble,
            max_retries=max_retries,
            delay_between_requests=delay_between_requests,
            timeout=timeout,
            batch_size=batch_size,
            delay_between_batches=delay_between_batches,
            use_few_shot=use_few_shot
        )
    
    # Стандартная версия без ансамблирования
    results = []
    
    few_shot_examples = None
    if use_few_shot and df_train is not None:
        few_shot_examples = create_few_shot_examples(df_train, n_examples=7)
        print(f"Подготовлено {len(few_shot_examples)} few-shot примеров")
    
    with GigaChat(credentials=token, verify_ssl_certs=False, timeout=timeout) as giga:
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing QA")):
            question = row['вопрос']
            answers = row['варианты']
            category = row.get('категория', None)
            record_id = row['id']
            
            # Подготавливаем few-shot примеры с учетом категории
            current_few_shot = None
            if use_few_shot and df_train is not None:
                current_few_shot = create_few_shot_examples(df_train, category=category, n_examples=7)
            
            prediction = -1
            
            for attempt in range(max_retries):
                try:
                    msg = qa_message_template_advanced(
                        question, 
                        answers, 
                        category=category,
                        few_shot_examples=current_few_shot,
                        use_reasoning=True
                    )
                    
                    response = giga.chat(msg)
                    response_text = response.choices[0].message.content.strip()
                    prediction = extract_answer(response_text)
                    
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
                        break
                        
                except Exception as e:
                    print(f"\nОшибка при обработке id={record_id}, попытка {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1 + attempt)
                    prediction = -1
            
            if prediction == -1:
                category = row.get('категория', None)
                if df_train is not None:
                    prediction = get_default_prediction_by_category(df_train, category)
                else:
                    prediction = 1
                print(f"\nПредупреждение: не удалось извлечь ответ для id={record_id}, используется предсказание на основе категории={prediction}")
            
            results.append({'id': record_id, 'prediction': prediction})
            
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)
            
            if (i + 1) % batch_size == 0 and (i + 1) < len(df):
                print(f"\nОбработано {i + 1} запросов, пауза {delay_between_batches} сек...")
                time.sleep(delay_between_batches)
    
    return pd.DataFrame(results)

