"""
Задание 1: Анализ датасета Iris - ШАБЛОН
Цель: Загрузить датасет Iris, провести анализ целевой переменной и признаков

ЗАДАЧИ:
1. Загрузить данные в функции load_data()
2. Вывести информацию о целевой переменной в target_analysis()
3. Вычислить статистику признаков в feature_statistics()
4. Создать графики распределения целевой переменной в visualize_target()
5. Создать графики распределения признаков в visualize_features()
6. Создать графики признаков по видам цветков в features_by_target()
7. Вычислить матрицу корреляции в correlation_analysis()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Установите русский язык для графиков
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """Загрузить датасет Iris и конвертировать в DataFrame"""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df


def target_analysis(df):
    """Анализ целевой переменной (видов цветков)"""
    print("\nРаспределение видов:")
    counts = df['species'].value_counts()
    print(counts)
    print("\nПроцентное распределение:")
    percents = df['species'].value_counts(normalize=True) * 100
    print(percents)


def feature_statistics(df):
    """Вычислить статистику по признакам"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'target']  # Исключаем 'target'
    stats = df[numeric_cols].describe()
    print("\nСтатистика по признакам:")
    print(stats)


def visualize_target(df):
    """Визуализировать распределение целевой переменной"""
    # TODO: Создать столбчатую диаграмму распределения видов
    # TODO: Создать круговую диаграмму с процентами
    # TODO: Сохранить как 01_iris_target_distribution.png
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Столбчатая диаграмма
    counts = df['species'].value_counts()
    axes[0].bar(counts.index, counts.values)
    axes[0].set_title('Распределение видов (столбчатая)')
    axes[0].set_ylabel('Количество')

    # Круговая диаграмма
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%')
    axes[1].set_title('Распределение видов (круговая)')

    plt.tight_layout()
    plt.savefig('01_iris_target_distribution.png')
    plt.close()


def visualize_features(df):
    """Визуализировать распределение признаков"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'target']  # Исключаем 'target'

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=15, edgecolor='black')
        axes[i].set_title(f'Распределение: {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Частота')

    plt.tight_layout()
    plt.savefig('01_iris_features_distribution.png')
    plt.close()


def features_by_target(df):
    """Визуализировать признаки в разрезе целевой переменной"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'target']  # Исключаем 'target'
    species_list = df['species'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, col in enumerate(numeric_cols):
        for species in species_list:
            data = df[df['species'] == species][col]
            axes[i].hist(data, alpha=0.6, label=species, bins=15)
        axes[i].set_title(f'{col} по видам')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Частота')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('01_iris_features_by_species.png')
    plt.close()


def correlation_analysis(df):
    """Анализ корреляций между признаками"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'target']  # Исключаем 'target'
    corr_matrix = df[numeric_cols].corr()
    print("\nМатрица корреляции:")
    print(corr_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Матрица корреляции признаков')
    plt.tight_layout()
    plt.savefig('01_iris_correlation_matrix.png')
    plt.close()


def main():
    """Главная функция"""
    print("=" * 60)
    print("ЗАДАНИЕ 1: EXPLORATORY DATA ANALYSIS - IRIS DATASET")
    print("=" * 60)

    df = load_data()
    print(f"\nДатасет загружен. Размер: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())

    target_analysis(df)
    feature_statistics(df)
    visualize_target(df)
    visualize_features(df)
    features_by_target(df)
    correlation_analysis(df)

    print("\n" + "=" * 60)
    print("Анализ завершен!")
    print("=" * 60)


if __name__ == "__main__":
    main()
