"""
Модуль для анализа датасета Iris.

Загружает датасет Iris, вычисляет статистику по видам растений,
строит корреляционную матрицу и создаёт scatter-визуализации.
Результаты сохраняются в JSON-файл и PNG-изображение.
"""

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

IRIS_NAMES = ["setosa", "versicolor", "virginica"]


def load_iris_dataset() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Загружает датасет Iris из scikit-learn и преобразует в DataFrame.

    Returns:
        Кортеж из:
            - pd.DataFrame: данные с колонками признаков, target и species,
            - np.ndarray: исходные целевые метки (target).

    Examples:
        >>> df, targets = load_iris_dataset()
        >>> df.shape
        (150, 6)
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].map(
        {i: name for i, name in enumerate(IRIS_NAMES)}
    )
    return df, iris.target


def analyze_species_statistics(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Вычисляет базовую статистику по каждому признаку для каждого вида ириса.

    Args:
        df: DataFrame с колонками признаков, 'target' и 'species'.

    Returns:
        Словарь вида:
        {
            "setosa": {
                "sepal length (cm)": {"mean": , "std": , "min": , "max": },
                ...
            },
            ...
        }

    Examples:
        >>> df, _ = load_iris_dataset()
        >>> stats = analyze_species_statistics(df)
        >>> "setosa" in stats
        True
    """
    feature_names = [
        col for col in df.columns if col not in ["target", "species"]
    ]
    stats = {}

    for species in IRIS_NAMES:
        species_data = df[df["species"] == species]
        species_stats = {}
        for feature in feature_names:
            values = species_data[feature]
            species_stats[feature] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max())
            }
        stats[species] = species_stats

    return stats


def calculate_correlation_matrix(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Tuple[str, str]]:
    """
    Вычисляет корреляционную матрицу Пирсона и находит пару признаков
    с максимальной по модулю корреляцией (исключая диагональ).

    Args:
        df: DataFrame с признаками и метаданными.

    Returns:
        Кортеж из:
            - pd.DataFrame: корреляционная матрица (только признаки),
            - Tuple[str, str]: пара признаков с наибольшей корреляцией.

    Examples:
        >>> df, _ = load_iris_dataset()
        >>> corr, pair = calculate_correlation_matrix(df)
        >>> len(pair) == 2
        True
    """
    feature_names = [
        col for col in df.columns if col not in ["target", "species"]
    ]
    corr_df = df[feature_names].corr()

    # Создаем копию и обнуляем диагональ, чтобы не
    # учитывать корреляцию признака с самим собой
    corr_no_diag = corr_df.copy()
    np.fill_diagonal(corr_no_diag.values, 0)

    # Находим индексы максимального по модулю значения
    max_corr_idx = np.unravel_index(
        np.argmax(np.abs(corr_no_diag.values)),
        corr_no_diag.shape
    )
    feature_list = corr_df.columns.tolist()
    max_corr_pair = (
        feature_list[max_corr_idx[0]], feature_list[max_corr_idx[1]]
    )

    return corr_df, max_corr_pair


def analyze_feature_differences(
        df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Анализирует различия между видами по средним значениям признаков.

    Для каждого признака вычисляется максимальное и минимальное среднее
    среди трёх видов и их разница.

    Args:
        df: DataFrame с признаками и колонкой 'species'.

    Returns:
        Словарь вида:
        {
            "sepal length (cm)": {
                "max_mean": 5.936,
                "min_mean": 5.006,
                "difference": 0.93
            },
            ...
        }

    Examples:
        >>> df, _ = load_iris_dataset()
        >>> diffs = analyze_feature_differences(df)
        >>> "petal length (cm)" in diffs
        True
    """
    feature_names = [
        col for col in df.columns if col not in ["target", "species"]
    ]
    grouped = df.groupby("species")[feature_names].mean()
    differences = {}

    for feature in feature_names:
        max_val = grouped[feature].max()
        min_val = grouped[feature].min()
        differences[feature] = {
            "max_mean": float(max_val),
            "min_mean": float(min_val),
            "difference": float(max_val - min_val)
        }

    return differences


def create_visualization(df: pd.DataFrame) -> None:
    """
    Создаёт два scatter-графика:
    - Длина и ширина чашелистика (sepal)
    - Длина и ширина лепестка (petal)
    Каждый вид отображается своим цветом и легендой.

    Args:
        df: DataFrame с колонками 'species' и признаками.

    Side Effects:
        Сохраняет изображение в файл 'visualization.png'.
        Закрывает фигуру после сохранения.

    Examples:
        >>> df, _ = load_iris_dataset()
        >>> create_visualization(df)  # создаёт файл visualization.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sepal scatter plot: длина vs ширина чашелистика
    for species in IRIS_NAMES:
        species_data = df[df["species"] == species]
        axes[0].scatter(
            species_data["sepal length (cm)"],
            species_data["sepal width (cm)"],
            label=species,
            alpha=0.7
        )
    axes[0].set_xlabel("Sepal Length (cm)")
    axes[0].set_ylabel("Sepal Width (cm)")
    axes[0].set_title("Sepal: Length vs Width")
    axes[0].legend()

    # Petal scatter plot: длина vs ширина лепестка
    for species in IRIS_NAMES:
        species_data = df[df["species"] == species]
        axes[1].scatter(
            species_data["petal length (cm)"],
            species_data["petal width (cm)"],
            label=species,
            alpha=0.7
        )
    axes[1].set_xlabel("Petal Length (cm)")
    axes[1].set_ylabel("Petal Width (cm)")
    axes[1].set_title("Petal: Length vs Width")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("visualization.png")
    plt.close()


def main():
    """
    Основная точка входа: выполняет полный анализ датасета Iris.

    Действия:
        1. Загружает данные.
        2. Вычисляет статистику по видам.
        3. Строит корреляционную матрицу.
        4. Анализирует различия между видами.
        5. Создаёт визуализацию.
        6. Сохраняет результаты в 'iris_results.json'.

    Side Effects:
        Создаёт файлы:
            - iris_results.json
            - visualization.png

    Examples:
        >>> main()  # создаёт файлы результатов
    """
    df, _ = load_iris_dataset()

    stats = analyze_species_statistics(df)
    corr_matrix, highest_corr_pair = calculate_correlation_matrix(df)
    feature_differences = analyze_feature_differences(df)
    create_visualization(df)

    results = {
        "species_statistics": stats,
        "correlation_matrix": corr_matrix.to_dict(),
        "highest_correlation_pair": highest_corr_pair,
        "feature_differences": feature_differences
    }

    with open("iris_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
