"""
Unit тесты для варианта 8 - Iris Flower Classification Analysis.
"""

import unittest
import numpy as np
import pandas as pd
from iris_analysis import (
    load_iris_dataset,
    calculate_correlation_matrix,
    analyze_species_statistics,
    analyze_feature_differences,
    create_visualization,
    main
)


class TestIrisAnalysis(unittest.TestCase):
    """Тесты для анализа Iris датасета."""

    def setUp(self):
        """Подготовка данных перед каждым тестом."""
        self.df, _ = load_iris_dataset()

    def test_iris_species_count(self):
        """Тест: должно быть 3 вида."""
        species_unique = self.df["species"].unique()
        self.assertEqual(len(species_unique), 3)

    def test_feature_count(self):
        """Тест: должно быть 4 признака."""
        feature_names = [col for col in self.df.columns if col not in ["target", "species"]]
        self.assertEqual(len(feature_names), 4)

    def test_correlation_matrix_is_square(self):
        """Тест: корреляционная матрица квадратная."""
        corr_matrix, _ = calculate_correlation_matrix(self.df)
        rows, cols = corr_matrix.shape
        self.assertEqual(rows, cols)

    def test_correlation_values_in_range(self):
        """Тест: корреляция в [-1, 1]."""
        corr_matrix, _ = calculate_correlation_matrix(self.df)
        self.assertTrue((corr_matrix.values >= -1.0).all().all())
        self.assertTrue((corr_matrix.values <= 1.0).all().all())


if __name__ == "__main__":
    unittest.main()