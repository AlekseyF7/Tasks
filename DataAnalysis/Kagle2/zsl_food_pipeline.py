"""
Zero-shot image classification pipeline for food dataset.

Ожидаемая структура данных (рядом с этим скриптом):
    ./train/
        labels.csv   (колонки: ID или filename, class, group, language)
        *.jpg|*.jpeg|*.png
    ./test/
        labels.csv   (колонки: ID или filename, group, language; без class)
        *.jpg|*.jpeg|*.png

Скрипт:
  1) Строит промпты для zero-shot на основе class, group, language.
  2) Использует мощную CLIP-модель из transformers (openai/clip-vit-large-patch14).
  3) Классифицирует train для оценки accuracy.
  4) Классифицирует test и сохраняет submit.csv c колонками: ID, predicted_class.

Запуск:
    python zsl_food_pipeline.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Iterable

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import pipeline


ROOT_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER_TRAIN = ROOT_DIR / "train"
IMAGE_FOLDER_TEST = ROOT_DIR / "test"
OUTPUT_CSV = ROOT_DIR / "submit.csv"

# Локальная директория, куда вы сами скачаете модель CLIP
LOCAL_CLIP_DIR = ROOT_DIR / "models" / "openai-clip-vit-base-patch32"


def _normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим имя колонки с ID к единому виду: 'ID'.
    Встречаются варианты: ID, id, filename, file_name.
    """
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    if "id" in cols:
        df.rename(columns={cols["id"]: "ID"}, inplace=True)
    elif "filename" in cols:
        df.rename(columns={cols["filename"]: "ID"}, inplace=True)
    elif "file_name" in cols:
        df.rename(columns={cols["file_name"]: "ID"}, inplace=True)
    if "ID" not in df.columns:
        raise ValueError("Не найдена колонка с ID (ожидались 'ID', 'id', 'filename', 'file_name').")
    return df


def make_zs_prompts(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Генерируем улучшенные текстовые описания классов для zero-shot.

    Возвращает DataFrame с колонками:
        class  - исходное имя класса
        prompt - текстовый промпт для CLIP
    """
    uniq = (
        df_train[["class", "group", "language"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    prompts: List[str] = []
    for _, row in uniq.iterrows():
        cls = str(row["class"])
        group = str(row["group"]) if "group" in row and pd.notna(row["group"]) else ""
        lang = str(row["language"]).lower() if "language" in row and pd.notna(row["language"]) else ""

        # Базовый английский шаблон — CLIP лучше всего работает на en
        if group and group.lower() not in {"nan", "none", ""}:
            base_en = f"a photo of {cls}, a type of {group}"
        else:
            base_en = f"a photo of {cls}"

        # Немного различаем по языку, добавляя подсказку
        if "ru" in lang:
            # Модель CLIP понимает русский хуже, поэтому оставляем английский,
            # но добавляем русскую подсказку как контекст.
            prompt = f"{base_en}. This is a food dish, name originally in Russian."
        else:
            prompt = f"{base_en}. This is a food dish."

        prompts.append(prompt)

    uniq["prompt"] = prompts
    return uniq[["class", "prompt"]]


def build_classifier():
    """
    Инициализация zero-shot-image-classification pipeline с использованием
    уже скачанной локально модели (без обращений в интернет).
    """
    import os

    # Принудительно работаем в оффлайн-режиме HuggingFace
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if not LOCAL_CLIP_DIR.exists():
        raise RuntimeError(
            f"Локальная модель не найдена по пути: {LOCAL_CLIP_DIR}\n"
            "Скачайте содержимое репозитория модели 'openai/clip-vit-base-patch32' "
            "с HuggingFace в эту папку, чтобы в ней лежали файлы вроде:\n"
            "  config.json, merges.txt, vocab.json, preprocessor_config.json, pytorch_model.bin и т.д."
        )

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        # Apple Silicon / MPS
        device = "mps"
    else:
        device = -1

    clf = pipeline(
        "zero-shot-image-classification",
        model=str(LOCAL_CLIP_DIR),
        device=device,
    )
    return clf


def iter_image_paths(folder: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def batch(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    buf: List[Any] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def classify_folder_batched(
    image_folder: Path,
    classifier,
    candidate_labels: List[str],
    id_column_from_fs: bool = True,
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Батчовая классификация изображений в папке.

    Параметры:
        image_folder: папка с изображениями.
        classifier: HF pipeline zero-shot-image-classification.
        candidate_labels: список текстовых промптов для классов.
        id_column_from_fs: если True, ID = имя файла.
        batch_size: размер батча по изображениям.
    """
    image_paths = list(iter_image_paths(image_folder))

    results: List[Dict[str, Any]] = []

    for img_batch in tqdm(list(batch(image_paths, batch_size)), desc=f"Classifying {image_folder.name}"):
        images = []
        ids = []
        for p in img_batch:
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                # пропускаем битые файлы
                continue
            images.append(im)
            ids.append(p.name if id_column_from_fs else str(p))

        if not images:
            continue

        preds_batch = classifier(images, candidate_labels=candidate_labels)

        # HF возвращает список списков (по картинкам)
        for img_id, preds in zip(ids, preds_batch):
            if not preds:
                continue
            top_pred = preds[0]["label"]
            results.append({"ID": img_id, "predicted_class": top_pred})

    return pd.DataFrame(results)


def evaluate_on_train(
    df_train: pd.DataFrame,
    prompts_df: pd.DataFrame,
    classifier,
) -> float:
    """
    Оцениваем accuracy на train (full-train inference).
    WARNING: вычислительно дорого, но дает честную оценку.
    """
    class2prompt = dict(zip(prompts_df["class"], prompts_df["prompt"]))
    candidate_labels = list(class2prompt.values())

    preds_df = classify_folder_batched(
        image_folder=IMAGE_FOLDER_TRAIN,
        classifier=classifier,
        candidate_labels=candidate_labels,
        batch_size=8,
    )

    # Джойним по ID / filename
    df_train_norm = _normalize_id_column(df_train)
    merged = pd.merge(df_train_norm, preds_df, on="ID", how="inner")

    # К маппингу: нам нужно восстановить исходный class из промпта.
    # Мы строили prompts как уникальные для каждого класса, так что можно обратный словарь:
    prompt2class = {v: k for k, v in class2prompt.items()}

    def map_pred_to_class(pred_label: str) -> str:
        return prompt2class.get(pred_label, pred_label)

    merged["pred_class_clean"] = merged["predicted_class"].apply(map_pred_to_class)
    merged["is_correct"] = merged["class"] == merged["pred_class_clean"]
    acc = float(merged["is_correct"].mean())
    return acc


def run_pipeline() -> None:
    print("Загрузка train/labels.csv и test/labels.csv ...")
    df_train = pd.read_csv(IMAGE_FOLDER_TRAIN / "labels.csv")
    df_test = pd.read_csv(IMAGE_FOLDER_TEST / "labels.csv")

    df_train = _normalize_id_column(df_train)
    df_test = _normalize_id_column(df_test)

    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

    print("Формирование улучшенных zero-shot промптов ...")
    prompts_df = make_zs_prompts(df_train)
    print(f"Уникальных классов: {len(prompts_df)}")

    classifier = build_classifier()
    candidate_labels = list(prompts_df["prompt"].values)

    # (опционально) оценка на train — можно закомментировать, если долго
    try:
        print("Оценка accuracy на train (может занять время) ...")
        acc = evaluate_on_train(df_train, prompts_df, classifier)
        print(f"Train accuracy (zero-shot, improved prompts): {acc:.4f}")
    except Exception as e:
        print(f"Не удалось посчитать accuracy на train: {e}")

    print("Классификация test ...")
    preds_test = classify_folder_batched(
        image_folder=IMAGE_FOLDER_TEST,
        classifier=classifier,
        candidate_labels=candidate_labels,
        batch_size=8,
    )

    # Восстановим имя класса из промпта
    prompt2class = {row["prompt"]: row["class"] for _, row in prompts_df.iterrows()}

    def map_pred_to_class(pred_label: str) -> str:
        return prompt2class.get(pred_label, pred_label)

    preds_test["predicted_class"] = preds_test["predicted_class"].apply(map_pred_to_class)

    # Убедимся, что мы выдали предсказание для всех ID из test/labels.csv
    df_test_ids = df_test[["ID"]].drop_duplicates()
    merged_submit = pd.merge(df_test_ids, preds_test, on="ID", how="left")

    # Если по какой-то причине предсказание отсутствует, подставим самый частый класс train
    most_common_class = df_train["class"].value_counts().idxmax()
    merged_submit["predicted_class"].fillna(most_common_class, inplace=True)

    merged_submit[["ID", "predicted_class"]].to_csv(OUTPUT_CSV, index=False)
    print(f"Файл submit.csv сохранен по пути: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_pipeline()


