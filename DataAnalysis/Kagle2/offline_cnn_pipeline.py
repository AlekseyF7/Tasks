"""
Offline supervised image classification pipeline without external model downloads.

Использует:
  - только локальные библиотеки (torch, pandas, pillow, numpy);
  - простой сверточный CNN, обучаемый с нуля на train;
  - предсказания для test и сохранение submit.csv.

Ожидаемая структура (та же, что и для zero-shot скрипта):
  Kagle2/
    train/
      labels.csv  (ID, class, group, language)
      *.jpg|*.jpeg|*.png
    test/
      labels.csv  (ID, group, language)
      *.jpg|*.jpeg|*.png

Запуск:
  cd Kagle2
  python offline_cnn_pipeline.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


ROOT_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER_TRAIN = ROOT_DIR / "train"
IMAGE_FOLDER_TEST = ROOT_DIR / "test"
OUTPUT_CSV = ROOT_DIR / "submit_cnn.csv"


def _normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
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


class FoodDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_folder: Path,
        class2idx: Optional[Dict[str, int]] = None,
        train: bool = True,
        img_size: int = 128,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.train = train
        self.img_size = img_size
        self.class2idx = class2idx or {}

        if train:
            classes = sorted(self.df["class"].unique().tolist())
            if not self.class2idx:
                self.class2idx = {c: i for i, c in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, img_id: str) -> torch.Tensor:
        img_path = self.img_folder / img_id
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0  # HWC, [0,1]
        # простой нормализации достаточно
        arr = arr.transpose(2, 0, 1)  # CHW
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_id = row["ID"]
        x = self._load_image(img_id)

        if self.train:
            cls = row["class"]
            y = self.class2idx[cls]
            return {"image": x, "label": y, "ID": img_id}
        else:
            return {"image": x, "ID": img_id}


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # при img_size=128 после 3 пула: 128/(2^3)=16
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=running_loss / max(total, 1),
                acc=correct / max(total, 1),
            )

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    idx2class: Dict[int, str],
) -> pd.DataFrame:
    model.eval()
    preds: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predict"):
            images = batch["image"].to(device)
            ids = batch["ID"]
            outputs = model(images)
            _, pred_idx = outputs.max(1)
            pred_idx = pred_idx.cpu().numpy()
            for img_id, idx_val in zip(ids, pred_idx):
                preds.append({"ID": img_id, "predicted_class": idx2class[int(idx_val)]})

    return pd.DataFrame(preds)


def run_pipeline() -> None:
    print("Загрузка train/labels.csv и test/labels.csv ...")
    df_train = pd.read_csv(IMAGE_FOLDER_TRAIN / "labels.csv")
    df_test = pd.read_csv(IMAGE_FOLDER_TEST / "labels.csv")

    df_train = _normalize_id_column(df_train)
    df_test = _normalize_id_column(df_test)

    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

    classes = sorted(df_train["class"].unique().tolist())
    num_classes = len(classes)
    print(f"Уникальных классов: {num_classes}")

    class2idx = {c: i for i, c in enumerate(classes)}
    idx2class = {i: c for c, i in class2idx.items()}

    # Датасеты и даталоадеры
    train_dataset = FoodDataset(df_train, IMAGE_FOLDER_TRAIN, class2idx=class2idx, train=True, img_size=128)
    test_dataset = FoodDataset(df_test, IMAGE_FOLDER_TEST, class2idx=class2idx, train=False, img_size=128)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = get_device()
    print(f"Используется устройство: {device}")

    model = SimpleCNN(num_classes=num_classes)
    print("Обучение модели (без предобученных весов, может занять несколько минут)...")
    train_model(model, train_loader, device, epochs=15, lr=1e-3)

    print("Предсказания для test ...")
    preds_test = predict(model, test_loader, device, idx2class)

    # Убедимся, что есть предсказание для каждого ID
    df_test_ids = df_test[["ID"]].drop_duplicates()
    merged_submit = pd.merge(df_test_ids, preds_test, on="ID", how="left")

    # На случай отсутствия предсказаний для каких-то ID (не должно случиться)
    most_common_class = df_train["class"].value_counts().idxmax()
    merged_submit["predicted_class"].fillna(most_common_class, inplace=True)

    merged_submit[["ID", "predicted_class"]].to_csv(OUTPUT_CSV, index=False)
    print(f"Файл submit_cnn.csv сохранен по пути: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_pipeline()


