"""
ResNet18 fine-tuning pipeline for food classification.

Использует предобученный ResNet18 из torchvision (ImageNet) и дообучает только верхние слои
на вашем train. Вес модели (~45MB) скачивается с серверов PyTorch (НЕ HuggingFace),
что обычно не блокируется, в отличие от CLIP.

Структура данных:
  Kagle2/
    train/
      labels.csv  (ID или filename/file_name, class, group, language)
      *.jpg|*.jpeg|*.png
    test/
      labels.csv  (ID или filename/file_name, group, language)
      *.jpg|*.jpeg|*.png

Запуск:
  cd Kagle2
  python resnet_finetune_pipeline.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models


ROOT_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER_TRAIN = ROOT_DIR / "train"
IMAGE_FOLDER_TEST = ROOT_DIR / "test"
OUTPUT_CSV = ROOT_DIR / "submit_resnet.csv"

# Локальный файл с весами ResNet18, который вы скачаете вручную
LOCAL_RESNET_WEIGHTS = ROOT_DIR / "models" / "resnet18-f37072fd.pth"


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


class FoodResnetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_folder: Path,
        class2idx: Optional[Dict[str, int]] = None,
        train: bool = True,
        image_size: int = 224,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.train = train
        self.class2idx = class2idx or {}

        if train:
            classes = sorted(self.df["class"].unique().tolist())
            if not self.class2idx:
                self.class2idx = {c: i for i, c in enumerate(classes)}

        # Стандартные трансформации ImageNet + аугментации
        if train:
            self.transform = T.Compose(
                [
                    T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(10),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_id = row["ID"]
        img_path = self.img_folder / img_id

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.train:
            cls = row["class"]
            label = self.class2idx[cls]
            return {"image": image, "label": label, "ID": img_id}
        else:
            return {"image": image, "ID": img_id}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_resnet18(num_classes: int) -> nn.Module:
    """
    Загружаем ResNet18 и инициализируем его весами из локального файла,
    чтобы не делать никаких сетевых запросов.
    """
    if not LOCAL_RESNET_WEIGHTS.exists():
        raise RuntimeError(
            f"Локальный файл весов ResNet18 не найден по пути: {LOCAL_RESNET_WEIGHTS}\n"
            "Скачайте файл 'resnet18-f37072fd.pth' вручную через браузер по ссылке:\n"
            "  https://download.pytorch.org/models/resnet18-f37072fd.pth\n"
            "и положите его в папку 'models' рядом со скриптом."
        )

    # создаём «чистый» ResNet18 без автозагрузки весов
    resnet = models.resnet18(weights=None)

    # грузим state_dict из локального файла
    state_dict = torch.load(str(LOCAL_RESNET_WEIGHTS), map_location="cpu", weights_only=False)
    resnet.load_state_dict(state_dict)

    # Замораживаем все слои, кроме последних блоков и головы
    for name, param in resnet.named_parameters():
        param.requires_grad = False
        # Разморозим последний блок и голову
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True

    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)
    return resnet


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
) -> None:
    criterion = nn.CrossEntropyLoss()

    # Разные learning rate для головы и размороженного бэкбона
    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc"):
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.Adam(
        [
            {"params": head_params, "lr": lr_head},
            {"params": backbone_params, "lr": lr_backbone},
        ]
    )

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

    # Датасеты и лоадеры
    train_dataset = FoodResnetDataset(df_train, IMAGE_FOLDER_TRAIN, class2idx=class2idx, train=True, image_size=224)
    test_dataset = FoodResnetDataset(df_test, IMAGE_FOLDER_TEST, class2idx=class2idx, train=False, image_size=224)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = get_device()
    print(f"Используется устройство: {device}")

    print("Загрузка предобученного ResNet18 и настройка последнего слоя ...")
    model = build_resnet18(num_classes)

    print("Дообучение модели на вашем train (разморожен последний блок + голова, 15 эпох)...")
    train_model(model, train_loader, device, epochs=15, lr_head=1e-3, lr_backbone=1e-4)

    print("Предсказания для test ...")
    preds_test = predict(model, test_loader, device, idx2class)

    df_test_ids = df_test[["ID"]].drop_duplicates()
    merged_submit = pd.merge(df_test_ids, preds_test, on="ID", how="left")

    # На всякий случай заполним пропуски самым частым классом
    most_common_class = df_train["class"].value_counts().idxmax()
    merged_submit["predicted_class"].fillna(most_common_class, inplace=True)

    merged_submit[["ID", "predicted_class"]].to_csv(OUTPUT_CSV, index=False)
    print(f"Файл submit_resnet.csv сохранен по пути: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_pipeline()


