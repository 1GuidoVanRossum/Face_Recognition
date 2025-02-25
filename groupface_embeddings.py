import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from GroupFace import GroupFace


class GroupFaceDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        required_columns = {"image_path", "subject_id"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"CSV-файл должен содержать колонки: {', '.join(required_columns)}")
        self.image_paths = self.df["image_path"].values
        self.labels = self.df["subject_id"].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        if not os.path.exists(path):
            print(f"[WARN] Файл не найден: {path}")
            return None, label
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Ошибка загрузки {path}: {e}")
            return None, label
        return img, label


def collate_fn(batch):
    images, labels = [], []
    for img, label in batch:
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels


def main():
    csv_file = "dataset.csv"  # CSV-файл с колонками "image_path" и "subject_id"
    output_file = "embeddings_groupface.npz"

    # Используем GPU, если доступен, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    # Определяем трансформацию: изменение размера до 112x112, преобразование в тензор и нормализация
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    dataset = GroupFaceDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("Загружаем модель GroupFace (resnet=101, feature_dim=512, groups=4, mode='S')...")
    model = GroupFace(resnet=101, feature_dim=512, groups=4, mode='S')
    model.to(device)
    model.eval()

    embeddings_list = []
    labels_list = []
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Извлечение эмбеддингов GroupFace"):
            if not images:
                continue
            # Применяем трансформацию к каждому изображению в батче
            imgs = [transform(img) for img in images]
            tensor = torch.stack(imgs).to(device)
            # Модель GroupFace возвращает кортеж, где второй элемент – финальное представление
            _, final_embeddings, _, _ = model(tensor)
            embeddings_list.append(final_embeddings.cpu().numpy())
            labels_list.append(np.array(labels))
    elapsed = time.time() - start_time
    print(f"Извлечение эмбеддингов для {len(dataset)} изображений завершено за {elapsed:.2f} секунд.")

    if embeddings_list:
        embeddings_all = np.vstack(embeddings_list)
        labels_all = np.concatenate(labels_list)
    else:
        embeddings_all = np.empty((0, 512), dtype="float32")
        labels_all = np.array([], dtype=object)

    np.savez(output_file, embeddings=embeddings_all, labels=labels_all)
    print(f"Эмбеддинги модели GroupFace сохранены в {output_file}")


if __name__ == "__main__":
    main()
