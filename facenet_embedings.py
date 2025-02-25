import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from facenet_pytorch import MTCNN, InceptionResnetV1

CSV_PATH = "dataset.csv"
OUTPUT_PATH = "embeddings.npz"
BATCH_SIZE = 128
NUM_WORKERS = 16
IMAGE_SIZE = 160
MIN_FACE_SIZE = 20
MARGIN = 0


class ImagesDataset(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        if not {"image_path", "subject_id"}.issubset(self.df.columns):
            raise ValueError("CSV-файл должен содержать колонки 'image_path' и 'subject_id'.")
        self.image_paths = self.df["image_path"].values
        self.labels = self.df["subject_id"].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        if not os.path.exists(img_path):
            print(f"[WARN] Файл не найден: {img_path}")
            return None, label
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Ошибка чтения {img_path}: {e}")
            return None, label
        return image, label


def collate_fn(batch):
    images, labels = [], []
    for img, label in batch:
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels


def process_batch(mtcnn, resnet, images, labels, device):
    """
    Обрабатывает батч изображений по одному:
      - Для каждого изображения вызывается mtcnn для детекции и обрезки лица.
      - Если лицо обнаружено, оно добавляется в список, а метка сохраняется.
      - Собранные лица объединяются в тензор и передаются в resnet для вычисления эмбеддингов.
    """
    face_tensors = []
    valid_labels = []

    with torch.inference_mode():
        for image, label in zip(images, labels):
            face = mtcnn(image)
            if face is not None:
                face_tensors.append(face)
                valid_labels.append(label)

    if len(face_tensors) == 0:
        return None, None

    faces = torch.stack(face_tensors).to(device)
    with torch.inference_mode():
        embeddings = resnet(faces)

    return embeddings.cpu().numpy().astype("float32"), np.array(valid_labels)


def extract_embeddings():
    print(f"[INFO] Загружаем CSV: {CSV_PATH}")
    dataset = ImagesDataset(CSV_PATH)
    print(f"[INFO] Всего записей: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Используем устройство: {device}")

    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=MARGIN,
        min_face_size=MIN_FACE_SIZE,
        device=device,
        keep_all=False
    ).eval()

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    all_embeddings = []
    all_labels = []

    start_time = time.time()
    for images, labels in tqdm(dataloader, desc="Извлечение эмбеддингов"):
        if not images:
            continue
        emb, lbl = process_batch(mtcnn, resnet, images, labels, device)
        if emb is not None:
            all_embeddings.append(emb)
            all_labels.append(lbl)

    total_time = time.time() - start_time

    if all_embeddings:
        embeddings_all = np.vstack(all_embeddings)
        labels_all = np.concatenate(all_labels)
    else:
        print("[WARN] Эмбеддинги не получены.")
        embeddings_all = np.empty((0, 512), dtype="float32")
        labels_all = np.array([], dtype=object)

    print(f"\n[INFO] Всего эмбеддингов: {embeddings_all.shape[0]}")
    print(f"[INFO] Время обработки: {total_time:.2f} сек ({total_time / 60:.1f} мин)")

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    np.savez(OUTPUT_PATH, embeddings=embeddings_all, labels=labels_all)
    print(f"[INFO] Эмбеддинги сохранены в '{OUTPUT_PATH}'")


if __name__ == "__main__":
    extract_embeddings()
