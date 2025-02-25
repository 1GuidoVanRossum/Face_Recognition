import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# Импорт модели из вашего файла model.py
from model import Backbone


def load_arcface_model(model_path):
    """
    Загружает модель ArcFace (IR-SE50) с весами из model_ir_se50.pth.
    Модель создаётся для 50-слойной версии в режиме 'ir_se'.
    Загружается на CPU.
    """
    model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def quantize_model(model):
    """
    Применяет динамическое квантование к модели (квантуются слои Linear).
    Динамическое квантование работает только на CPU.
    """
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return quantized_model


class ImagesDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        if not {"image_path", "subject_id"}.issubset(self.df.columns):
            raise ValueError("CSV-файл должен содержать колонки 'image_path' и 'subject_id'")
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
            print(f"[WARN] Ошибка чтения {path}: {e}")
            return None, label
        return img, label


def collate_fn(batch):
    images, labels = [], []
    for img, label in batch:
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels


def extract_embeddings(model, dataloader, transform, device):
    model.to(device)
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Извлечение эмбеддингов (квант.)"):
            if not images:
                continue
            imgs = [transform(img) for img in images]
            tensor = torch.stack(imgs).to(device)
            emb = model(tensor)
            embeddings_list.append(emb.cpu().numpy())
            labels_list.append(np.array(labels))
    if embeddings_list:
        embeddings_all = np.vstack(embeddings_list)
        labels_all = np.concatenate(labels_list)
    else:
        embeddings_all = np.empty((0, 512), dtype="float32")
        labels_all = np.array([], dtype=object)
    return embeddings_all, labels_all


def main():
    csv_file = "dataset.csv"  # CSV с колонками image_path и subject_id
    model_path = "model_ir_se50.pth"  # Переобученные веса модели ArcFace
    output_file = "embeddings_quantized_arcface.npz"

    device = torch.device("cpu")  # Квантованная модель работает только на CPU
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImagesDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("Загружаем модель ArcFace (оригинальная, на CPU)...")
    orig_model = load_arcface_model(model_path)
    print("Применяем квантование модели ArcFace...")
    quant_model = quantize_model(orig_model)

    print("Извлекаем эмбеддинги квантованной модели (на CPU)...")
    embeddings, labels = extract_embeddings(quant_model, dataloader, transform, device)
    np.savez(output_file, embeddings=embeddings, labels=labels)
    print(f"Эмбеддинги квантованной модели сохранены в {output_file}")


if __name__ == "__main__":
    main()
