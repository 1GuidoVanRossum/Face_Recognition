import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import net  # ваша реализация модели из файла net.py
from torchvision import transforms  # Для Resize

# Обновлённый словарь контрольных точек для модели ir_50
adaface_models = {
    'ir_50': "pretrained/adaface_ir50_webface4m.ckpt",
}


def load_pretrained_model(architecture='ir_50'):
    """
    Загружает модель с заданной архитектурой и весами из контрольной точки.
    """
    assert architecture in adaface_models, f"Architecture {architecture} not found."
    model = net.build_model(architecture)
    checkpoint = torch.load(adaface_models[architecture], map_location='cuda')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    # Убираем префикс 'model.' если он присутствует
    model_statedict = {key[6:]: val for key, val in checkpoint.items() if key.startswith('model.')}
    if len(model_statedict) == 0:
        model_statedict = checkpoint
    model.load_state_dict(model_statedict)
    model.eval()
    model = model.to('cuda')
    return model


def to_input(pil_rgb_image):
    """
    Преобразует PIL-изображение в тензор:
    - Изменяет размер до 112x112
    - Конвертирует RGB в BGR
    - Нормализует значения в диапазоне [-1, 1]
    """
    resize_transform = transforms.Resize((112, 112))
    pil_rgb_image = resize_transform(pil_rgb_image)
    np_img = np.array(pil_rgb_image)
    if np_img.ndim != 3:
        raise ValueError(f"Ожидалось изображение с 3 измерениями, получено {np_img.ndim}")
    # Меняем порядок каналов: RGB -> BGR, приводим к диапазону [-1,1]
    bgr_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor(bgr_img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor  # форма: (1, 3, 112, 112)


class AdafaceDataset(Dataset):
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

        # Проверка существования файла
        if not os.path.exists(path):
            print(f"[WARN] Файл не найден: {path}")
            return None, label

        # Проверка целостности изображения
        try:
            img = Image.open(path)
            img.verify()  # Проверка целостности
            img = Image.open(path)  # Повторное открытие после verify
        except Exception as e:
            print(f"[WARN] Ошибка при открытии изображения {path}: {e}")
            return None, label

        # Преобразуем изображение в тензор
        try:
            tensor = to_input(img)
        except Exception as e:
            print(f"[WARN] Ошибка преобразования изображения {path}: {e}")
            return None, label

        # Убираем размер батча (unsqueeze в to_input) для формирования батчей в DataLoader
        return tensor.squeeze(0), label  # итоговая форма: (3, 112, 112)


def collate_fn(batch):
    images, labels = [], []
    for img, label in batch:
        if img is not None:
            images.append(img)
            labels.append(label)
    if images:
        images = torch.stack(images)  # Форма: [batch_size, 3, 112, 112]
    else:
        images = None
    return images, labels


def main():
    csv_file = "dataset.csv"  # CSV-файл с колонками "image_path" и "subject_id"
    output_file = "embeddings_adaface.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    dataset = AdafaceDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("Загружаем модель adaface (ir_50)...")
    model = load_pretrained_model('ir_50')

    embeddings_list = []
    labels_list = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Извлечение эмбеддингов adaface")):
            if images is None:
                continue
            print(images.shape)  # Проверка формы тензора
            images = images.to(device)
            # Получаем эмбеддинги; модель возвращает кортеж (embedding, norm)
            embedding, _ = model(images)
            embeddings_list.append(embedding.cpu().numpy())
            labels_list.extend(labels)

    elapsed = time.time() - start_time
    print(f"Извлечение эмбеддингов для {len(dataset)} изображений завершено за {elapsed:.2f} секунд.")

    if embeddings_list:
        embeddings_all = np.vstack(embeddings_list)
        labels_all = np.array(labels_list)
    else:
        embeddings_all = np.empty((0, 512), dtype="float32")
        labels_all = np.array([], dtype=object)

    np.savez_compressed(output_file, embeddings=embeddings_all, subject_ids=labels_all)
    print(f"Эмбеддинги сохранены в {output_file}")


if __name__ == "__main__":
    main()