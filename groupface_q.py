import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from face_alignment import align
from inference import load_pretrained_model, to_input
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

def quantize_model(model):
    """
    Применяет динамическое квантование к модели AdaFace для слоев Linear.
    Динамическое квантование поддерживается только на CPU, поэтому переводим модель на CPU.
    """
    model.cpu()  # перевод модели на CPU
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return quantized_model

class AdaFaceDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        required = {"image_path", "subject_id"}
        if not required.issubset(self.df.columns):
            raise ValueError("CSV-файл должен содержать колонки: " + ", ".join(required))
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
            aligned_rgb = align.get_aligned_face(path)
            # Если результат не имеет ожидаемой размерности (H, W, 3), пропускаем изображение
            if aligned_rgb is None or not (hasattr(aligned_rgb, 'shape') and len(aligned_rgb.shape) == 3):
                # Можно опционально вывести предупреждение, затем пропустить
                # print(f"[WARN] Неверная размерность выровненного изображения для {path}. Пропускаем.")
                return None, label
            bgr_input = to_input(aligned_rgb)
        except Exception as e:
            # Если произошла ошибка при обработке, пропускаем изображение
            # print(f"[WARN] Ошибка обработки {path}: {e}")
            return None, label
        return bgr_input, label

def collate_fn(batch):
    inputs, labels = zip(*batch)
    valid_inputs = [inp for inp in inputs if inp is not None]
    valid_labels = [lab for inp, lab in batch if inp is not None]
    if valid_inputs:
        inputs_tensor = torch.cat(valid_inputs, dim=0)
    else:
        inputs_tensor = None
    return inputs_tensor, valid_labels

def extract_embeddings(model, dataloader, device):
    model.to(device)
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Извлечение эмбеддингов AdaFace (Quantized)"):
            if inputs is None:
                continue
            inputs = inputs.to(device)
            feature, _ = model(inputs)
            embeddings_list.append(feature.cpu().numpy())
            labels_list.append(np.array(labels))
    if embeddings_list:
        embeddings_all = np.vstack(embeddings_list)
        labels_all = np.concatenate(labels_list)
    else:
        embeddings_all = np.empty((0, feature.shape[-1]), dtype="float32")
        labels_all = np.array([], dtype=object)
    return embeddings_all, labels_all

def main():
    csv_file = "dataset.csv"          # CSV-файл с колонками "image_path" и "subject_id"
    output_file = "embeddings_adaface_quantized.npz"

    # Для квантованной модели используем только CPU
    device = torch.device("cpu")
    print("Используем устройство:", device)

    dataset = AdaFaceDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("Загружаем модель AdaFace...")
    model = load_pretrained_model('ir_50')

    print("Применяем динамическое квантование модели AdaFace (на CPU)...")
    quant_model = quantize_model(model)

    print("Извлекаем эмбеддинги с помощью квантованной модели AdaFace (на CPU)...")
    embeddings, labels = extract_embeddings(quant_model, dataloader, device)
    np.savez(output_file, embeddings=embeddings, labels=labels)
    print(f"Эмбеддинги квантованной модели AdaFace сохранены в {output_file}")

if __name__ == "__main__":
    main()
