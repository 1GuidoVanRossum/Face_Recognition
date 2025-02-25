import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from GhostFaceNets import __init_model_from_name__
from tqdm import tqdm

def load_ghostfacenet_model(model_path, input_shape=(112,112,3)):
    """
    Создает модель GhostFaceNet (например, ghostnetv1) с заданным input_shape,
    затем загружает веса из файла model_path.
    Для избежания ошибки несоответствия количества слоев используем:
      by_name=True, skip_mismatch=True.
    """
    model = __init_model_from_name__("ghostnetv1", input_shape=input_shape, weights=None)
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    return model

def preprocess_image(image_path, target_size=(112,112)):
    """
    Загружает изображение, изменяет его размер до target_size,
    приводит к массиву float32 и выполняет нормализацию: (img/255.0 - 0.5)/0.5.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    return img_array

def load_dataset(csv_file):
    """
    Загружает CSV-файл и возвращает массив путей к изображениям и массив меток.
    CSV должен содержать колонки "image_path" и "subject_id".
    """
    df = pd.read_csv(csv_file)
    if "Path" not in df.columns or "id" not in df.columns:
        raise ValueError("CSV должен содержать колонки 'image_path' и 'subject_id'")
    return df["Path"].values, df["id"].values

def extract_embeddings(model, image_paths, batch_size=32):
    """
    Извлекает эмбеддинги для списка путей к изображениям.
    Обрабатывает изображения батчами с использованием preprocess_image,
    а затем вычисляет эмбеддинги через model.predict.
    """
    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [preprocess_image(p) for p in batch_paths]
        batch_images = np.stack(batch_images, axis=0)  # shape: (batch, 112,112,3)
        emb = model.predict(batch_images)
        embeddings.append(emb)
    return np.vstack(embeddings)

def main():
    csv_file = "dataset_1.csv"  # CSV с колонками "image_path" и "subject_id"
    model_path = "GhostFaceNet_W1.3_S2_ArcFace.h5"  # Путь к контрольной точке модели
    output_file = "embeddings_ghost_lfw.npz"

    # Конфигурация GPU (если доступен)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU for inference.")
        except Exception as e:
            print("GPU configuration error:", e)
    else:
        print("GPU not available, using CPU.")

    image_paths, subject_ids = load_dataset(csv_file)

    print("Loading original GhostFaceNet model...")
    model = load_ghostfacenet_model(model_path, input_shape=(112,112,3))

    print("Extracting embeddings using GhostFaceNet...")
    emb = extract_embeddings(model, image_paths, batch_size=32)

    np.savez(output_file, embeddings=emb, labels=subject_ids)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()
