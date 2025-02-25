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
    if "image_path" not in df.columns or "subject_id" not in df.columns:
        raise ValueError("CSV должен содержать колонки 'image_path' и 'subject_id'")
    return df["image_path"].values, df["subject_id"].values

def quantize_model(model):
    """
    Квантует модель с использованием посттренировочного квантования.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()
    return quantized_model


def extract_embeddings_tflite(quantized_model, image_paths, batch_size=32):
    interpreter = tf.lite.Interpreter(model_content=quantized_model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = np.stack([preprocess_image(p) for p in batch_paths], axis=0)  # shape: (batch, 112,112,3)

        # Изменяем размер входного тензора под размер батча
        interpreter.resize_tensor_input(input_details[0]['index'], batch_images.shape)
        interpreter.allocate_tensors()  # Перевыделение памяти после изменения формы
        interpreter.set_tensor(input_details[0]['index'], batch_images)
        interpreter.invoke()
        emb = interpreter.get_tensor(output_details[0]['index'])
        embeddings.append(emb)
    return np.vstack(embeddings)


def main():
    csv_file = "dataset.csv"  # CSV с колонками "image_path" и "subject_id"
    model_path = "GhostFaceNet_W1.3_S2_ArcFace.h5"  # Путь к контрольной точке модели
    output_file = "embeddings_ghost_quantized.npz"

    # Конфигурация CPU
    print("Using CPU for inference.")

    image_paths, subject_ids = load_dataset(csv_file)

    print("Loading original GhostFaceNet model...")
    model = load_ghostfacenet_model(model_path, input_shape=(112,112,3))

    print("Quantizing GhostFaceNet model...")
    quantized_model = quantize_model(model)

    print("Extracting embeddings using quantized GhostFaceNet...")
    emb = extract_embeddings_tflite(quantized_model, image_paths, batch_size=32)

    np.savez(output_file, embeddings=emb, labels=subject_ids)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()