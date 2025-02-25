import numpy as np
import faiss
from collections import Counter


def load_embeddings(npz_path):
    """
    Загружает эмбеддинги и метки из NPZ-файла.
    Ожидается, что файл содержит ключи "embeddings" и "labels".
    """
    data = np.load(npz_path)
    embeddings = data['embeddings']
    labels = data['labels']
    return embeddings, labels


def compute_metrics_faiss(embeddings, labels, threshold):
    """
    Вычисляет precision, recall и F1-score на основе евклидовых расстояний между эмбеддингами с использованием FAISS.

    Для каждой пары (i, j) с i < j:
      - Если labels[i] == labels[j], то ground truth = 1 (пара положительная).
      - Если расстояние между эмбеддингами меньше threshold, пара считается предсказанной как положительная.

    FAISS использует L2^2, поэтому для range search используется radius = threshold**2.

    Возвращает: precision, recall, f1.
    """
    n, d = embeddings.shape
    # Убедимся, что embeddings имеют тип float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # Создаем индекс по евклидову расстоянию (L2)
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # FAISS возвращает квадрат евклидова расстояния, поэтому порог возводим в квадрат
    radius = threshold ** 2
    res = faiss.RangeSearchResult(n)
    index.range_search(embeddings, radius, res)

    # Собираем все предсказанные пары (только для i < j, чтобы не дублировать)
    predicted_pairs = []
    for i in range(n):
        start = res.lims[i]
        end = res.lims[i + 1]
        for idx in range(start, end):
            j = int(res.labels[idx])
            # Исключаем самосовпадения и дублируем пары: берем только j > i
            if j <= i:
                continue
            predicted_pairs.append((i, j))

    # Подсчитываем TP и FP по предсказанным парам
    TP = 0
    FP = 0
    for i, j in predicted_pairs:
        if labels[i] == labels[j]:
            TP += 1
        else:
            FP += 1
    total_predicted = len(predicted_pairs)

    # Подсчет общего количества "ground truth" положительных пар
    # Для каждого уникального субъекта количество положительных пар = n*(n-1)/2
    label_counts = Counter(labels)
    total_gt = sum(count * (count - 1) // 2 for count in label_counts.values())

    FN = total_gt - TP  # пары, которые должны были быть распознаны, но не найдены

    precision = TP / total_predicted if total_predicted > 0 else 0
    recall = TP / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def main():
    npz_path = "embeddings_original.npz"  # Путь к файлу с эмбеддингами
    embeddings, labels = load_embeddings(npz_path)

    # Задайте порог для евклидова расстояния (в единицах исходного пространства)
    threshold = 0.8  # Подберите оптимальное значение под ваш датасет
    precision, recall, f1 = compute_metrics_faiss(embeddings, labels, threshold)

    print("Метрики распознавания через FAISS:")
    print(f"Порог (threshold): {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


if __name__ == '__main__':
    main()
