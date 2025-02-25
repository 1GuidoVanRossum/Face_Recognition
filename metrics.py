import argparse
import numpy as np
import faiss
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def load_embeddings(npz_path, allow_pickle=True):
    data = np.load(npz_path, allow_pickle=allow_pickle)
    embeddings = data['embeddings']
    if 'labels' in data:
        labels = data['labels']
    elif 'subject_ids' in data:
        labels = data['subject_ids']
    else:
        raise KeyError("Ни 'labels', ни 'subject_ids' не найдены в архиве.")
    return embeddings, labels

def flatten_embeddings(embeddings):
    # Если эмбеддинги имеют больше двух измерений, приводим к форме (N, -1)
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    return embeddings

def l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)


def batched_nn_accumulation(embeddings, labels, batch_size=256, k=21):
    """
    Для каждого эмбеддинга ищем k ближайших соседей (включая себя) с помощью FAISS.
    Собираем пары (i, j) только если j > i, чтобы избежать повторений.
    Это существенно уменьшает число пар по сравнению с полным перебором.
    """
    n, d = embeddings.shape
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    sim_list = []
    gt_list = []

    # Обрабатываем батчи по эмбеддингам для уменьшения потребления памяти
    for i in tqdm(range(0, n, batch_size), desc="Обработка батчей"):
        i_end = min(n, i + batch_size)
        queries = embeddings[i:i_end]
        # Поиск ближайших k соседей для батча
        distances, indices = index.search(queries, k)
        for local_idx in range(i_end - i):
            global_idx = i + local_idx
            # Для каждого запроса рассматриваем найденных соседей (исключая самого себя, обычно в позиции 0)
            for j_idx in range(1, k):
                neighbor = int(indices[local_idx, j_idx])
                # Чтобы не считать пару дважды
                if neighbor <= global_idx:
                    continue
                dist = np.sqrt(distances[local_idx, j_idx])
                sim_list.append(dist)
                gt_list.append(1 if labels[global_idx] == labels[neighbor] else 0)
    return np.array(sim_list), np.array(gt_list)


def find_optimal_threshold_sorted(sim_scores, gt, n_thresholds=2000):
    N = sim_scores.shape[0]
    sorted_idx = np.argsort(sim_scores)
    sorted_sim = sim_scores[sorted_idx]
    sorted_gt = gt[sorted_idx]
    cum_tp = np.cumsum(sorted_gt)
    total_pairs = N
    total_neg = total_pairs - cum_tp[-1]

    grid = np.linspace(0, 2, n_thresholds)
    best_acc = -1
    best_thresh = None
    for thresh in grid:
        k = np.searchsorted(sorted_sim, thresh, side='right')
        TP = cum_tp[k - 1] if k > 0 else 0
        FP = k - TP
        TN = total_neg - FP
        acc = (TP + TN) / total_pairs
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def compute_metrics_sorted(sim_scores, gt, thresh):
    final_preds = (sim_scores < thresh).astype(int)
    # Вычисляем метрики с использованием sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(gt, final_preds, average='binary')
    accuracy = accuracy_score(gt, final_preds)
    return precision, recall, f1, accuracy


def main(npz_path):
    embeddings, labels = load_embeddings(npz_path)
    embeddings = flatten_embeddings(embeddings)
    embeddings = l2_normalize(embeddings)

    print("Начало накопления пар эмбеддингов (через поиск ближайших соседей)...")
    t0 = time.time()
    # Используем поиск ближайших соседей для каждого в батчах
    sim_scores, gt = batched_nn_accumulation(embeddings, labels, batch_size=256, k=21)
    elapsed_pairs = time.time() - t0
    print(f"Накоплено {len(sim_scores)} пар за {elapsed_pairs:.2f} секунд.")

    print("Поиск оптимального порога через сортировку...")
    t1 = time.time()
    best_thresh, best_acc = find_optimal_threshold_sorted(sim_scores, gt, n_thresholds=2000)
    elapsed_thresh = time.time() - t1
    print(
        f"Оптимальный порог: {best_thresh:.4f}, Accuracy: {best_acc * 100:.2f}% (найдено за {elapsed_thresh:.2f} секунд)")

    precision, recall, f1, accuracy = compute_metrics_sorted(sim_scores, gt, best_thresh)
    print("\n=== Итоговые метрики ===")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")
    print(f"Accuracy:  {accuracy * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Быстрый расчёт метрик верификации по эмбеддингам")
    parser.add_argument("--npz_path", type=str, required=True, help="Путь к файлу .npz с эмбеддингами")
    args = parser.parse_args()
    main(args.npz_path)
