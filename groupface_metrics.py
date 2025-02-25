import numpy as np
import faiss
from tqdm import tqdm
import time
from collections import Counter


def load_embeddings(npz_path):
    data = np.load(npz_path)
    embeddings = data['embeddings']
    labels = data['labels']
    return embeddings, labels


def l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)


def batched_pair_accumulation_gpu(embeddings, labels, batch_size=256, radius=4.0):
    n, d = embeddings.shape
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # Создаем GPU индекс и переносим его в CPU для range_search (так как GPU range_search не реализован)
    gpu_res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
    gpu_index.add(embeddings)
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)

    sim_list = []
    gt_list = []

    t0 = time.time()
    for i_start in tqdm(range(0, n, batch_size), desc="Обработка батчей (GPU->CPU)"):
        i_end = min(n, i_start + batch_size)
        queries = embeddings[i_start:i_end]
        res = cpu_index.range_search(queries, radius + 1e-6)
        # Если объект имеет атрибут lims, используем его, иначе считаем, что res – кортеж
        if hasattr(res, 'lims'):
            lims = res.lims
            sq_dists = res.distances
            neighbors = res.labels
        else:
            lims, sq_dists, neighbors = res
        for local_idx in range(i_end - i_start):
            global_idx = i_start + local_idx
            start_idx = lims[local_idx]
            end_idx = lims[local_idx + 1]
            for j_idx in range(start_idx, end_idx):
                j = int(neighbors[j_idx])
                if j <= global_idx:
                    continue
                dist = np.sqrt(sq_dists[j_idx])
                sim_list.append(dist)
                gt_list.append(1 if labels[global_idx] == labels[j] else 0)
    elapsed_pairs = time.time() - t0
    print(f"Накоплено {len(sim_list)} пар за {elapsed_pairs:.2f} секунд.")
    return np.array(sim_list), np.array(gt_list)


def find_optimal_threshold(sim_scores, gt, n_thresholds=2000):
    grid = np.linspace(0, 2, n_thresholds)
    preds = (sim_scores[None, :] < grid[:, None]).astype(int)
    acc = np.mean(preds == gt[None, :], axis=1)
    best_idx = np.argmax(acc)
    best_thresh = grid[best_idx]
    best_acc = acc[best_idx]
    return best_thresh, best_acc


def compute_metrics(sim_scores, gt, best_thresh):
    final_preds = (sim_scores < best_thresh).astype(int)
    TP = np.sum((final_preds == 1) & (gt == 1))
    FP = np.sum((final_preds == 1) & (gt == 0))
    total_predicted = np.sum(final_preds)
    total_gt = np.sum(gt)
    FN = total_gt - TP
    precision = TP / total_predicted if total_predicted > 0 else 0
    recall = TP / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def main():
    npz_path = "embeddings_groupface.npz"
    embeddings, labels = load_embeddings(npz_path)
    embeddings = l2_normalize(embeddings)

    print("Начало накопления пар эмбеддингов (с использованием FAISS GPU->CPU) ...")
    t0 = time.time()
    sim_scores, gt = batched_pair_accumulation_gpu(embeddings, labels, batch_size=256, radius=4.0)
    elapsed_pairs = time.time() - t0
    print(f"Накоплено {len(sim_scores)} пар за {elapsed_pairs:.2f} секунд.")

    t1 = time.time()
    best_thresh, best_acc = find_optimal_threshold(sim_scores, gt, n_thresholds=2000)
    precision, recall, f1 = compute_metrics(sim_scores, gt, best_thresh)
    elapsed_thresh = time.time() - t1

    total_time = time.time() - t0
    print("\n=== Результаты верификации эмбеддингов (GroupFace, GPU->CPU) ===")
    print(f"Оптимальный порог (L2): {best_thresh:.4f}")
    print(f"Accuracy: {best_acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")
    print(f"Время поиска порога: {elapsed_thresh:.2f} секунд")
    print(f"Общее время оценки: {total_time:.2f} секунд")


if __name__ == "__main__":
    main()
