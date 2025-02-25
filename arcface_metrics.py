import numpy as np
import faiss
from tqdm import tqdm
import time
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def load_embeddings(npz_path):
    data = np.load(npz_path)
    embeddings = data['embeddings']
    labels = data['labels']
    return embeddings, labels


def l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)


def batched_pair_accumulation(embeddings, labels, batch_size=256, radius=4.0):
    n, d = embeddings.shape
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    sim_list = []
    gt_list = []
    for i_start in tqdm(range(0, n, batch_size), desc="Обработка батчей"):
        i_end = min(n, i_start + batch_size)
        queries = embeddings[i_start:i_end]
        res = index.range_search(queries, radius + 1e-6)
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
    for thresh in tqdm(grid, desc="Поиск оптимального порога"):
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
    print(f"final_preds shape: {final_preds.shape}")
    print(f"gt shape: {gt.shape}")

    TP = np.sum((final_preds == 1) & (gt == 1))
    FP = np.sum((final_preds == 1) & (gt == 0))
    FN = np.sum((final_preds == 0) & (gt == 1))
    TN = np.sum((final_preds == 0) & (gt == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    return precision, recall, f1, accuracy


def main():
    npz_path = "embeddings_original_1.npz"
    embeddings, labels = load_embeddings(npz_path)
    embeddings = l2_normalize(embeddings)

    print("Начало накопления пар эмбеддингов (батчами через FAISS)...")
    t0 = time.time()
    sim_scores, gt = batched_pair_accumulation(embeddings, labels, batch_size=256, radius=4.0)
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

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt, (sim_scores < best_thresh).astype(int))
    print("\nConfusion Matrix:")
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for Verification")
    plt.show()


if __name__ == "__main__":
    import seaborn as sns

    main()
