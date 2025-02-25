import argparse
import numpy as np
import faiss
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def load_embeddings(npz_path, allow_pickle=True):
    data = np.load(npz_path, allow_pickle=allow_pickle)
    embeddings = data['embeddings']
    if 'labels' in data:
        labels = data['labels']
    elif 'subject_ids' in data:
        labels = data['subject_ids']
    else:
        raise KeyError("Neither 'labels' nor 'subject_ids' found in the file.")
    return embeddings, labels


def l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)

def flatten_embeddings(embeddings):
    # Если эмбеддинги имеют больше двух измерений, приводим к форме (N, -1)
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    return embeddings


def batched_nn_accumulation(embeddings, labels, batch_size=256, k=21):
    """
    For each embedding, search for k nearest neighbors (including itself)
    using FAISS. Collect pairs (i, j) only if j > i to avoid duplicates.
    """
    n, d = embeddings.shape
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    sim_list = []
    gt_list = []

    for i in tqdm(range(0, n, batch_size), desc="Processing batches"):
        i_end = min(n, i + batch_size)
        queries = embeddings[i:i_end]
        distances, indices = index.search(queries, k)
        for local_idx in range(i_end - i):
            global_idx = i + local_idx
            # Skip the first neighbor (itself) and consider the rest
            for j_idx in range(1, k):
                neighbor = int(indices[local_idx, j_idx])
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
        k_val = np.searchsorted(sorted_sim, thresh, side='right')
        TP = cum_tp[k_val - 1] if k_val > 0 else 0
        FP = k_val - TP
        TN = total_neg - FP
        acc = (TP + TN) / total_pairs
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def compute_metrics_sorted(sim_scores, gt, thresh):
    final_preds = (sim_scores < thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(gt, final_preds, average='binary')
    accuracy = accuracy_score(gt, final_preds)
    return precision, recall, f1, accuracy


def cross_validation_metrics(embeddings, labels, n_splits=5, batch_size=256, k=21):
    """
    Split data into n_splits folds and compute metrics for each fold.
    Return the average metrics over all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []
    for _, test_idx in kf.split(embeddings):
        fold_embeddings = embeddings[test_idx]
        fold_labels = np.array(labels)[test_idx]
        sim_scores, gt = batched_nn_accumulation(fold_embeddings, fold_labels, batch_size=batch_size, k=k)
        best_thresh, _ = find_optimal_threshold_sorted(sim_scores, gt)
        precision, recall, f1, accuracy = compute_metrics_sorted(sim_scores, gt, best_thresh)
        metrics_list.append((precision, recall, f1, accuracy))
    metrics_array = np.array(metrics_list)
    return metrics_array.mean(axis=0)


def main(npz_path, model_name):
    embeddings, labels = load_embeddings(npz_path)
    embeddings = flatten_embeddings(embeddings)
    embeddings = l2_normalize(embeddings)

    max_size = embeddings.shape[0]
    # Define candidate sample sizes: max, 10000, 1000, 100 (if available)
    candidate_sizes = [max_size, 10000, 1000, 100]
    sample_sizes = sorted([s for s in candidate_sizes if s <= max_size], reverse=True)

    results = {'sample_size': [], 'precision': [], 'recall': [], 'f1': []}

    print(f"Model: {model_name}")
    for size in sample_sizes:
        indices = np.random.choice(max_size, size, replace=False)
        subset_embeddings = embeddings[indices]
        subset_labels = np.array(labels)[indices]

        print(f"\nEvaluating sample of {size} subjects for model {model_name}:")
        prec, rec, f1, acc = cross_validation_metrics(subset_embeddings, subset_labels, n_splits=5, batch_size=256,
                                                      k=21)
        results['sample_size'].append(size)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1'].append(f1)
        print(f"Precision: {prec * 100:.2f}%")
        print(f"Recall:    {rec * 100:.2f}%")
        print(f"F1-score:  {f1 * 100:.2f}%")
        print(f"Accuracy:  {acc * 100:.2f}%")

    # Plotting the metrics
    plt.figure(figsize=(8, 6))
    plt.plot(results['sample_size'], results['precision'], marker='o', label=f'{model_name} Precision')
    plt.plot(results['sample_size'], results['recall'], marker='s', label=f'{model_name} Recall')
    plt.plot(results['sample_size'], results['f1'], marker='^', label=f'{model_name} F1-score')
    plt.xlabel('Number of Subjects')
    plt.ylabel('Metric Value')
    plt.title('Cross-Validation Metrics for Different Sample Sizes')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validation of embedding verification metrics with plotting")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the .npz file with embeddings")
    parser.add_argument("--model", type=str, required=True, help="Name of the model (e.g., FaceNet)")
    args = parser.parse_args()
    main(args.npz_path, args.model)
