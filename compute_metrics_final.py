import numpy as np
import faiss
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA


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


def flatten_embeddings(embeddings):
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    return embeddings


def l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)


def compute_pca(embeddings, n_components=2):
    """
    Сводим эмбеддинги к 2D с помощью PCA.
    """
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(embeddings)
    return X_pca


def compute_fold_size(n_sub):
    """
    Размер фолда = ceil(n_sub^(2/3)).
    Например, для n_sub = 1000: fold_size = ceil(1000^(2/3)) = 100.
    """
    return math.ceil(n_sub ** (2 / 3))


def compute_n_splits(n_sub, fold_size):
    """
    Количество фолдов = ceil(n_sub / fold_size).
    """
    return math.ceil(n_sub / fold_size)


def compute_k(n_subset):
    """
    k = ceil(n_subset^(2/3)); если меньше 2, то 2.
    """
    k_val = math.ceil(n_subset ** (2 / 3))
    return k_val if k_val >= 2 else 2


def batched_nn_accumulation(embeddings, labels, batch_size, k):
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


def cross_validation_metrics(embeddings, labels, batch_size):
    """
    Делит данные подвыборки на фолды, где:
      - fold_size = ceil(n_sub^(2/3))
      - n_splits = ceil(n_sub / fold_size)
      - Для каждого фолда, если k не задан, k = fold_size (минимум 2)
    Выводит в консоль: n_sub, fold_size, n_splits и для каждого фолда размер и k_fold.
    Возвращает усреднённые метрики (precision, recall, f1, accuracy).
    """
    n_sub = embeddings.shape[0]
    fold_size = compute_fold_size(n_sub)
    n_splits = compute_n_splits(n_sub, fold_size)
    print(f"Total samples in sub-dataset: {n_sub}")
    print(f"Computed fold size: {fold_size}")
    print(f"Number of folds (n_splits): {n_splits}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []
    fold_counter = 1
    for _, test_idx in kf.split(embeddings):
        fold_embeddings = embeddings[test_idx]
        fold_labels = np.array(labels)[test_idx]
        k_fold = compute_k(len(fold_embeddings))
        print(f"Fold {fold_counter}: {len(fold_embeddings)} samples, computed k = {k_fold}")
        fold_counter += 1
        sim_scores, gt = batched_nn_accumulation(fold_embeddings, fold_labels, batch_size, k_fold)
        best_thresh, _ = find_optimal_threshold_sorted(sim_scores, gt)
        precision, recall, f1, accuracy = compute_metrics_sorted(sim_scores, gt, best_thresh)
        metrics_list.append((precision, recall, f1, accuracy))
    metrics_array = np.array(metrics_list)
    avg_metrics = metrics_array.mean(axis=0)
    print(
        f"Average metrics over folds: Precision={avg_metrics[0]:.4f}, Recall={avg_metrics[1]:.4f}, F1={avg_metrics[2]:.4f}, Accuracy={avg_metrics[3]:.4f}")
    return avg_metrics


def process_model(npz_path, model_name, start_size, num_points, batch_size):
    """
    Для заданной модели:
      1. Загружаются эмбеддинги, приводятся к 2D (PCA), нормализуются.
      2. Для ряда подвыборок (от start_size до общего числа объектов, равномерно по num_points) вычисляются усреднённые метрики (через кросс-валидацию).
    Выводит в консоль вычисленные значения k для подвыборки.
    Возвращает массив размеров подвыборок, а также значения recall, precision и accuracy.
    """
    print(f"\nProcessing model {model_name} from file: {npz_path}")
    embeddings, labels = load_embeddings(npz_path)
    embeddings = flatten_embeddings(embeddings)
    # Сводим эмбеддинги к 2D
    embeddings = compute_pca(embeddings, n_components=2)
    embeddings = l2_normalize(embeddings)
    total_samples = embeddings.shape[0]
    if start_size > total_samples:
        print(f"Start size ({start_size}) > total samples ({total_samples}). Using total samples.")
        start_size = total_samples
    sample_sizes = np.linspace(start_size, total_samples, num_points, dtype=int)
    sample_sizes = np.unique(sample_sizes)
    recall_values = []
    precision_values = []
    accuracy_values = []
    for size in sample_sizes:
        print(f"\nModel {model_name}: Evaluating sub-sample of {size} subjects")
        indices = np.random.choice(total_samples, size, replace=False)
        subset_embeddings = embeddings[indices]
        subset_labels = np.array(labels)[indices]
        k_subset = compute_k(len(subset_embeddings))
        print(f"For sub-sample of size {size}, computed k = {k_subset}")
        prec, rec, f1, acc = cross_validation_metrics(subset_embeddings, subset_labels, batch_size)
        recall_values.append(rec)
        precision_values.append(prec)
        accuracy_values.append(acc)
        print(f"Sub-sample metrics: Precision: {prec * 100:.2f}%, Recall: {rec * 100:.2f}%, Accuracy: {acc * 100:.2f}%")
    return sample_sizes, recall_values, precision_values, accuracy_values


def main():
    models = [
        {"npz_path": "embeddings_facenet_lfw.npz", "model_name": "FaceNet LFW dataset", "start_size": 1000},
        {"npz_path": "embeddings_arcface_lfw.npz", "model_name": "ArcFace LFW dataset", "start_size": 10000},
        {"npz_path": "embeddings_groupface_lfw.npz", "model_name": "GroupFace LFW dataset", "start_size": 1000},
        {"npz_path": "embeddings_ghost_lfw.npz", "model_name": "GhostFace LFW dataset", "start_size": 10000},
        {"npz_path": "embeddings_adaface_lfw.npz", "model_name": "AdaFace LFW dataset", "start_size": 10000}
    ]
    num_points = 5
    batch_size = 256
    model_sample_sizes = {}
    model_recall = {}
    model_precision = {}
    model_accuracy = {}
    for model in models:
        sample_sizes, recall_vals, precision_vals, accuracy_vals = process_model(
            model["npz_path"],
            model["model_name"],
            model["start_size"],
            num_points,
            batch_size
        )
        model_sample_sizes[model["model_name"]] = sample_sizes
        model_recall[model["model_name"]] = recall_vals
        model_precision[model["model_name"]] = precision_vals
        model_accuracy[model["model_name"]] = accuracy_vals
    # График Recall
    plt.figure(figsize=(10, 6))
    for model_name in model_recall:
        plt.plot(model_sample_sizes[model_name], model_recall[model_name], marker='o', label=model_name)
    plt.xlabel('Number of Subjects')
    plt.ylabel('Recall')
    plt.title('Recall vs Number of Subjects for 5 Models')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
    # График Precision
    plt.figure(figsize=(10, 6))
    for model_name in model_precision:
        plt.plot(model_sample_sizes[model_name], model_precision[model_name], marker='s', label=model_name)
    plt.xlabel('Number of Subjects')
    plt.ylabel('Precision')
    plt.title('Precision vs Number of Subjects for 5 Models')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
    # График Accuracy
    plt.figure(figsize=(10, 6))
    for model_name in model_accuracy:
        plt.plot(model_sample_sizes[model_name], model_accuracy[model_name], marker='^', label=model_name)
    plt.xlabel('Number of Subjects')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Subjects for 5 Models')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
