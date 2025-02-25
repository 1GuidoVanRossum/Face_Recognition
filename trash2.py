import numpy as np
import random
import faiss
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


def load_embeddings(npz_path, allow_pickle=True):
    data = np.load(npz_path, allow_pickle=allow_pickle)
    embeddings = data['embeddings']
    if 'labels' in data:
        labels = data['labels']
    elif 'subject_ids' in data:
        labels = data['subject_ids']
    else:
        raise KeyError("Ни 'labels', ни 'subject_ids' не найдено в файле.")
    return embeddings, labels


def group_indices_by_subject(labels):
    subject_to_indices = defaultdict(list)
    for idx, subj in enumerate(labels):
        subject_to_indices[subj].append(idx)
    return subject_to_indices


def run_single_experiment(embeddings, labels, subject_to_indices, n_subjects=100):
    subjects = list(subject_to_indices.keys())
    if len(subjects) < n_subjects:
        raise ValueError(f"Недостаточно субъектов для выборки {n_subjects} классов.")
    selected_subjects = random.sample(subjects, n_subjects)

    queries = []
    gallery = []
    for subj in selected_subjects:
        indices = subject_to_indices[subj]
        selected = random.sample(indices, 2)
        query_idx, gallery_idx = selected
        queries.append(embeddings[query_idx])
        gallery.append(embeddings[gallery_idx])

    queries = np.array(queries).astype('float32')
    gallery = np.array(gallery).astype('float32')
    n = queries.shape[0]

    d = gallery.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(gallery)
    distances, indices = index.search(queries, n)

    correct_count = 0
    reciprocal_ranks = []
    for i in range(n):
        correct_rank = int(np.where(indices[i] == i)[0][0]) + 1
        if correct_rank == 1:
            correct_count += 1
        reciprocal_ranks.append(1.0 / correct_rank)

    accuracy = correct_count / n
    mrr = np.mean(reciprocal_ranks)
    return accuracy, mrr


def run_experiments(embeddings, labels, n_subjects, n_experiments=6):
    subject_to_indices = group_indices_by_subject(labels)
    subject_to_indices = {subj: idxs for subj, idxs in subject_to_indices.items() if len(idxs) >= 2}
    accuracies = []
    mrrs = []
    for _ in tqdm(range(n_experiments), desc=f"Эксперименты для {n_subjects} классов", leave=False):
        acc, mrr = run_single_experiment(embeddings, labels, subject_to_indices, n_subjects)
        accuracies.append(acc)
        mrrs.append(mrr)
    return accuracies, mrrs


def run_models_and_plot(models, sample_sizes, n_experiments=6):
    results_acc = {}
    results_mrr = {}

    for model in models:
        npz_path = model["npz_path"]
        model_name = model["model_name"]
        print(f"\nОбработка модели {model_name} из файла {npz_path}")
        embeddings, labels = load_embeddings(npz_path)
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        subject_to_indices = group_indices_by_subject(labels)
        subject_to_indices = {subj: idxs for subj, idxs in subject_to_indices.items() if len(idxs) >= 2}
        available_subjects = list(subject_to_indices.keys())
        max_classes = len(available_subjects)

        model_sample_sizes = []
        avg_accuracies = []
        avg_mrrs = []

        for n_subjects in sample_sizes:
            if n_subjects > max_classes:
                print(f" Для выборки {n_subjects} классов: доступно только {max_classes}, пропускаем.")
                continue
            print(f" Запуск экспериментов для {n_subjects} классов")
            accuracies, mrrs = run_experiments(embeddings, labels, n_subjects, n_experiments=n_experiments)
            avg_acc = np.mean(accuracies)
            avg_mrr = np.mean(mrrs)
            model_sample_sizes.append(n_subjects)
            avg_accuracies.append(avg_acc)
            avg_mrrs.append(avg_mrr)
            print(f"  Среднее Accuracy: {avg_acc:.4f}, Среднее MRR: {avg_mrr:.4f}")

        results_acc[model_name] = (model_sample_sizes, avg_accuracies)
        results_mrr[model_name] = (model_sample_sizes, avg_mrrs)

    plt.figure(figsize=(10, 6))
    for model_name, (x, y) in results_acc.items():
        plt.plot(x, y, marker='o', label=model_name)
    plt.xlabel('Количество классов')
    plt.ylabel('Accuracy')
    plt.title('Accuracy в зависимости от количества классов')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for model_name, (x, y) in results_mrr.items():
        plt.plot(x, y, marker='o', label=model_name)
    plt.xlabel('Количество классов')
    plt.ylabel('MRR')
    plt.title('MRR в зависимости от количества классов')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    models = [
        {"npz_path": "embeddings_facenet.npz", "model_name": "FaceNet RFW dataset"},
        {"npz_path": "embeddings_arcface.npz", "model_name": "ArcFace RFW dataset"},
        {"npz_path": "embeddings_groupface.npz", "model_name": "GroupFace RFW dataset"},
        {"npz_path": "embeddings_ghost.npz", "model_name": "GhostFace RFW dataset"},
        {"npz_path": "embeddings_adaface.npz", "model_name": "AdaFace RFW dataset"}
    ]

    sample_sizes = [100, 1000, 5000, 10000]

    run_models_and_plot(models, sample_sizes, n_experiments=6)
