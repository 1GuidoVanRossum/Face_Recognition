from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Уменьшаем размерность эмбеддингов до 2D с помощью t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Визуализируем
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="viridis", alpha=0.5)
plt.colorbar()
plt.title("t-SNE визуализация эмбеддингов")
plt.show()