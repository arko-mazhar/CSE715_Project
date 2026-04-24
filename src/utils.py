import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def ensure_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def run_tsne(features, random_state=42, perplexity=30):
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca"
    )
    tsne_features = tsne.fit_transform(features)
    return tsne_features


def run_umap(features, random_state=42, n_neighbors=15, min_dist=0.1):
    import umap.umap_ as umap

    umap_model = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )
    umap_features = umap_model.fit_transform(features)
    return umap_features


def plot_scatter_by_label(x, y, labels, title, xlabel, ylabel, save_path=None):
    plt.figure(figsize=(8, 6))

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        x_subset = [x[i] for i in range(len(labels)) if labels[i] == label]
        y_subset = [y[i] for i in range(len(labels)) if labels[i] == label]
        plt.scatter(x_subset, y_subset, alpha=0.7, label=str(label))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
