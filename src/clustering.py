from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd


def run_kmeans(features, n_clusters=2, random_state=42):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    cluster_labels = model.fit_predict(features)
    return cluster_labels, model


def run_pca(features, n_components=8, random_state=42):
    pca = PCA(
        n_components=n_components,
        random_state=random_state
    )
    reduced_features = pca.fit_transform(features)
    return reduced_features, pca


def run_pca_kmeans(features, n_components=8, n_clusters=2, random_state=42):
    reduced_features, pca_model = run_pca(
        features=features,
        n_components=n_components,
        random_state=random_state
    )

    cluster_labels, kmeans_model = run_kmeans(
        features=reduced_features,
        n_clusters=n_clusters,
        random_state=random_state
    )

    return reduced_features, cluster_labels, pca_model, kmeans_model


def build_easy_comparison_table(vae_sil, vae_ch, pca_sil, pca_ch):
    comparison_df = pd.DataFrame({
        "method": ["VAE + KMeans", "PCA + KMeans"],
        "silhouette_score": [vae_sil, pca_sil],
        "calinski_harabasz_index": [vae_ch, pca_ch]
    })
    return comparison_df
