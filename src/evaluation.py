from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd


def compute_silhouette_score(features, cluster_labels):
    score = silhouette_score(features, cluster_labels)
    return score


def compute_calinski_harabasz_score(features, cluster_labels):
    score = calinski_harabasz_score(features, cluster_labels)
    return score


def evaluate_easy_clustering(features, cluster_labels):
    sil_score = compute_silhouette_score(features, cluster_labels)
    ch_score = compute_calinski_harabasz_score(features, cluster_labels)

    return {
        "silhouette_score": sil_score,
        "calinski_harabasz_index": ch_score
    }


def build_cluster_language_table(df, language_col, cluster_col):
    table = pd.crosstab(df[language_col], df[cluster_col])
    return table
