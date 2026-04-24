
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def get_audio_feature_columns():
    return [
        "duration_ms",
        "danceability",
        "acousticness",
        "energy",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "mode",
        "key",
        "valence"
    ]


def clean_lyrics_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_hard_dataframe(df):
    df = df.copy()
    df = df.dropna(subset=["track_name", "lyrics", "playlist_genre", "language"]).copy()
    df["lyrics"] = df["lyrics"].apply(clean_lyrics_text)
    df = df[df["lyrics"].str.strip() != ""].copy()
    return df


def build_scaled_audio_features(df):
    audio_cols = get_audio_feature_columns()
    X_audio = df[audio_cols].copy()

    scaler = StandardScaler()
    X_audio_scaled = scaler.fit_transform(X_audio)

    return X_audio, X_audio_scaled, scaler


def build_lyrics_tfidf_features(df, max_features=3000):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features
    )
    X_lyrics_tfidf = vectorizer.fit_transform(df["lyrics"])
    return X_lyrics_tfidf, vectorizer


def reduce_lyrics_features(X_lyrics_tfidf, n_components=100, random_state=42):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_lyrics_reduced = svd.fit_transform(X_lyrics_tfidf)
    return X_lyrics_reduced, svd


def build_fused_features(X_audio_scaled, X_lyrics_reduced):
    return np.hstack([X_audio_scaled, X_lyrics_reduced])


def encode_labels(series):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(series)
    return encoded, encoder
