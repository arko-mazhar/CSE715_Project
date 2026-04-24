import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_bengali_dataset(file_path):
    df = pd.read_excel(file_path)
    return df


def load_spotify_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


def select_common_columns(df_bengali, df_spotify):
    bengali_keep = [
        "track_name",
        "লিরিক্স",
        "playlist_genre",
        "playlist_subgenre",
        "track_artist",
        "duration_ms",
        "language",
        "track_album_release_date",
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

    spotify_keep = [
        "track_name",
        "lyrics",
        "playlist_genre",
        "playlist_subgenre",
        "track_artist",
        "duration_ms",
        "language",
        "track_album_release_date",
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

    df_bengali_small = df_bengali[bengali_keep].copy()
    df_spotify_small = df_spotify[spotify_keep].copy()

    df_bengali_small = df_bengali_small.rename(columns={"লিরিক্স": "lyrics"})

    return df_bengali_small, df_spotify_small


def build_balanced_easy_dataset(df_bengali_small, df_spotify_small, random_state=42):
    df_spotify_en = df_spotify_small[df_spotify_small["language"] == "en"].copy()

    df_bengali_small["language"] = "bn"
    df_spotify_en["language"] = "en"

    df_bengali_clean = df_bengali_small.dropna(subset=["track_name", "lyrics"]).copy()
    df_spotify_en_clean = df_spotify_en.dropna(subset=["track_name", "lyrics"]).copy()

    df_bengali_clean = df_bengali_clean[
        df_bengali_clean["lyrics"].astype(str).str.strip() != ""
    ].copy()

    df_spotify_en_clean = df_spotify_en_clean[
        df_spotify_en_clean["lyrics"].astype(str).str.strip() != ""
    ].copy()

    sample_size = len(df_bengali_clean)
    df_spotify_sample = df_spotify_en_clean.sample(
        n=sample_size,
        random_state=random_state
    ).copy()

    df_easy = pd.concat([df_bengali_clean, df_spotify_sample], ignore_index=True)
    return df_easy


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


def prepare_easy_features(df_easy):
    audio_feature_cols = get_audio_feature_columns()

    X = df_easy[audio_feature_cols].copy()
    y_language = df_easy["language"].copy()
    y_genre = df_easy["playlist_genre"].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y_language, y_genre, scaler
