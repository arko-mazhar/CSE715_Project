
import pandas as pd

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

def load_hard_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return df
