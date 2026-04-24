
from pathlib import Path
import pandas as pd

def build_gtzan_inventory(gtzan_root):
    root = Path(gtzan_root)
    wav_files = sorted(root.rglob("*.wav"))

    rows = []
    for wav_path in wav_files:
        genre = wav_path.parent.name
        track_id = wav_path.stem
        rows.append({
            "genre": genre,
            "track_id": track_id,
            "file_path": str(wav_path)
        })

    df = pd.DataFrame(rows)
    return df
