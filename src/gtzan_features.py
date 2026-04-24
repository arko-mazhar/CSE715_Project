import numpy as np
import librosa


def load_audio_fixed(file_path, sr=22050, duration=30):
    target_length = sr * duration
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))

    return y


def pad_or_truncate_2d(array_2d, target_frames=128):
    current_frames = array_2d.shape[1]

    if current_frames > target_frames:
        return array_2d[:, :target_frames]
    elif current_frames < target_frames:
        pad_width = target_frames - current_frames
        return np.pad(array_2d, ((0, 0), (0, pad_width)))
    else:
        return array_2d


def minmax_normalize(array_2d):
    min_val = np.min(array_2d)
    max_val = np.max(array_2d)

    if max_val - min_val == 0:
        return np.zeros_like(array_2d)

    return (array_2d - min_val) / (max_val - min_val)


def extract_mel_spectrogram(
    file_path,
    sr=22050,
    duration=30,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    target_frames=128
):
    y = load_audio_fixed(file_path=file_path, sr=sr, duration=duration)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = pad_or_truncate_2d(mel_db, target_frames=target_frames)
    mel_norm = minmax_normalize(mel_db)

    return mel_norm.astype(np.float32)


def extract_mfcc_summary(
    file_path,
    sr=22050,
    duration=30,
    n_mfcc=20,
    n_fft=2048,
    hop_length=512
):
    y = load_audio_fixed(file_path=file_path, sr=sr, duration=duration)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    feature_vector = np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)
    return feature_vector
