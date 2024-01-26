import pandas as pd
import librosa
import numpy as np


def extract_mfccs(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = mfccs.T
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mfccs = delta_mfccs

    # (216) rows = frames ,  and (26) columns = features
    features = np.hstack([mfccs, delta_mfccs])
    feature_names = [f"MFCC_{i + 1}" for i in range(features.shape[1])]
    return pd.DataFrame(features, columns=feature_names)


def extract_pitch(file_path):
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.85])
    return pd.DataFrame({"Pitch": [pitch]})


def extract_chroma(file_path):
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = chroma.T
    feature_names = [f"Chroma_{i + 1}" for i in range(chroma.shape[1])]
    return pd.DataFrame(chroma, columns=feature_names)


def extract_zero_crossings(file_path):
    y, sr = librosa.load(file_path)
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    zero_crossings = zero_crossings.T
    feature_names = [f"ZeroCrossings_{i + 1}" for i in range(zero_crossings.shape[1])]
    return pd.DataFrame(zero_crossings, columns=feature_names)


def extract_spectral_contrast(file_path):
    y, sr = librosa.load(file_path)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast = spectral_contrast.T
    feature_names = [
        f"SpectralContrast_{i + 1}" for i in range(spectral_contrast.shape[1])
    ]
    return pd.DataFrame(spectral_contrast, columns=feature_names)


# extract all the features intoon df for input
def extract_input_features(input_path):
    y, sr = librosa.load(input_path)
    # Extract various audio features using librosa functions
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    spectrogram = np.abs(librosa.stft(y))

    # Concatenate the mean values of different features along axis=1
    all_features = np.concatenate([
        np.mean(chroma, axis=1),
        np.mean(spectrogram, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(zero_crossing_rate, axis=1),
        np.mean(mfccs, axis=1),
        # np.mean(spectral_centroid, axis=1),
        # np.mean(spectral_bandwidth, axis=1),
    ])
    print("all features:", all_features)
    return all_features
