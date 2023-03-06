import librosa as rosa
import numpy as np


def extract_features_for_embedding(inp, sr, target_sr=22050, max_length=1, hop=512):

    inp = rosa.resample(inp, sr, target_sr)

    crop_inp = inp[:target_sr * max_length]

    # extend with 0s if we are trying to get more feature frames than this is long
    if len(crop_inp) < (target_sr * max_length):
        extend = (target_sr * max_length) - len(crop_inp)
        crop_inp = np.concatenate([crop_inp, np.zeros(extend)])

    # normalize sound before feature extraction
    if max(crop_inp) > 0:
        crop_inp = np.clip(crop_inp / max(crop_inp), 0, 1)

    # spectrogram = rosa.pseudo_cqt(crop_inp, target_sr, hop_length=hop, n_bins=42)
    n_fft = 1024

    spec_mag = np.abs(rosa.stft(crop_inp, n_fft=n_fft, hop_length=hop))
    polys = rosa.feature.spectral.poly_features(sr=sr, S=spec_mag, hop_length=hop, order=2)

    # comment out lines to exclude items from the featureset
    feats = [
        rosa.feature.spectral_centroid(sr=sr,  S=spec_mag, n_fft=n_fft, hop_length=hop)[0],
        rosa.feature.spectral_contrast(sr=sr, S=spec_mag, n_fft=n_fft, hop_length=hop)[0],
        rosa.feature.spectral_bandwidth(sr=sr, S=spec_mag, n_fft=n_fft, hop_length=hop)[0],
        rosa.feature.spectral_flatness(S=spec_mag, n_fft=n_fft, hop_length=hop)[0],
        rosa.feature.spectral_rolloff(sr=sr, S=spec_mag, n_fft=n_fft, hop_length=hop)[0],
        # polys[1],
        # polys[2],
        rosa.feature.rms(S=spec_mag, frame_length=n_fft, hop_length=hop)[0]
    ]
    all_feats = np.concatenate(feats)

    return all_feats


def extract_features_from_audio(inp, sr, len_segments=2**9, internal_segments=32, out_segments=5, normalize=True):
    n_fft = len_segments * 2
    hop = len_segments // 4

    # normalize sound before feature extraction
    if max(inp) > 0 and normalize:
        inp = np.clip(inp / max(inp), 0, 1)

    # extend with 0s if we are trying to get more feature frames than this is long
    if len(inp) < len_segments * internal_segments:
        extend = (len_segments * internal_segments) - len(inp)
        inp = np.concatenate([inp, np.zeros(extend)])

    # spectrogram = np.abs(rosa.stft(inp, n_fft=n_fft, hop_length=hop))
    polys = rosa.feature.spectral.poly_features(inp, sr, hop_length=hop, order=2)

    # comment out lines to exclude items from the featureset
    feats = {
        'spec_centroid': rosa.feature.spectral_centroid(inp, sr, n_fft=n_fft, hop_length=hop)[0],
        'spec_contrast': rosa.feature.spectral_contrast(inp, sr)[0],
        'spec_bandwidth': rosa.feature.spectral_bandwidth(inp, sr, n_fft=n_fft, hop_length=hop)[0],
        'spec_flatness': rosa.feature.spectral_flatness(inp, n_fft=n_fft, hop_length=hop)[0],
         # 'spec_rolloff': rosa.feature.spectral_rolloff(inp, sr, n_fft=n_fft, hop_length=hop),
        'poly_features_1': polys[1],
        'poly_features_2': polys[2],
        'rms': rosa.feature.rms(inp, frame_length=n_fft, hop_length=hop)[0],
    }

    interp_points = (np.linspace(0, 1, out_segments) ** 1.75) * internal_segments
    interp_feats = {
        f: np.interp(interp_points, np.arange(0, len(feats[f])), feats[f])
        for f in feats.keys()
    }

    flat_feats = {}
    for key in feats.keys():
        feat_seq = feats[key].ravel()

        for i in range(out_segments):
            try:
                flat_feats[f'{key}_{i}'] = feat_seq[i]
            except IndexError:
                flat_feats[f'{key}_{i}'] = 0

    return flat_feats