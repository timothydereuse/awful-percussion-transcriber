import librosa as rosa
import numpy as np
from scipy.interpolate import interp1d
import audio_slice as ausl


def downsample_cqt(spectrogram, out_segments, power=1.75):
    interp_points = (np.linspace(0, 1, out_segments) ** power) * (spectrogram.shape[1] - 1)
    interp_func = interp1d(np.arange(spectrogram.shape[1]), spectrogram, axis=1, kind='quadratic')
    downsampled = interp_func(interp_points)
    return downsampled


def extract_cqt_for_slices(inp, sr, target_sr=22050, length_limit=None, max_length=1, hop=256, bins=64, out_segments=32, ret_slice_objects=True, given_onsets=None, track_beats=True):

    if sr != target_sr:
        inp = rosa.resample(inp, sr, target_sr)

    if length_limit and (len(inp) / target_sr) > length_limit:
        left = len(inp) // 2 - (length_limit * target_sr // 2)
        right = len(inp) // 2 + (length_limit * target_sr // 2)
        inp = inp[left:right]

    if given_onsets is not None:
        onsets = given_onsets
    else:
        onsets = rosa.onset.onset_detect(y=inp, sr=target_sr, backtrack=True, hop_length=hop)
    onsets = np.concatenate([onsets, [rosa.samples_to_frames(len(inp), hop_length=hop)]])
    onset_times = rosa.frames_to_samples(onsets, hop_length=hop)

    # # remove onsets with deltas too small
    # to_remove = np.where(np.diff(onset_times) <= 2)[0] + 1
    # onset_times = np.delete(onset_times, to_remove)

    # normalize every slice individually before taking cqt
    for i in range(onset_times.shape[0] - 1):
        left = onset_times[i]
        right = onset_times[i + 1]
        inp[left:right] = np.clip(inp[left:right] / np.max(np.abs(inp[left:right])), -1, 1)

    # take cqt
    inp = np.nan_to_num(inp, nan=1e-10)
    if np.any(np.isnan(inp)):
        print('THERES NANS CCC')
        raise ValueError

    cqt = rosa.pseudo_cqt(inp, target_sr, hop_length=hop, n_bins=bins)
    cqt[cqt == 0] = 1e-10
    cqt = np.log(cqt)

    # slice up cqt based on onsets
    max_length_frames = rosa.samples_to_frames(target_sr * max_length, hop_length=hop)
    cqt_slices = np.zeros([len(onsets) - 1, bins, max_length_frames])
    for i in range(onsets.shape[0] - 1):
        chunk = cqt[:, onsets[i]:onsets[i + 1]]
        num_frames = min(max_length_frames, chunk.shape[1]) - 1
        cqt_slices[i][:, :num_frames] = chunk[:, :num_frames]


    # make AudioSlice objects with downsampled cqts
    segmented = [inp[onset_times[n]:onset_times[n + 1]] for n in range(len(onset_times) - 1)]
    slices = []
    for i, s in enumerate(segmented):
        down_feats = downsample_cqt(cqt_slices[i], out_segments, power=1.75)
        if ret_slice_objects:
            slice = ausl.AudioSlice(s, target_sr, precomputed_feats=down_feats)
        else:
            slice = down_feats
        slices.append(slice)

    # beat tracking and reinterpolating
    onset_times_beats = None
    tempo = None
    if track_beats:
        tempo, beats = rosa.beat.beat_track(y=inp, sr=target_sr, hop_length=hop)
        try:
            beat_frames = rosa.frames_to_samples(beats, hop_length=hop)
            beat_frames = np.concatenate([[0], beat_frames])
            interp_func = interp1d(beat_frames, np.arange(len(beat_frames)), kind='linear', fill_value='extrapolate')
            onset_times_beats = interp_func(onset_times)
        except ValueError:
            print('beat finding failed')

    return slices, onset_times, tempo, onset_times_beats


def extract_cqt_for_embedding(inp, sr, target_sr=22050, max_length=1, hop=256, bins=64, out_segments=32, ret_spec=True):
    # aaa = extract_cqt_for_embedding(s, sr, ret_spec=True)
    # plt.imshow(aaa)
    # plt.show()

    inp = rosa.resample(inp, sr, target_sr)
    crop_inp = inp[:target_sr * max_length]

    # extend with 0s if we are trying to get more feature frames than this is long
    if len(crop_inp) < (target_sr * max_length):
        extend = (target_sr * max_length) - len(crop_inp)
        crop_inp = np.concatenate([crop_inp, np.zeros(extend)])

    # normalize sound before feature extraction
    if max(crop_inp) > 0:
        crop_inp = np.clip(crop_inp / max(crop_inp), -1, 1)
    
    spectrogram = rosa.pseudo_cqt(crop_inp, target_sr, hop_length=hop, n_bins=bins)
    spectrogram[spectrogram == 0] = 1e-10
    spectrogram = np.log(spectrogram)

    downsampled = downsample_cqt(spectrogram, out_segments, power=1.75)

    if np.any(np.isnan(downsampled)):
        print('THERES NANS CCC')
        raise ValueError

    return downsampled if ret_spec else downsampled.ravel()


def extract_features_for_embedding(inp, sr, target_sr=22050, max_length=1, hop=256):

    inp = rosa.resample(inp, sr, target_sr)

    crop_inp = inp[:target_sr * max_length]

    # extend with 0s if we are trying to get more feature frames than this is long
    if len(crop_inp) < (target_sr * max_length):
        extend = (target_sr * max_length) - len(crop_inp)
        crop_inp = np.concatenate([crop_inp, np.zeros(extend)])

    # normalize sound before feature extraction
    if max(crop_inp) > 0:
        crop_inp = np.clip(crop_inp / max(crop_inp), -1, 1)

    # spectrogram = rosa.pseudo_cqt(crop_inp, target_sr, hop_length=hop, n_bins=42)
    n_fft = 512

    spec_mag = np.abs(rosa.stft(crop_inp, n_fft=n_fft, hop_length=hop))
    polys = rosa.feature.spectral.poly_features(sr=sr, S=spec_mag, hop_length=hop, order=2)

    # comment out lines to exclude items from the featureset
    feats = [
        rosa.feature.spectral_centroid(sr=sr,  S=spec_mag, n_fft=n_fft, hop_length=hop)[0],
        rosa.feature.spectral_contrast(sr=sr, S=spec_mag, n_fft=n_fft, hop_length=hop)[0],
        rosa.feature.spectral_bandwidth(sr=sr, S=spec_mag, n_fft=n_fft, hop_length=hop)[0],
        rosa.feature.spectral_flatness(S=spec_mag, n_fft=n_fft, hop_length=hop)[0],
        rosa.feature.spectral_rolloff(sr=sr, S=spec_mag, n_fft=n_fft, hop_length=hop)[0],
        polys[1],
        polys[2],
        rosa.feature.rms(S=spec_mag, frame_length=n_fft, hop_length=hop)[0]
    ]
    all_feats = np.concatenate(feats).astype(float)

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