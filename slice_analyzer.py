import librosa as rosa
import numpy as np
import soundfile
import os
import datetime
from pathlib import Path
from pydub.exceptions import CouldntDecodeError
from pydub import AudioSegment
from pydub.utils import get_array_type
import array
import feature_extraction as fe
import audio_slice as ausl
import sklearn.cluster as clust
import sklearn.decomposition as decomp
import mido

ftypes = ['wav', 'aiff', 'aif', 'mp3', 'flac']     # only bother with files of these types

# path to target file - this will be sliced, and sound slices from the sources will be matched to it

match_volume = True         # attempt to match the energy between each source slice and the slice its replacing
pca_reduce_amt = 10         # strength of dimensionality reduction on extracted features. higher = messier
                            # categorization by the knn but more information included
slice_threshold_secs = 6    # if a source is longer than this number of seconds, then slice it up before
                            # adding it to the pool of source audio clips
length_limit_secs = 200000  # if a source is longer than this number of seconds, then discard anything
                            # past this point so you don't accidentally slice up a 20 min file
num_clusters = 5

sound_fpath = r"C:\Users\Tim\Documents\MUSIC DATA\noize boyz\valek\pv_noise_groove box.wav"
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
target_short_name = sound_fpath.split('\\')[-1][:-4]
out_fname = f'slicer{target_short_name}-{timestamp}.mid'   # output filename with timestamp
poll_every = 50             # controls how often console output is produced when calculating features


def get_soundfile(fname):
    s = AudioSegment.from_file(file=fname)
    s = s.set_channels(1)
    sr = s.frame_rate
    bit_depth = s.sample_width * 8
    array_type = get_array_type(bit_depth)
    s = np.array(array.array(array_type, s._data))
    s = s / (2 ** (bit_depth - 1))
    return s, sr


def slice_long_sample(y, sr, declick_samples=4, length_limit=None, fname=''):

    if length_limit and (len(y) / sr) > length_limit:
        y = y[len(y) - (length_limit * sr // 2):len(y) + (length_limit * sr // 2)]

    onsets = rosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = rosa.frames_to_samples(onsets)
    onset_times = np.concatenate([onset_times, [len(y)]])
    segmented = [y[onset_times[n]:onset_times[n + 1]] for n in range(len(onset_times) - 1)]
    segmented = [s for s in segmented if len(s) >= declick_samples]

    if declick_samples > 1:
        declick_envelope = np.linspace(1 / declick_samples, 1 - (1 / declick_samples), declick_samples)
        for i in range(len(segmented)):
            segmented[i][0:declick_samples] *= declick_envelope

    slices = []
    for i, s in enumerate(segmented):
        if not i % poll_every and i > 1:
            print(rf'calculating features for slice {i}/{len(segmented)} of {fname}...')
        slices.append(ausl.AudioSlice(s, sr, fname))

    return slices, onset_times

sound, sr = get_soundfile(sound_fpath)
slices, onset_times = slice_long_sample(sound, sr)

keys_sorted = sorted(slices[0].feats.keys())
X = np.array([[s.feats[k] for k in keys_sorted] for s in slices])
# normalize by column
X_norm = (X - np.mean(X, 0)) / np.std(X, 0)
kpca = decomp.KernelPCA(pca_reduce_amt)
reduced = kpca.fit_transform(X_norm)

aggc = clust.AgglomerativeClustering(n_clusters=num_clusters)
clustering = aggc.fit_predict(reduced)

# optics = clust.DBSCAN()
# clustering = optics.fit(reduced)

# from sklearn.manifold import TSNE
# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(reduced)
# import matplotlib.pyplot as plt
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=50, alpha=0.8)

from mido import Message, MidiFile, MidiTrack
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
track.append(Message('program_change', program=12, time=0))

s_to_t = (mid.ticks_per_beat / sr) * (100 / 60)

def add_note(tr, note, vel, dur):
    tr.append(Message('note_on', note=note, velocity=vel, time=0))
    tr.append(Message('note_off', note=note, velocity=vel, time=dur))
    return tr

base_note = 32

onset_times = np.concatenate([onset_times, [onset_times[-1] + sr]])
for i, label in enumerate(clustering):
    stime = int(s_to_t * onset_times[i])
    etime = int(s_to_t * onset_times[i + 1])

    vel = int(slices[i].rms * 127)

    track = add_note(track, label + base_note, vel, etime - stime)

# track.append(Message('program_change', program=12, time=0)
mid.save(out_fname)
