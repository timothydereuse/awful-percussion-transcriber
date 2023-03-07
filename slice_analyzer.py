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

import torch
import triplet_loss_net as tln

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

sound_fpath = r"C:\Users\Tim\Music\MusicBee\Music\L\Liliane Chlela\Safala\1-04 Moukassarat.mp3"
length_limit_secs = 300      # if source longer than this number of seconds, discard anything past this point
n_clusters = 15
controls_per_note = 1
target_short_name = sound_fpath.split('\\')[-1][:-4]
out_fname = f'slicer{target_short_name}-{timestamp}.mid'   # output filename with timestamp
poll_every = 50             # controls how often console output is produced when calculating features

embedding_model_path = "saved_model.pt"

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
            print(rf'calculating features for slice {i}/{len(segmented)}')
        slices.append(ausl.AudioSlice(s, sr, fname))

    tempo, beats = rosa.beat.beat_track(y=y, sr=sr)

    from scipy.interpolate import interp1d
    beat_frames = rosa.frames_to_samples(beats)
    beat_frames = np.concatenate([[0], beat_frames])
    interp_func = interp1d(beat_frames, np.arange(len(beat_frames)), kind='linear', fill_value='extrapolate')
    onset_times_beats = interp_func(onset_times)

    return slices, onset_times, tempo, onset_times_beats


sound, sr = get_soundfile(sound_fpath)
slices, onset_times, tempo, onset_times_beats = slice_long_sample(sound, sr, length_limit=length_limit_secs)
X = np.array([s.feats for s in slices])

embedding_model = (torch.load(embedding_model_path))
model = tln.EmbeddingNetwork(*embedding_model['model_args'])
model.eval()
embedding = model(torch.Tensor(X).float()).detach().numpy()

# optics = clust.OPTICS()
# clustering = optics.fit(embedding)
# labels = clustering.labels_

aggc = clust.AgglomerativeClustering(n_clusters=None, distance_threshold=2000, compute_distances=True)
# aggc = clust.AgglomerativeClustering(n_clusters=n_clusters)
clustering = aggc.fit_predict(embedding)
labels = clustering

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(embedding)
import matplotlib.pyplot as plt 
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=50, alpha=0.8, c=labels)
plt.show()

control_changes = np.zeros([len(labels), controls_per_note])
for l in set(labels):
    cluster_locs = np.where(labels == l)
    just_this_cluster = embedding[cluster_locs]
    pca = decomp.PCA(n_components=controls_per_note)
    ccs = pca.fit_transform(just_this_cluster)
    ccs = ccs - np.min(ccs, 0)
    ccs = (127 * ccs / np.max(ccs, 0)).round()
    control_changes[cluster_locs] = ccs
control_changes = control_changes.astype(np.uint8)

from mido import MetaMessage, Message, MidiFile, MidiTrack
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
track.append(MetaMessage('set_tempo', tempo=int(mido.tempo2bpm(tempo))))

def add_note(tr, note, vel, dur, ccs):
    base_note = 32

    multiplier = len(ccs)
    for i, cc in enumerate(ccs):
        cc_num = (np.abs(note) * multiplier) + 1 + i
        tr.append(Message('control_change', control=cc_num, value=cc, time=0))

    tr.append(Message('note_on', note=note + base_note, channel=10, velocity=vel, time=0))
    tr.append(Message('note_off', note=note + base_note, channel=10, velocity=vel, time=dur))
    return tr

max_rms  = max([s.rms for s in slices])
onset_times_beats = np.concatenate([onset_times_beats, [onset_times_beats[-1] + sr]])
for i, label in enumerate(labels):
    ccs = control_changes[i]

    stime = int(mid.ticks_per_beat * onset_times_beats[i])
    etime = int(mid.ticks_per_beat * onset_times_beats[i + 1])

    vel = int(slices[i].rms / max_rms * 127)
    track = add_note(track, label, vel, etime - stime, ccs)

# track.append(Message('program_change', program=12, time=0)
mid.save(out_fname)
