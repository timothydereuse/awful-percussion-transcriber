import librosa as rosa
import numpy as np
import soundfile
import os
import datetime
from pathlib import Path
from pydub.exceptions import CouldntDecodeError
from pydub import AudioSegment
from pydub.utils import get_array_type
import feature_extraction as fe
import audio_slice as ausl
import sklearn.cluster as clust
import sklearn.decomposition as decomp
from mido import MetaMessage, Message, MidiFile, MidiTrack, tempo2bpm
import torch
import triplet_loss_net as tln
import audio_management as aumg
import feature_extraction as fe

def visualize_clustering(embedding, labels):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt 
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=200).fit_transform(embedding)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=50, alpha=0.8, c=labels)
    plt.show()


def make_ccs(embedding, labels, controls_per_note):
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
    return control_changes

# sort clusters by average frequency
def cluster_sorting(labels, raw_feats):
    label_sort_idx = {}
    for l in set(labels):
        cluster_data = raw_feats[labels == l]
        cluster_sum = cluster_data.sum(0).sum(-1)
        cluster_sum = cluster_sum - np.min(cluster_sum)
        prob = np.sum((cluster_sum / np.sum(cluster_sum)) * np.arange(cluster_sum.shape[0]))
        label_sort_idx[l] = prob 
    label_ordering = sorted(list(set(labels)), key=lambda x: label_sort_idx[x])

    sorted_positions = [label_sort_idx[i] for i in sorted(label_sort_idx.keys())]
    return label_ordering, sorted_positions


def add_note(tr, note, vel, dur, ccs):
    base_note = 32

    multiplier = len(ccs)
    for i, cc in enumerate(ccs):
        cc_num = (np.abs(note) * multiplier) + 1 + i
        tr.append(Message('control_change', control=cc_num, value=cc, time=0))

    tr.append(Message('note_on', note=note + base_note, channel=10, velocity=vel, time=0))
    tr.append(Message('note_off', note=note + base_note, channel=10, velocity=vel, time=dur))
    return tr


def cluster_slices(sound_fpath, embedding_model_path, n_clusters, length_limit):

    # load and slice sound
    print('slicing audio and getting features of slices...')
    sound, sr = aumg.get_soundfile(sound_fpath)
    slices, onset_times, tempo, onset_times_beats = fe.extract_cqt_for_slices(sound, sr, length_limit=length_limit)
    # slices, onset_times, tempo, onset_times_beats = aumg.slice_long_sample(sound, sr, length_limit=length_limit)


    # set up model for embedding transformation
    embedding_model = (torch.load(embedding_model_path))
    model = tln.EmbeddingNetwork(*embedding_model['model_args'])
    model.eval()

    # transform slices into embedding dimension
    print('embedding audio features...')
    raw_feats = np.array([s.feats for s in slices])
    X = torch.Tensor(raw_feats).float().reshape(len(slices), -1)
    embedding = model(X).detach().numpy()

    # perform clustering
    print('clustering...')
    if n_clusters:
        aggc = clust.AgglomerativeClustering(n_clusters=n_clusters)
    else:
        aggc = clust.AgglomerativeClustering(n_clusters=None, distance_threshold=1, compute_distances=True)
    clustering = aggc.fit_predict(embedding)
    labels = clustering
    print(f'num clusters found: {len(set(labels))}')
    return (sr, labels, embedding, slices, raw_feats, tempo, onset_times_beats)

# set up midi file
def build_midi_file(sr, labels, embedding, slices, raw_feats, tempo, onset_times_beats):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=int(tempo2bpm(tempo))))
    control_changes = make_ccs(embedding, labels, controls_per_note)

    label_ordering, sorted_positions = cluster_sorting(labels, raw_feats)
    print(f'freq positions: {sorted_positions}')

    max_rms  = max([s.rms for s in slices])
    onset_times_beats = np.concatenate([onset_times_beats, [onset_times_beats[-1] + sr]])
    for i, label in enumerate(labels):
        ccs = control_changes[i]

        stime = int(mid.ticks_per_beat * onset_times_beats[i])
        etime = int(mid.ticks_per_beat * onset_times_beats[i + 1])

        transformed_label = label_ordering[label]

        vel = int(slices[i].rms / max_rms * 127)
        track = add_note(track, transformed_label, vel, etime - stime, ccs)

    return mid


if __name__ == '__main__':

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    sound_fpath = r"C:\Users\Tim\Documents\MUSIC DATA\Loops\VIBRIS CLASSIC BREAKS\101_classicbrk-32.aif"
    length_limit_secs = 1000      # if source longer than this number of seconds, discard anything past this point
    n_clusters = None
    controls_per_note = 1
    target_short_name = sound_fpath.split('\\')[-1][:-4]
    out_fname = f'./out_midi/slicer{target_short_name}-{timestamp}.mid'   # output filename with timestamp 
    embedding_model_path = "cqt_saved_model.pt"

    res = cluster_slices(sound_fpath, embedding_model_path, n_clusters, length_limit_secs)
    print('building midi file...')
    mid = build_midi_file(*res)

    mid.save(out_fname)
