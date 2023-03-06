import os
from pathlib import Path
from pydub.exceptions import CouldntDecodeError
from pydub import AudioSegment
from pydub.utils import get_array_type
import csv
import librosa
import array
import numpy as np
from feature_extraction import extract_features_for_embedding
import h5py

def get_soundfile(fname):
    s = AudioSegment.from_file(file=fname)
    s = s.set_channels(1)
    sr = s.frame_rate
    bit_depth = s.sample_width * 8
    array_type = get_array_type(bit_depth)
    s = np.array(array.array(array_type, s._data))
    s = s / (2 ** (bit_depth - 1))
    return s, sr

groups_file = "supervised_groups.txt"
processed_file = "./dataset_groups.h5"
max_per_group = 25

groups = []
with open(groups_file, newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        groups.append(row)

group_files = []
for pair in groups:
    fpath, mod = pair
    mod = mod.strip(' ')
    wavs = os.listdir(fpath)
    wavs = [os.path.join(fpath, x) for x in wavs if x[-4:] in ['.wav', '.WAV', '.aif']]
    if mod:
        wavs = [x for x in wavs if mod in x]
    wavs = wavs[:max_per_group]
    group_files.append(wavs)

for i, g in enumerate(group_files):
    feats = []
    group_name = g[0].split('\\')[-2]
    group_id = f'{i}{group_name}'

    for s in g:
        print(s)
        x, sr = get_soundfile(s)
        try:
            feat = extract_features_for_embedding(x, sr)
        except librosa.ParameterError:
            print(f'failed {s}, skipping')
            continue
        feats.append(feat)

    all_feats = np.stack(feats, 0)

    with h5py.File(processed_file, 'a') as f:
        f.create_dataset(group_id, data=all_feats)
    




