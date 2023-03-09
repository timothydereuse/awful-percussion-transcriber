import os
from pathlib import Path
from pydub.exceptions import CouldntDecodeError
from pydub import AudioSegment
from pydub.utils import get_array_type
import csv
import librosa as rosa
import array
import numpy as np
import feature_extraction as fe
import h5py
from audio_management import get_soundfile, AudioAugmenter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

groups_file = "supervised_groups.txt"
processed_file = "./dataset_cqt_train.h5"
val_file = "./dataset_cqt_validate.h5"
noise_files = 'noise_augs.txt'

max_per_group = 150
num_val_feats = 10
num_augs = 7
min_amt = 0.1
max_amt = 0.6

aa = AudioAugmenter(noise_files, num_active_files=15, switch_after=8)
max_length=1
hop=256
bins=64
out_segments=32
target_sr=22050

# parse groups from text files
groups = []
with open(groups_file, newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        groups.append(row)

# get audio files from directories
group_files = []
for pair in groups:
    fpaths, mod = pair
    mod = mod.strip(' ')

    fpaths = fpaths.split('&')
    wavs = []
    for fpath in fpaths:
        fpath = fpath.strip('" ')
        dir_wavs = os.listdir(fpath)
        dir_wavs = [os.path.join(fpath, x) for x in dir_wavs if (x[-4:] in ['.wav', '.WAV', '.aif']) or (x[-3:] == '.wv')]
        wavs.extend(dir_wavs)

    if mod:
        wavs = [x for x in wavs if mod in x]   
    np.random.shuffle(wavs)
    wavs = wavs[:max_per_group]
    group_files.append(wavs)


for i, g in enumerate(group_files):
    feats = []
    group_name = g[0].split('\\')[-2]
    group_id = f'{i}{group_name}'

    audios = []
    for s in g:
        print(s)

        try:
            x, sr = get_soundfile(s, target_sr)
        except Exception:
             print(f'failed {s}, skipping')
             continue
        # cut down to near max length in case it's a weirdly long file
        x = x[:int((max_length + 0.25) * target_sr)]

        # pad in case it's less than a frame
        if len(x) < 2 * hop:
            difference = (2 * hop) - len(x)
            x = np.concatenate([x, np.zeros(difference)])
        
        # normalize
        x = np.clip(x / np.max(np.abs(x)), -1, 1)

        audios.append(x)

    slice_locs = np.cumsum([0] + [len(x) for x in audios][:-1])
    slice_locs_frames = rosa.samples_to_frames(slice_locs, hop_length=hop)

    synth_audio = np.concatenate(audios)

    augmented_audios = aa.augment_long_file(synth_audio, num_augs, min_amt, max_amt)

    all_cqt_feats = []
    for audio_version in ([synth_audio] + augmented_audios):

        cqts, _, _, _ = fe.extract_cqt_for_slices(
            audio_version,
            target_sr,
            max_length=max_length,
            hop=hop,
            bins=bins,
            out_segments=out_segments,
            given_onsets=slice_locs_frames,
            ret_slice_objects=False,
            track_beats=False
            )

        all_cqt_feats.extend(cqts)

    stacked_feats = np.stack(all_cqt_feats, 0)
    
    if stacked_feats.shape[0] > num_val_feats * 4:

        val_choose = np.random.choice(len(stacked_feats), num_val_feats, replace=False)
        val_feats = stacked_feats[val_choose]
        
        with h5py.File(val_file, 'a') as f:
            f.create_dataset(group_id, data=val_feats)

        train_feats = np.delete(stacked_feats, val_choose, axis=0)
    else:
        train_feats = stacked_feats

    with h5py.File(processed_file, 'a') as f:
        f.create_dataset(group_id, data=train_feats)



# for i, g in enumerate(group_files):
#     feats = []
#     group_name = g[0].split('\\')[-2]
#     group_id = f'{i}{group_name}'

#     for s in g:
#         print(s)

#         try:
#             x, sr = get_soundfile(s)
#         except Exception:
#              print(f'failed {s}, skipping')
#              continue

#         augmented = aa.augment_file(x, num_augs=num_augs, min_amt=min_amt, max_amt=max_amt)

#         for sound in ([x] + augmented):
#             try:
#                 feat = extract_cqt_for_embedding(sound, sr, hop=256, bins=64, out_segments=32, ret_spec=True)
#             except librosa.ParameterError:
#                 print(f'failed {s}, skipping')
#                 continue
#             feats.append(feat)

#     all_feats = np.stack(feats, 0)

#     with h5py.File(processed_file, 'a') as f:
#         f.create_dataset(group_id, data=all_feats)






