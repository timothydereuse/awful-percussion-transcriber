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

class AudioAugmenter():

    def __init__(self, noise_file, num_active_files=10, switch_after=15):
        with open(noise_file) as file:
            self.files = [line.rstrip().strip('"') for line in file]
            self.counter = 0
            self.num_active_files = num_active_files
            self.switch_after = switch_after
            self.load_new_files()

    def load_new_files(self):
        self.new_files = np.random.choice(self.files, self.num_active_files, replace=False)
        self.active_files = [get_soundfile(fpath)[0] for fpath in self.new_files]

    def augment_file(self, inp, num_augs=2, min_amt=0, max_amt=0.2):
        tar_len = len(inp)
        max_inp = np.max(np.abs(inp))
        res = []
        for i in range(num_augs):
            choose_noise = np.random.choice(self.active_files)
            if len(choose_noise) <= tar_len:
                choose_noise = np.concatenate([choose_noise, np.zeros(tar_len)])
            start = np.random.randint(0, len(choose_noise) - tar_len)
            end = start + tar_len
            snippet = choose_noise[start:end]
            amt = np.random.uniform(min_amt, max_amt)
            augmented = inp + (snippet * amt * max_inp / np.max(snippet))
            res.append(augmented)
        
        self.counter += 1
        if self.counter > self.switch_after:
            self.load_new_files()

        return res

# a = AudioAugmenter('noise_augs.txt')
# res = a.augment_file(s, num_augs=10, min_amt=0.05, max_amt=0.2)
# data = res[6]
# out =  np.int16(data / np.max(np.abs(data)) * 32767) 
# write('test7.wav', sr, out)

groups_file = "supervised_groups.txt"
processed_file = "./dataset_groups_aug.h5"
max_per_group = 5
num_augs = 3
min_amt = 0.1
max_amt = 0.25
aa = AudioAugmenter('noise_augs.txt')

groups = []
with open(groups_file, newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        groups.append(row)

group_files = []
for pair in groups:
    fpaths, mod = pair
    mod = mod.strip(' ')

    fpaths = fpaths.split('&')
    wavs = []
    for fpath in fpaths:
        fpath = fpath.strip('" ')
        dir_wavs = os.listdir(fpath)
        dir_wavs = [os.path.join(fpath, x) for x in dir_wavs if x[-4:] in ['.wav', '.WAV', '.aif']]
        wavs.extend(dir_wavs)

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

        augmented = aa.augment_file(x, num_augs=num_augs, min_amt=min_amt, max_amt=max_amt)

        for sound in ([x] + augmented):
            try:
                feat = extract_features_for_embedding(sound, sr)
            except librosa.ParameterError:
                print(f'failed {s}, skipping')
                continue
            feats.append(feat)

    all_feats = np.stack(feats, 0)

    with h5py.File(processed_file, 'a') as f:
        f.create_dataset(group_id, data=all_feats)
    




