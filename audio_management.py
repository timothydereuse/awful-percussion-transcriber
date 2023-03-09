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
import audio_slice as ausl
from scipy.interpolate import interp1d


def get_soundfile(fname, to_sr=None):
    s = AudioSegment.from_file(file=fname)
    s = s.set_channels(1)
    sr = s.frame_rate
    bit_depth = s.sample_width * 8
    array_type = get_array_type(bit_depth)
    s = np.array(array.array(array_type, s._data))
    s = s / (2 ** (bit_depth - 1))
    if to_sr and (sr != to_sr):
        s = rosa.resample(s, sr, to_sr)
        sr = to_sr
    return s, sr


def normalize(inp):
    return np.clip(inp / np.max(np.abs(inp)), -1, 1)


class AudioAugmenter():

    def __init__(self, noise_file, num_active_files=10, switch_after=15, target_sr=22050):
        with open(noise_file) as file:
            self.files = [line.rstrip().strip('"') for line in file]
            self.counter = 0
            self.num_active_files = num_active_files
            self.switch_after = switch_after
            self.target_sr = target_sr
            self.load_new_files()

    def load_new_files(self):
        print(f'audio augmenter is loading {self.num_active_files} new files...')
        self.new_files = np.random.choice(self.files, self.num_active_files, replace=False)
        af = []
        for fpath in self.new_files:
            print(f'   loading {fpath}')
            af.append(get_soundfile(fpath, to_sr=self.target_sr)[0])
        self.active_files = af
        self.counter = 0


    def augment_file(self, inp, num_augs=2, min_amt=0, max_amt=0.2):
        tar_len = len(inp)
        max_inp = np.max(np.abs(inp))
        res = []
        for i in range(num_augs):
            choose_noise = np.random.choice(self.active_files)
            if len(choose_noise) <= tar_len:
                choose_noise = np.concatenate([choose_noise, np.zeros(tar_len)])

            snippet = np.zeros(1)
            counter = 0
            while max(snippet) == 0:
                start = np.random.randint(0, len(choose_noise) - tar_len)
                end = start + tar_len
                snippet = choose_noise[start:end]
                counter += 1
                if counter == 50:
                    snippet = np.random.uniform(-1, 1, tar_len)

            amt = np.random.uniform(min_amt, max_amt)
            augmented = inp + (snippet * amt * max_inp / np.max(snippet))
            res.append(augmented)
        
        self.counter += 1
        if self.counter > self.switch_after:
            self.load_new_files()

        return res


    def stitch_together(self, target_samples, min_amt=0.1, max_amt=0.4, min_snip_len=1, max_snip_len=5):

        snippets = []
        while sum([len(x) for x in snippets]) < target_samples:
            choose_noise = np.random.choice(self.active_files)
            tar_len = np.random.randint(min_snip_len * self.target_sr, max_snip_len * self.target_sr)
            amt = np.random.uniform(min_amt, max_amt)

            start = np.random.randint(0, max(len(choose_noise) - tar_len, 1))
            end = start + tar_len
            snippet = choose_noise[start:end]
            snippet = normalize(snippet) * amt
            snippets.append(snippet)

        stitched = np.concatenate(snippets)
        self.counter += 1

        if self.counter > self.switch_after:
            self.load_new_files()

        return stitched


    def augment_long_file(self, inp, num_augs=2, min_amt=0, max_amt=0.2, min_snip_len=1, max_snip_len=5):
        aug_files = []
        for i in range(num_augs):
            new_stitch = self.stitch_together(len(inp), min_amt, max_amt, min_snip_len, max_snip_len)
            aug = inp + new_stitch[:len(inp)]
            aug_files.append(aug)
        return aug_files


def slice_long_sample(y, sr, length_limit=None, hop=256, poll_every=50):

    if length_limit and (len(y) / sr) > length_limit:
        y = y[len(y) - (length_limit * sr // 2):len(y) + (length_limit * sr // 2)]

    onsets = rosa.onset.onset_detect(y=y, sr=sr, backtrack=True, hop_length=hop)
    onset_times = rosa.frames_to_samples(onsets, hop_length=hop)

    # remove onsets with deltas too small
    to_remove = np.where(np.diff(onset_times) <= 2)[0] + 1
    onset_times = np.delete(onset_times, to_remove)

    onset_times = np.concatenate([onset_times, [len(y)]])
    segmented = [y[onset_times[n]:onset_times[n + 1]] for n in range(len(onset_times) - 1)]

    slices = []
    for i, s in enumerate(segmented):
        if not i % poll_every and i > 1:
            print(rf'calculating features for slice {i}/{len(segmented)}')
        try:    
            slice = ausl.AudioSlice(s, sr)
        except ValueError:
            print(rf'calculating features for slice {i} failed, skipping')
            continue
        slices.append(slice)

    tempo, beats = rosa.beat.beat_track(y=y, sr=sr)

    beat_frames = rosa.frames_to_samples(beats)
    beat_frames = np.concatenate([[0], beat_frames])
    interp_func = interp1d(beat_frames, np.arange(len(beat_frames)), kind='linear', fill_value='extrapolate')
    onset_times_beats = interp_func(onset_times)

    return slices, onset_times, tempo, onset_times_beats

