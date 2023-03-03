import feature_extraction as fe
from importlib import reload
from librosa import resample
from librosa.util.exceptions import ParameterError
import numpy as np
reload(fe)

class AudioSlice(object):

    def __init__(self, audio, sr, fname=None, normalize=True):

        try:
            self.feats = fe.extract_features_from_audio(audio, sr)
        except ParameterError:
            self.feats = None
            return

        self.fname = fname
        self.audio = audio
        if normalize:
            self.audio = audio / max(np.abs(audio))
        self.sr = sr
        self.rms = np.sqrt(np.sum(audio * audio) / audio.shape[0])

    def get_rev_slice(self):
        new_slice = AudioSlice(self.audio[::-1], self.sr, self.fname)
        return new_slice

    def get_resampled_slice(self, scale_amt=None):
        if not scale_amt:
            scale_amt = np.random.exponential(0.9)
        new_slice = AudioSlice(resample(self.audio[::-1], self.sr, self.sr * scale_amt), self.sr, self.fname)
        return new_slice