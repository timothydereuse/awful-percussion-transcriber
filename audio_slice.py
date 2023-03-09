import feature_extraction as fe
from importlib import reload
from librosa import resample
from librosa.util.exceptions import ParameterError
import numpy as np
reload(fe)

class AudioSlice(object):

    def __init__(self, audio, sr, precomputed_feats=None, fname=None, normalize=False):

        if precomputed_feats is not None:
            self.feats = precomputed_feats
        else:
            try:
                self.feats = fe.extract_cqt_for_embedding(audio, sr)
            except ParameterError:
                self.feats = None
                return

        self.fname = fname
        self.audio = audio
        self.rms = np.sqrt(np.sum(audio * audio) / audio.shape[0])
        if normalize:
            self.audio = audio / max(np.abs(audio))
        self.sr = sr
        