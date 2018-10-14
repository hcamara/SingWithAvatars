from __future__ import print_function
import librosa
import wave
import numpy as np
from pydub import AudioSegment


class Sounds(object):

    def __init__(self, source):
        self.y, self.sr = librosa.load(source, sr=None)
        self.src = source

    def speed(self, sp):
        tmp = librosa.stft(self.y, n_fft=2048, hop_length=512)
        tmp_speed = librosa.phase_vocoder(tmp, sp, hop_length=512)
        return librosa.istft(tmp_speed, hop_length=512)

    def removePhase(self):
        S_full, phase = librosa.magphase(librosa.stft(self.y))
        return S_full
        
    def separate(self):
        S_full, phase = librosa.magphase(librosa.stft(self.y))
        S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=self.sr)))

        S_filter = np.minimum(S_full, S_filter)
        margin_i, margin_v = 2, 10
        power = 100
        mask_i = librosa.util.softmask(S_filter,
                                    margin_i * (S_full - S_filter),
                                    power=power)

        mask_v = librosa.util.softmask(S_full - S_filter,
                                    margin_v * S_filter,
                                    power=power)
        S_foreground = mask_v * S_full
        S_background = mask_i * S_full

        return [S_full, S_foreground, S_background]

    def write(self, name, src):
        y_sp = librosa.istft(src, hop_length=512)
        librosa.output.write_wav(name, y_sp, self.sr)