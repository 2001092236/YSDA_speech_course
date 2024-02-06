from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        '''
            waveform: np.array (1d)
            output: 2D array (time, time-in-frame)
        '''
        # Your code here
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^
        pad_len = self.window_size // 2
        waveform = np.pad(waveform, (pad_len, pad_len))
        
        windows = []
        for i in range(0, len(waveform) - self.window_size + 1, self.hop_length):
            windows.append(waveform[i: i + self.window_size])
        
        windows = np.vstack(windows)
        return windows



class Hann:
    def __init__(self, window_size=1024):
        # Your code here
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^
        self.window_size = window_size
        self.hann = scipy.signal.windows.hann(window_size, sym=False)[None, :]
    
    def __call__(self, windows):
        assert self.window_size == windows.shape[1]

        return self.hann * windows


class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        '''
            windows: [n_frames, frame_length] --- wav
        '''
        spec = np.abs(np.fft.rfft(windows, axis=1))

        if self.n_freqs is not None:
            spec = spec[:, :self.n_freqs]

        return spec


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        self.mel_banks = librosa.filters.mel(sr=sample_rate,
                                             n_fft=n_fft,
                                             n_mels=n_mels,
                                             fmin=1,
                                             fmax=8192).T
        inverse_mel_sq = np.linalg.inv(self.mel_banks.T @ self.mel_banks)
        self.pseudo_inverse_mel = (self.mel_banks @ inverse_mel_sq).T

    def __call__(self, spec):
        mel = spec @ self.mel_banks
        return mel

    def restore(self, mel):
        spec = mel @ self.pseudo_inverse_mel
        return spec
        
class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        # return self.square(self.fft(self.hann(self.windowing(waveform))))
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class PitchUp:
    def __init__(self, num_mels_up):
        # (shift up)
        self.num_mels_up = num_mels_up

    def __call__(self, mel):
        res = np.zeros_like(mel)
        res[:, self.num_mels_up:] = mel[:, :-self.num_mels_up]
        return res



class PitchDown:
    def __init__(self, num_mels_down):
        self.num_mels_down = num_mels_down

    def __call__(self, mel):
        res = np.zeros_like(mel)
        res[:, :-self.num_mels_down] = mel[:, self.num_mels_down:]
        return res



class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        '''

        '''

        self.speed_up_factor = speed_up_factor


    def __call__(self, mel):
        ids = np.linspace(0, len(mel) - 1, int(len(mel) / self.speed_up_factor)).astype(int)
        return mel[ids, :]


class Loudness:
    def __init__(self, loudness_factor):
        self.loudness_factor = loudness_factor

    def __call__(self, mel):
        return mel * self.loudness_factor


class TimeReverse:
    def __call__(self, mel):
        return mel[::-1]



class VerticalSwap:
    def __call__(self, mel):
        return mel[:, ::-1]


class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        self.quantile = quantile

    def __call__(self, mel):
        threshold = np.quantile(a=mel, q=self.quantile)
        return np.where(mel > threshold, mel, 0.)




class Cringe1:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class Cringe2:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^
