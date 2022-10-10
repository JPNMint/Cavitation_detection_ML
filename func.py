from pathlib import Path
from scipy.io import wavfile

import librosa







def read_wav(dataset_names):
    samplerate = []
    df = []
    for data in dataset_names:
        path =  Path.cwd()/"data"
        sf, d = wavfile.read(path/data)
        samplerate.append(sf)
        df.append(d)
    return samplerate , df