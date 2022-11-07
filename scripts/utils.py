from pathlib import Path
from scipy.io import wavfile
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.stats import norm, kurtosis, skew
import pandas as pd
import warnings


## reading in multiple wav files
def read_wav(dataset_names):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samplerate = []
        df = []
        for data in dataset_names:
            path =  Path.cwd()/"data"
            sf, d = wavfile.read(path/data)
            samplerate.append(sf)
            df.append(d)
        return samplerate , df

## windowing samples
def splitting (sr, df, sec):
    #get sr for section
    seg_len = int(sr * sec)

    splits = []

    for data in df:
        #get number of sections
        sections = int(np.ceil(len(data) / seg_len))
        for i in range(sections):
            #slice section range
            t = data[i * seg_len: (i + 1) * seg_len]
            splits.append(t)


    return splits


## fourier transform multiple samples
def fourier_trans(sr, data):
    df_ftt = []
    freq = []

    df = pd.DataFrame()
    x = 0
    for i in data:

        length = i.shape[0] / sr
        N = i.shape[0]
        n = np.arange(N)
        freq.append(n/length)
        df_i = pd.DataFrame(fft(i).real, columns = [x])
        x += 1
        df = pd.concat([df,df_i], axis = 1)

    return freq, df

def plot_whole_freq(frequency, spectrum):

    plt.figure(figsize = (12, 6))
    plt.subplot(121)

    plt.stem(frequency[1], np.abs(spectrum), 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')

    plt.show()

def plot_half_freq(frequency, spectrum):

    X = np.fft.fft(spectrum)
    X_mag = np.abs(X)
    f = np.linspace(0, sr[0], len(X_mag))
    half = int(len(X_mag)/2)
    fft_fre = np.fft.rfftfreq(len(X_mag), d=1./frequency)

    abs_spec = abs(X_mag[:half+1])
    plt.figure(figsize=(5, 5))
    plt.plot(fft_fre, abs_spec) # magnitude spectrum
    plt.xlabel('Frequency (Hz)')
    plt.show()




## statistical features

def feature_extraction(df, label):
    column_name = ["mean", "median", "quartile_25", "quartile_75", "Max", "Min", "quartile", "std", "rms", "sra", "ff", "clf", "cf", "kurtosis", "skew"]
    df_features = pd.DataFrame(columns = column_name) #

    for column in df:
        feature_list = []

           # central trend statistics
        data_mean = np.mean(df[column])
        data_median = np.median(df[column])
        data_quartile_025 = np.quantile(df[column], 0.25)
        data_quartile_075 = np.quantile(df[column], 0.75)

           # dispersion degree statistics
        data_Minimum = np.min(df[column])
        data_Maximum = np.max(df[column])
        data_quartile = data_quartile_075 - data_quartile_025
        data_std = np.std(df[column])
        data_rms = np.sqrt((np.mean(df[column]**2)))
        data_sra = (np.sum(np.sqrt(np.abs(df[column])))/len(df[column]))**2

           # distribution shape statistics
        data_kurtosis = kurtosis(df[column])
        data_skew = skew(df[column])

        data_avg = np.mean(np.abs(df[column]))
        data_ff = data_rms / data_avg

        data_clf = np.max(np.abs(df[column])) / data_sra
        data_cf = np.max(np.abs(df[column])) / data_rms

        feature_list = [data_mean, data_median, data_quartile_025, data_quartile_075, data_Maximum, data_Minimum, data_quartile, data_std, data_rms, data_sra, data_ff, data_clf, data_cf , data_kurtosis, data_skew]
        feature_list = pd.DataFrame(data=feature_list).T #,
        feature_list.columns = column_name

        df_features = pd.concat([df_features,feature_list])

    if label == "Cavitation":
        df_features["Cavitation"] = 1

    if label == "No Cavitation":
        df_features["Cavitation"] = 0

    if label != "No Cavitation" and label != "Cavitation":
        df_features["Cavitation"] = label
    return df_features








#feature pipeline

def feature_pipe (datasets, splits_in_sec, label):
    sr, df = read_wav(datasets)
    df_split = splitting(sr[0],df, splits_in_sec)
    freq, df_ftt = fourier_trans(sr[0],df_split)
    output_df = feature_extraction(df_ftt, label)
    return len(df_split), output_df