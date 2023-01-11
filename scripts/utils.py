from pathlib import Path
from scipy.io import wavfile
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.stats import norm, kurtosis, skew
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import librosa
import librosa.display



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



### plotting time waveform



def plot_timewave(datapath):
    sr, data = wavfile.read(datapath)
    length = data.shape[0] / sr
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data)

    plt.title("Raw Audio")

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.show()


##spectro librosa

def spectro(data_path):
    x, sr = librosa.load(data_path)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()


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

def feature_extraction(df, label, sr):
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

        # Mel Frequency Cepstral Coefficients (MFCC)
        # mel = librosa.feature.mfcc(y = np.array(df[column]))#, sr = sr, n_mels=128
        # mel_mean = np.mean(mel)
        # mel_var = np.var(mel)

        feature_list = [data_mean, data_median, data_quartile_025, data_quartile_075, data_Maximum, data_Minimum,
                        data_quartile, data_std, data_rms, data_sra, data_ff, data_clf, data_cf , data_kurtosis,
                        data_skew

                        ]
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


def feature_extraction_mel(df, label, sr):
    column_name = ["mean", "median", "quartile_25", "quartile_75", "Max", "Min", "quartile", "std", "rms", "sra", "ff", "clf", "cf", "kurtosis", "skew","mel_mean","mel_var",
                    "mel_median", "mel_quartile_25", "mel_quartile_75", "mel_Max", "mel_Min", "mel_quartile", "mel_std", "mel_rms", "mel_sra"]#, "mel_ff", "mel_clf", "mel_cf", "mel_kurtosis", "mel_skew", "mel_avg"]
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

        # Mel Frequency Cepstral Coefficients (MFCC)
        mel = librosa.feature.mfcc(y = np.array(df[column]))#, sr = sr, n_mels=128
        mel_mean = np.mean(mel)
        mel_var = np.var(mel)

           # central trend statistics

        mel_median = np.median(mel)
        mel_quartile_025 = np.quantile(mel, 0.25)
        mel_quartile_075 = np.quantile(mel, 0.75)

           # dispersion degree statistics
        mel_Minimum = np.min(mel)
        mel_Maximum = np.max(mel)
        mel_quartile = mel_quartile_075 - mel_quartile_025
        mel_std = np.std(mel)
        mel_rms = np.sqrt((np.mean(mel**2)))
        mel_sra = (np.sum(np.sqrt(np.abs(mel)))/len(mel))**2

        #    # distribution shape statistics
        # mel_kurtosis = kurtosis(mel)
        # mel_skew = skew(mel)
        #
        # mel_avg = np.mean(np.abs(mel))
        # mel_ff = mel_rms / mel_avg
        #
        # mel_clf = np.max(np.abs(mel)) / mel_sra
        # mel_cf = np.max(np.abs(mel)) / mel_rms


        feature_list = [data_mean, data_median, data_quartile_025, data_quartile_075, data_Maximum, data_Minimum,
                        data_quartile, data_std, data_rms, data_sra, data_ff, data_clf, data_cf , data_kurtosis,
                        data_skew, mel_mean, mel_var,
                        mel_median, mel_quartile_025, mel_quartile_075, mel_Maximum, mel_Minimum,
                         mel_quartile, mel_std, mel_rms, mel_sra #, mel_avg, mel_ff, mel_clf, mel_cf, mel_kurtosis,
                        # mel_skew
                        ]
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


def feature_extraction_single(df, label):
    column_name = ["mean", "median", "quartile_25", "quartile_75", "Max", "Min", "quartile", "std", "rms", "sra", "ff", "clf", "cf", "kurtosis", "skew"]
    df_features = pd.DataFrame(columns = column_name) #


    feature_list = []

           # central trend statistics
    data_mean = np.mean(df[df])
    data_median = np.median(df[df])
    data_quartile_025 = np.quantile(df[df], 0.25)
    data_quartile_075 = np.quantile(df[df], 0.75)

           # dispersion degree statistics
    data_Minimum = np.min(df[df])
    data_Maximum = np.max(df[df])
    data_quartile = data_quartile_075 - data_quartile_025
    data_std = np.std(df[df])
    data_rms = np.sqrt((np.mean(df[df]**2)))
    data_sra = (np.sum(np.sqrt(np.abs(df[df])))/len(df[df]))**2

        # distribution shape statistics
    data_kurtosis = kurtosis(df[df])
    data_skew = skew(df[df])

    data_avg = np.mean(np.abs(df[df]))
    data_ff = data_rms / data_avg

    data_clf = np.max(np.abs(df[df])) / data_sra
    data_cf = np.max(np.abs(df[df])) / data_rms

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







def feature_pipe (path, splits_in_sec):
    target_path = Path.cwd()/path
    print(f"Extracting features from {target_path}")
    if not (target_path/"cavitation").is_dir():
        print("Folder named cavitation does not exist!")
        return
    if not (target_path/"no_cavitation").is_dir():
        print("Folder named no_cavitationn does not exist!")
        return
    no_cav_f_names = list((target_path/"no_cavitation").glob('*.wav'))
    cav_f_names = list((target_path/"cavitation").glob('*.wav'))


    sr_cav, df_cav = read_wav(cav_f_names)
    df_split_cav = splitting(sr_cav[0],df_cav, splits_in_sec)
    freq_cav, df_ftt_cav = fourier_trans(sr_cav[0],df_split_cav)
    output_df_cav = feature_extraction(df_ftt_cav, "Cavitation", sr = sr_cav[0]) #cavitation = 1



    sr_no_cav, df_no_cav = read_wav(no_cav_f_names)
    df_split_no_cav = splitting(sr_no_cav[0], df_no_cav, splits_in_sec)
    freq_no_cav, df_ftt_no_cav = fourier_trans(sr_no_cav[0], df_split_no_cav)
    output_df_no_cav = feature_extraction(df_ftt_no_cav, "No Cavitation", sr = sr_no_cav[0]) #no_cavitation = 0


    df = pd.concat([output_df_cav, output_df_no_cav])
    print(df)
    return len(df_split_cav)+len(df_split_no_cav), df

import splitfolders
import shutil
import os
def create_split():


    splitfolders.ratio(Path().resolve()/"data/Training_all/oil/", output=Path().resolve()/"data/oil_split/", seed=1623806, ratio=(.8,0.2))

    splitfolders.ratio(Path().resolve()/"data/Training_all/teflon", output=Path().resolve()/"data/teflon_split/", seed=1623806, ratio=(.8,0.2))
    splitfolders.ratio(Path().resolve()/"data/Training_all/vacuum", output=Path().resolve()/"data/vacuum_split/", seed=1623806, ratio=(.8,0.2))


    # only oil
    if not os.path.exists(Path().resolve()/"data/train_oil"):
        os.makedirs(Path().resolve()/"data/train_oil")

    src_path = Path().resolve()/"data/oil_split/train/cavitation"
    dst_path = Path().resolve()/"data/train_oil/cavitation"
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    src_path = Path().resolve()/"data/oil_split/train/no_cavitation"
    dst_path = Path().resolve()/"data/train_oil/no_cavitatio"
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    ##test
    if not os.path.exists(Path().resolve() / "data/test_oil"):
        os.makedirs(Path().resolve() / "data/test_oil")
    src_path = Path().resolve()/"data/oil_split/val/cavitation"
    dst_path = Path().resolve()/"data/test_oil/cavitation"
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    src_path = Path().resolve()/"data/oil_split/val/no_cavitation"
    dst_path = Path().resolve()/"data/test_oil/no_cavitatio"
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
def create_oil_vacuum():

    # oil vacuum train
    if not os.path.exists(Path().resolve() / "data/train_oil_vacuum"):
        os.makedirs(Path().resolve() / "data/train_oil_vacuum")
    src_path = Path().resolve()/"data/oil_split/train/cavitation"
    dst_path = Path().resolve()/"data/train_oil_vacuum/cavitation"
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    src_path = Path().resolve()/"data/oil_split/train/no_cavitation"
    dst_path = Path().resolve()/"data/train_oil_vacuum/no_cavitation"
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    src_path = Path().resolve()/"data/vacuum_split/train/cavitation"
    dst_path = Path().resolve()/"data/train_oil_vacuum/cavitation"

    file_names = os.listdir(src_path)

    for file_name in file_names:
        shutil.copy2(os.path.join(src_path, file_name), dst_path)



    #oil vacuum test
    if not os.path.exists(Path().resolve() / "data/test_oil_vacuum"):
        os.makedirs(Path().resolve() / "data/test_oil_vacuum")
    src_path = Path().resolve() / "data/oil_split/val/cavitation"
    dst_path = Path().resolve() / "data/test_oil_vacuum/cavitation"
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    src_path = Path().resolve() / "data/oil_split/val/no_cavitation"
    dst_path = Path().resolve() / "data/test_oil_vacuum/no_cavitation"
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    src_path = Path().resolve() / "data/vacuum_split/val/cavitation"
    dst_path = Path().resolve() / "data/test_oil_vacuum/cavitation"

    file_names = os.listdir(src_path)

    for file_name in file_names:
        shutil.copy2(os.path.join(src_path, file_name), dst_path)

def create_oil_vacuum_teflon():

    # oil vacuum train teflon
    if not os.path.exists(Path().resolve() / "data/train_oil_vacuum_teflon"):
        os.makedirs(Path().resolve() / "data/train_oil_vacuum_teflon")
    src_path = Path().resolve() / "data/oil_split/train/cavitation"
    dst_path = Path().resolve() / "data/train_oil_vacuum_teflon"
    shutil.copy2(src_path, dst_path)
    src_path = Path().resolve() / "data/oil_split/train/no_cavitation"
    dst_path = Path().resolve() / "data/train_oil_vacuum_teflon"
    shutil.copy2(src_path, dst_path)

    src_path = Path().resolve() / "data/vacuum_split/train/cavitation"
    dst_path = Path().resolve() / "data/train_oil_vacuum_teflon/cavitation"

    file_names = os.listdir(src_path)

    for file_name in file_names:
        shutil.copy2(os.path.join(src_path, file_name), dst_path)

    src_path = Path().resolve() / "data/teflon_split/train/cavitation"
    dst_path = Path().resolve() / "data/train_oil_vacuum_teflon/cavitation"

    file_names = os.listdir(src_path)

    for file_name in file_names:
        shutil.copy2(os.path.join(src_path, file_name), dst_path)

    # oil vacuum  teflon test
    if not os.path.exists(Path().resolve() / "data/test_oil_vacuum_teflon"):
        os.makedirs(Path().resolve() / "data/test_oil_vacuum_teflon")

    src_path = Path().resolve() / "data/oil_split/val/cavitation"
    dst_path = Path().resolve() / "data/test_oil_vacuum_teflon"
    shutil.copy2(src_path, dst_path)

    src_path = Path().resolve() / "data/oil_split/val/no_cavitation"
    dst_path = Path().resolve() / "data/test_oil_vacuum_teflon"
    shutil.copy2(src_path, dst_path)


    src_path = Path().resolve() / "data/vacuum_split/val/cavitation"
    dst_path = Path().resolve() / "data/test_oil_vacuum_teflon/cavitation"

    file_names = os.listdir(src_path)

    for file_name in file_names:
        shutil.copy2(os.path.join(src_path, file_name), dst_path)

    src_path = Path().resolve() / "data/teflon_split/val/cavitation"
    dst_path = Path().resolve() / "data/test_oil_vacuum_teflon/cavitation"

    file_names = os.listdir(src_path)

    for file_name in file_names:
        shutil.copy2(os.path.join(src_path, file_name), dst_path)
def create_split_electro():


    splitfolders.ratio(Path().resolve()/"data/electrochem_20_splits/", output=Path().resolve()/"data/electrochem_20_splits_train_test/", seed=1623806, ratio=(.8,0.2))




if __name__ == "__main__":
    # create_split()
    # create_oil_vacuum()
    #create_oil_vacuum_teflon()
    create_split_electro()