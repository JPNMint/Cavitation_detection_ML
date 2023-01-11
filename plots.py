from scipy.io import wavfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.fft import fftshift
import librosa.display

def plot_timewave(data, sr):

    length = data.shape[0] / sr
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data)

    plt.title("Raw Audio")

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.show()



def fourier_trans_plot(data, sr):
    length = data.shape[0] / sr
    N = data.shape[0]
    n = np.arange(N)
    freq = n/length
    data_fft = fft(data)

    plt.figure(figsize = (12, 6))
    plt.subplot(121)


    plt.stem(freq, np.abs(data_fft), 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')

    plt.show()

# def spectro_librosa(data_path):
#     x, sr = librosa.load(data_path)
#     X = librosa.stft(x)
#     Xdb = librosa.amplitude_to_db(abs(X))
#     plt.figure(figsize=(14, 5))
#     librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#     plt.colorbar()
def spectro_librosa(data_path, figsize=(14,5), sr=250000):

    x, sr = librosa.load(data_path, sr=sr)

    X = librosa.stft(x,  center=False)

    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(figsize=figsize)

    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')

    plt.colorbar()

def spectro_matplot(data,sr):

    plt.specgram(data, Fs=sr)
    plt.title('Spectrogram Using matplotlib.pyplot.specgram() Method')
    plt.xlabel("Time in sec")
    plt.ylabel("Frequency [Hz]")
    plt.show()


# def spectro_scipy(data,sr):
#     #sr, df = wavfile.read(Path.cwd()/"data"/data_name)
#     f, t, Sxx = signal.spectrogram(data, sr, return_onesided=True)
#
#     plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))#, shading='gouraud'
#
#     plt.ylabel('Frequency [Hz]')
#
#     plt.xlabel('Time [sec]')
#
#     plt.show()


def spectro_scipy(data, sr, figsize=(14, 5)):
    plt.figure(figsize=figsize)

    f, t, X = signal.spectrogram(data, sr, nperseg=2048, noverlap=1024 + 512)

    Xdb = 20 * np.log10(X)

    plt.pcolormesh(t, f, Xdb, cmap='viridis')

    plt.ylabel('Frequency [Hz]')

    plt.xlabel('Time [sec]')

    plt.show()

