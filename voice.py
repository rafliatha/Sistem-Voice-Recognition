import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Fungsi untuk membaca dan menormalisasi file audio
def bacaaudio(namafile):
    lajusampling, data = wavfile.read(namafile)
    # Jika audio stereo (2 channel), ambil channel pertama saja
    if data.ndim > 1:
        data = data[:, 0]
    # Normalisasi amplitudo
    data = data / np.max(np.abs(data))
    return lajusampling, data

# Fungsi untuk menerapkan Fast Fourier Transform (FFT)
def hitungfft(data, lajusampling):
    N = len(data)
    datafft = np.fft.fft(data)
    frekuensifft = np.fft.fftfreq(N, d=1/lajusampling)
    magnitudo = np.abs(datafft) / N
    # Mengembalikan setengah bagian positif dari spektrum
    return frekuensifft[:N // 2], magnitudo[:N // 2]

# Fungsi untuk mencari titik puncak frekuensi
def carifrekuensidominan(frekuensi, magnitudo):
    indeks = np.argmax(magnitudo)
    return frekuensi[indeks]

# Fungsi untuk klasifikasi berdasarkan rentang frekuensi
def klasifikasikanfrekuensi(frekuensi):
    if 85 <= frekuensi <= 180:
        return "Laki-laki Dewasa"
    elif 165 <= frekuensi <= 255:
        return "Perempuan Dewasa"
    elif 200 <= frekuensi <= 300:
        return "Anak Laki-laki"
    elif 250 <= frekuensi <= 400:
        return "Anak Perempuan"
    else:
        return "Tidak Terklasifikasi"

# Main program
def prosesaudio(namafile):
    print(f"--- Memproses File: {namafile} ---")
    lajusampling, data = bacaaudio(namafile)
    
    # Menghitung FFT
    frekuensi, magnitudo = hitungfft(data, lajusampling)
    
    # Identifikasi
    frekuensidominan = carifrekuensidominan(frekuensi, magnitudo)
    kategori = klasifikasikanfrekuensi(frekuensidominan)
    
    print(f"Frekuensi Dominan : {frekuensidominan:.2f} Hz")
    print(f"Kategori Deteksi  : {kategori}\n")
    
    # Plot Grafik
    plt.figure(figsize=(10, 4))
    plt.plot(frekuensi, magnitudo)
    plt.title('Spektrum Frekuensi (Domain Frekuensi)')
    plt.xlabel('Frekuensi [Hz]')
    plt.ylabel('Magnitudo')
    plt.xlim(0, 1000)
    plt.grid()
    plt.show()

# Menginputkan file audio
if __name__ == "__main__":
    prosesaudio('DSP/Voice_Recognition/hell_yeah.wav')

# Masukkan file audio yang akan dianalisis
# Pastikan format audio .wav
# Contoh : prosesaudio('rekaman_suara.wav')