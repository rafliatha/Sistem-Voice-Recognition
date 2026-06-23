import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Fungsi untuk membaca dan menormalisasi file audio
def bacaaudio(namafile):
    lajusampling, data = wavfile.read(namafile)
    # Jika audio stereo (2 channel), ambil channel pertama saja
    if data.ndim > 1:
        data = data[:, 0]
        
    # Pengecekan agar terhindar dari error pembagian dengan 0
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    else:
        print("Peringatan: File audio kosong atau hening.")
        
    return lajusampling, data

# =====================================================================
# FUNGSI DFT (Pengganti FFT) - Versi Optimasi (Partial DFT)
# Menghitung persamaan: X_k = sum(x_n * e^(-2j * pi * k * n / N))
# =====================================================================
def hitungdft(data, lajusampling):
    N = len(data)
    
    # TRIK RAHASIA: 
    # Karena kita hanya butuh rentang 0 - 1000 Hz untuk plot dan klasifikasi,
    # kita tidak perlu memutar loop sampai ujung data N. Kita hitung batas maksimal indeks (k).
    batas_frekuensi_hz = 1000
    K_maks = int(batas_frekuensi_hz * N / lajusampling) + 1
    
    data_dft = np.zeros(K_maks, dtype=complex)
    frekuensi_dft = np.zeros(K_maks)
    n = np.arange(N)
    
    print(f"Total sampel audio utuh: {N} data")
    print(f"Menghitung DFT murni untuk rentang 0 - 1000 Hz ({K_maks} iterasi)...")
    
    for k in range(K_maks):
        # Menampilkan progress persentase di terminal
        if k % max(1, (K_maks // 10)) == 0:
            print(f"Progress: {int((k / K_maks) * 100)}%")
            
        # Perhitungan matematis DFT
        eksponensial = np.exp(-2j * np.pi * k * n / N)
        data_dft[k] = np.sum(data * eksponensial)
        frekuensi_dft[k] = k * lajusampling / N
        
    print("Progress: 100% - Perhitungan DFT Selesai!\n")
    
    # Menghitung magnitudo (amplitudo absolut dibagi N)
    magnitudo = np.abs(data_dft) / N
    
    return frekuensi_dft, magnitudo

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
    
    # Menghitung FFT diganti dengan DFT Manual
    frekuensi, magnitudo = hitungdft(data, lajusampling)
    
    # Identifikasi
    frekuensidominan = carifrekuensidominan(frekuensi, magnitudo)
    kategori = klasifikasikanfrekuensi(frekuensidominan)
    
    print(f"Frekuensi Dominan : {frekuensidominan:.2f} Hz")
    print(f"Kategori Deteksi  : {kategori}\n")
    
    # Plot Grafik
    plt.figure(figsize=(10, 4))
    plt.plot(frekuensi, magnitudo)
    plt.title('Spektrum Frekuensi (Domain Frekuensi - DFT Manual)')
    plt.xlabel('Frekuensi [Hz]')
    plt.ylabel('Magnitudo')
    # Limit tetap 0-1000 Hz, hasil grafik akan sama persis dengan FFT
    plt.xlim(0, 1000) 
    plt.grid()
    plt.show()

# Menginputkan file audio
if __name__ == "__main__":
    prosesaudio('DSP/Voice_Recognition/hell_yeah.wav')