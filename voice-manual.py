import numpy as np
import cmath # Library matematika dasar untuk bilangan kompleks (imajiner/eksponensial)
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 1. Fungsi membaca dan menormalisasi file audio
def bacaaudio(namafile):
    lajusampling, data = wavfile.read(namafile)
    
    # Jika audio stereo, ubah jadi mono (ambil channel pertama)
    if data.ndim > 1:
        data = data[:, 0]
        
    # --- PERBAIKAN: POTONG DI TENGAH & PERPANJANG RENTANG ---
    # Kita mulai dari sampel ke 22.050 (sekitar detik ke-0.5 jika laju sampling 44100Hz)
    # Ini untuk memastikan kita "menangkap" suara, bukan keheningan saat tombol record baru ditekan
    titik_awal = 22050 
    
    # KITA PERPANJANG RENTANGNYA menjadi 4096 sampel
    # (Bisa diganti jadi 8192 jika ingin lebih detail, tapi loading akan sangat lama!)
    jumlah_sampel = 4096 
    titik_akhir = titik_awal + jumlah_sampel
    
    # Potong array data audio
    data = data[titik_awal:titik_akhir] 
    
    # Normalisasi amplitudo DILAKUKAN SETELAH DIPOTONG
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    else:
        print("Peringatan: Potongan audio hening sama sekali!")
        
    return lajusampling, data

# =====================================================================
# 2. FUNGSI PERHITUNGAN MATEMATIS DFT MANUAL (TanPA LIBRARY FFT)
# Mengubah domain waktu ke domain frekuensi sesuai rumus di laporan:
# X_k = sum( x_n * e^(-i * 2pi * k * n / N) )
# =====================================================================
def hitung_dft_manual(data, lajusampling):
    N = len(data)
    data_fft = np.zeros(N, dtype=complex)
    frekuensi_fft = np.zeros(N)
    
    print("Sedang menghitung DFT manual secara matematis, mohon tunggu beberapa detik...")
    
    # Looping untuk mencari setiap indeks frekuensi (k)
    for k in range(N):
        X_k = 0j # 0j adalah bilangan kompleks 0 + 0i
        
        # Looping deret penjumlahan dari n=0 sampai N-1
        for n in range(N):
            # Rumus Eksponensial Euler: e^(-i * 2pi * k * n / N)
            sudut = -2j * cmath.pi * k * n / N
            eksponensial = cmath.exp(sudut)
            
            # x_n dikali eksponensial
            X_k += data[n] * eksponensial
            
        data_fft[k] = X_k
        
        # Rumus manual untuk mencari nilai Hertz-nya
        frekuensi_fft[k] = k * lajusampling / N
        
    # Menghitung magnitudo (nilai mutlak dari bilangan kompleks) dibagi N
    magnitudo = np.abs(data_fft) / N
    
    # Mengembalikan setengah bagian positif dari spektrum
    return frekuensi_fft[:N // 2], magnitudo[:N // 2]
# =====================================================================

# 3. Fungsi untuk mencari titik puncak frekuensi
def carifrekuensidominan(frekuensi, magnitudo):
    indeks = np.argmax(magnitudo)
    return frekuensi[indeks]

# 4. Fungsi untuk klasifikasi berdasarkan rentang frekuensi
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

# 5. Program Utama
def prosesaudio(namafile):
    print(f"--- Memproses File: {namafile} ---")
    lajusampling, data = bacaaudio(namafile)
    
    # Panggil fungsi perhitungan DFT manual yang sudah dibuat
    frekuensi, magnitudo = hitung_dft_manual(data, lajusampling)
    
    # Identifikasi
    frekuensidominan = carifrekuensidominan(frekuensi, magnitudo)
    kategori = klasifikasikanfrekuensi(frekuensidominan)
    
    print(f"Frekuensi Dominan : {frekuensidominan:.2f} Hz")
    print(f"Kategori Deteksi  : {kategori}\n")
    
    # Plot Grafik (Bukti untuk laporan)
    plt.figure(figsize=(10, 4))
    plt.plot(frekuensi, magnitudo)
    plt.title('Spektrum Frekuensi (DFT Manual)')
    plt.xlabel('Frekuensi [Hz]')
    plt.ylabel('Magnitudo')
    plt.xlim(0, 1000) 
    plt.grid()
    plt.show()

# Jalankan program dengan file wav kamu
# prosesaudio('rekaman_suara_saya.wav')

# Menginputkan file audio
if __name__ == "__main__":
    prosesaudio('DSP/testing_suara.wav')