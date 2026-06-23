import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

# Membaca file audio
def bacaaudio(namafile):
    lajusampling, data = wavfile.read(namafile)
    if data.ndim > 1:
        data = data[:, 0]
    return lajusampling, data

# Memotong audio menjadi 3 bagian (Awal, Tengah, Akhir)
def ekstrak_tiga_bagian(data, ukuran_jendela=4096):
    N = len(data)
    frames = []
    
    # Kita ambil titik tengah di 25% (Awal), 50% (Tengah), dan 75% (Akhir) durasi audio
    titik_pusat = [int(N * 0.25), int(N * 0.50), int(N * 0.75)]
    
    for titik in titik_pusat:
        awal = titik - (ukuran_jendela // 2)
        akhir = titik + (ukuran_jendela // 2)
        
        # Mencegah error jika audio terlalu pendek
        awal = max(0, awal)
        akhir = min(N, akhir)
        
        frame_data = data[awal:akhir].astype(float) # Pastikan tipe datanya float
        
        # Normalisasi amplitudo khusus untuk potongan ini saja
        max_val = np.max(np.abs(frame_data))
        if max_val > 0:
            frame_data = frame_data / max_val
            
        frames.append(frame_data)
        
    return frames

# Fungsi untuk menerapkan Discerete Fourier Transform (DFT) (Partial DFT 0-1000 Hz)
# def hitungdft(data, lajusampling):
#     N = len(data)
#     batas_frekuensi_hz = 1000
#     K_maks = int(batas_frekuensi_hz * N / lajusampling) + 1
    
#     data_dft = np.zeros(K_maks, dtype=complex)
#     frekuensi_dft = np.zeros(K_maks)
#     n = np.arange(N)
    
#     for k in range(K_maks):
#         eksponensial = np.exp(-2j * np.pi * k * n / N)
#         data_dft[k] = np.sum(data * eksponensial)
#         frekuensi_dft[k] = k * lajusampling / N
        
#     magnitudo = np.abs(data_dft) / N
#     return frekuensi_dft, magnitudo

# Fungsi untuk menerapkan Fast Fourier Transform (FFT)
def hitungfft(data, lajusampling):
    N = len(data)
    datafft = np.fft.fft(data)
    frekuensifft = np.fft.fftfreq(N, d=1/lajusampling)
    magnitudo = np.abs(datafft) / N
    # Mengembalikan setengah bagian positif dari spektrum
    return frekuensifft[:N // 2], magnitudo[:N // 2]

# Fungsi mencari frekuensi dominan (Mendeteksi Fundamental Frequency / F0)
def carifrekuensidominan(frekuensi, magnitudo):
    # 1. Filter rentang suara manusia (80 Hz - 450 Hz)
    rentang_valid = np.where((frekuensi >= 80) & (frekuensi <= 450))[0]
    frek_valid = frekuensi[rentang_valid]
    mag_valid = magnitudo[rentang_valid]
    
    # 2. Gunakan find_peaks untuk mencari SEMUA bukit (peaks) pada grafik
    # Syarat: bukit harus memiliki ketinggian minimal 30% dari puncak tertinggi di rentang itu
    batas_ketinggian = np.max(mag_valid) * 0.3
    peaks, _ = find_peaks(mag_valid, height=batas_ketinggian)
    
    # 3. Ambil bukit PERTAMA (frekuensi terendah) sebagai Suara Asli (F0)
    if len(peaks) > 0:
        indeks_f0 = peaks[0]
        return frek_valid[indeks_f0]
    else:
        # Jika gagal mendeteksi bukit, kembali ke cara lama
        indeks_puncak = np.argmax(mag_valid)
        return frek_valid[indeks_puncak]

# Fungsi untuk klasifikasi
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

# Program Utama
def prosesaudio(namafile):
    print(f"=== Memproses File: {namafile} ===")
    lajusampling, data = bacaaudio(namafile)
    
    # Membagi audio menjadi 3 bagian berdasarkan ide Anda
    frames = ekstrak_tiga_bagian(data, ukuran_jendela=4096)
    
    frekuensi_dominan_list = []
    grafik_frek = []
    grafik_mag = []
    nama_bagian = ["Bagian Awal (25%)", "Bagian Tengah (50%)", "Bagian Akhir (75%)"]
    
    # Melakukan perulangan (loop) perhitungan DFT untuk 3 bagian tersebut
    for i, frame in enumerate(frames):
        print(f"Menghitung DFT untuk {nama_bagian[i]}...")
        frek, mag = hitungfft(frame, lajusampling)
        dom_frek = carifrekuensidominan(frek, mag)
        
        frekuensi_dominan_list.append(dom_frek)
        grafik_frek.append(frek)
        grafik_mag.append(mag)
        
        print(f"-> Frekuensi Dominan: {dom_frek:.2f} Hz\n")
    
    # MENGAMBIL KESIMPULAN AKHIR (Rata-rata dari 3 bagian)
    rata_rata_frekuensi = np.mean(frekuensi_dominan_list)
    kategori_akhir = klasifikasikanfrekuensi(rata_rata_frekuensi)
    
    print("==================================================")
    print(f"RATA-RATA FREKUENSI DOMINAN : {rata_rata_frekuensi:.2f} Hz")
    print(f"KESIMPULAN KATEGORI SUARA   : {kategori_akhir}")
    print("==================================================\n")
    
   # VISUALISASI: Membuat 3 Grafik bersusun ke bawah
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    for i in range(3):
        axs[i].plot(grafik_frek[i], grafik_mag[i], color='tab:blue')
        axs[i].set_title(f'Spektrum Frekuensi - {nama_bagian[i]}\n(Puncak: {frekuensi_dominan_list[i]:.2f} Hz)', fontsize=12, fontweight='bold')
        axs[i].set_xlabel('Frekuensi [Hz]')
        axs[i].set_ylabel('Magnitudo')
        axs[i].set_xlim(0, 1000)
        axs[i].grid(True)
        
    plt.subplots_adjust(hspace=0.8) 
    plt.tight_layout() 
    plt.show()

# Jalankan Program
if __name__ == "__main__":
    prosesaudio('DSP/Voice_Recognition/game_over.wav')