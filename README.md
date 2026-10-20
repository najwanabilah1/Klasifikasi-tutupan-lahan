# Klasifikasi Tutupan Lahan pada Citra Satelit

Studi kasus ini membandingkan **Random Forest** dan **Convolutional Neural
Network (CNN)** untuk mengklasifikasikan citra **Forest** versus **non-Forest**
pada dataset EuroSAT RGB.

> Klasifikasi ini mengidentifikasi kelas tutupan lahan pada satu citra. Ini
> bukan deteksi perubahan hutan dan tidak membuktikan kapan, mengapa, atau oleh
> siapa deforestasi terjadi. Analisis deforestasi membutuhkan citra
> multi-temporal dan metode untuk membandingkan perubahan antarwaktu.

## Dataset

Dataset dibaca dari folder lokal `EuroSAT_RGB`, dengan satu subfolder untuk
setiap kelas (misalnya `EuroSAT_RGB/Forest/*.jpg`). Eksperimen menggunakan versi
RGB EuroSAT yang pada rilis lengkap berisi 27.000 citra dari 10 kelas. Label
`Forest` menjadi kelas positif (1), sedangkan sembilan kelas lainnya digabung
menjadi kelas negatif (0). Format citra yang didukung: JPG, JPEG, PNG, dan TIFF;
loader mengubahnya ke RGB berukuran 64 × 64.

Sumber dataset: [EuroSAT di Zenodo](https://zenodo.org/records/7711810)
(DOI: [10.5281/zenodo.7711810](https://doi.org/10.5281/zenodo.7711810)).
Record tersebut menyediakan arsip RGB (`EuroSAT_RGB.zip`) dan multispektral;
proyek ini menggunakan versi RGB.

Data dibagi secara stratifikasi berdasarkan kelas asal menjadi train,
validation, dan test dengan proporsi 70/15/15 serta seed tetap. Validation hanya
digunakan selama pelatihan CNN; metrik akhir dihitung pada test yang sama untuk
kedua model.

Random Forest menggunakan fitur statistik mean/std tiga kanal RGB dan histogram
warna. CNN mempelajari fitur spasial langsung dari piksel citra.

## Persiapan dan menjalankan

Disarankan menggunakan lingkungan virtual Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python klasifikasi_satelit.py
```

Dataset tidak diunduh otomatis; letakkan hasil ekstraksi EuroSAT di folder
`EuroSAT_RGB` dalam direktori proyek, atau berikan lokasi foldernya melalui
`--data-dir`. Eksperimen standar melatih CNN selama lima epoch dan Random Forest
dengan 300 pohon. Opsi dapat diubah, misalnya:

```powershell
python klasifikasi_satelit.py --epochs 10 --estimators 200 --seed 42
```

Jika dataset berada di lokasi lain:

```powershell
python klasifikasi_satelit.py --data-dir "D:\datasets\EuroSAT_RGB"
```

Hasil metrik disimpan sebagai `results/metrics.json` dan grafik sebagai
`results/metrics.png` (folder ini diabaikan Git). Grafik merangkum accuracy,
macro precision/recall/F1 serta confusion matrix untuk kedua model. File JSON
mencakup metrik, ukuran tiap split, dan waktu ekstraksi fitur, pelatihan, serta
prediksi. Jalankan
eksperimen untuk memperoleh angka; repositori ini tidak mengklaim hasil yang
belum diukur. Program mencetak jumlah file per kelas dan memperingatkan jika
jumlah `Forest` bukan 3.000, jumlah yang diharapkan pada EuroSAT RGB lengkap.

Untuk membuat ulang gambar dari `metrics.json` yang sudah tersedia tanpa
mengulang pelatihan:

```powershell
python klasifikasi_satelit.py --plot-only
```

## Menjalankan tes

```powershell
python -m unittest discover -s tests -v
```

## Batasan eksperimen

- Kelas non-Forest menggabungkan beragam tutupan lahan, sehingga hasil tidak
  menjelaskan jenis atau penyebab kehilangan hutan.
- Pembagian berstrata dilakukan pada citra, bukan berdasarkan lokasi geografis.
  Citra yang berdekatan atau serupa dapat tersebar antar-split dan membuat
  evaluasi lebih optimistis. EuroSAT dalam alur ini tidak menyediakan
  pengelompokan spasial untuk mencegah kebocoran tersebut.
- Hasil hanya berlaku untuk dataset, label, pembagian data, dan pengaturan
  eksperimen ini; hasil tidak otomatis mewakili wilayah atau waktu lain.

## Struktur proyek

```text
klasifikasi-citra-satelit/
├── klasifikasi_satelit.py
├── requirements.txt
├── README.md
├── .gitignore
└── tests/
    └── test_klasifikasi_satelit.py
```

## Referensi

- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2018). [EuroSAT: A Novel
  Dataset and Deep Learning Benchmark for Land Use and Land Cover
  Classification](https://zenodo.org/records/7711810). Zenodo.
  https://doi.org/10.5281/zenodo.7711810
- [EuroSAT — TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/eurosat)
- Helber et al., “EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land
  Use and Land Cover Classification,” *IEEE Journal of Selected Topics in
  Applied Earth Observations and Remote Sensing*, 12(7), 2217–2226, 2019.
  https://doi.org/10.1109/JSTARS.2019.2918242
