# IF3070 G23 - Hilangarah

Program optimasi pemuatan kendaraan pada geladak kapal menggunakan tiga keluarga algoritma pencarian: Hill-Climbing, Simulated Annealing, dan Genetic Algorithm. Hill-Climbing menyediakan empat varian. Bonus Multi-kapal mengalokasikan kendaraan ke beberapa kapal dan memaksimalkan total shipping fee armada dengan batas ruang dan kapasitas diperiksa pada setiap kapal.

## Persiapan

Gunakan Python 3.10 atau lebih baru. Dari folder utama repository, pasang dependensi:

```bash
python -m pip install -r requirements.txt
```

## Jalankan algoritma secara interaktif

```bash
python -m src.main
```

Pilih algoritma dari menu:

1. Steepest Ascent Hill-Climbing
2. Hill-Climbing with Sideways Move
3. Stochastic Hill-Climbing
4. Random Restart Hill-Climbing
5. Simulated Annealing
6. Genetic Algorithm

Hill-Climbing dan Simulated Annealing berjalan tiga kali secara default; Genetic Algorithm berjalan satu kali. Tekan `0` pada menu untuk keluar. Hasil akhir ditampilkan di terminal. Untuk Multi-kapal, terminal juga merangkum jumlah kendaraan, fee, dan berat muatan setiap kapal.

Contoh menjalankan algoritma tertentu langsung:

```bash
python -m src.main --algorithm random-restart
python -m src.main --algorithm genetic-algorithm --population-size 80 --generations 150
```

Lihat semua opsi, termasuk batas iterasi, seed, dan parameter algoritma:

```bash
python -m src.main --help
```

## Multi-kapal

Dataset bawaan dapat diperluas menjadi beberapa kapal dengan ukuran geladak dan kapasitas yang sama:

```bash
python -m src.main --num-ships 3
```

Algoritma mencari penempatan pada semua kapal. Setiap kendaraan hanya dapat dimuat satu kali; kendaraan yang ditempatkan pada kapal berbeda tidak berbenturan. Kapasitas berat dan batas geladak divalidasi per kapal. Objective adalah jumlah shipping fee kendaraan yang dibawa seluruh armada.

Dataset JSON khusus dapat menggunakan `ship` untuk satu kapal atau `ships` untuk daftar kapal. Setiap kapal memiliki ID, dimensi geladak, dan kapasitas. Daftar `vehicles` berisi dimensi, shipping fee, berat, dan ETA kendaraan. Contoh bentuk data:

```json
{
  "ships": [
    {"id": "Kapal-1", "Dimensions": {"w": 10, "l": 20}, "MaxCapacity": 500},
    {"id": "Kapal-2", "Dimensions": {"w": 12, "l": 18}, "MaxCapacity": 600}
  ],
  "vehicles": [
    {"id": "V01", "Dimensions": {"w": 3, "l": 4}, "ShippingFee": 60, "Weight": 50, "ETA": 3}
  ]
}
```

Jalankan dataset tersebut dengan `python -m src.main --data path/ke/dataset.json`.

## Otomatisasi eksperimen

```bash
python -m src.run_experiments
```

Runner melakukan 33 eksperimen:

- 3 run untuk setiap varian Hill-Climbing (12 run total).
- 3 run Simulated Annealing.
- 18 run Genetic Algorithm dari 6 ukuran populasi (`20, 40, 60, 80, 100, 120`) x 3 jumlah generasi (`50, 100, 150`).

Eksperimen Multi-kapal menggunakan opsi yang sama:

```bash
python -m src.run_experiments --num-ships 3
```

Hasil single ship dan Multi-kapal disimpan terpisah:

| Jenis eksperimen | Folder hasil |
| --- | --- |
| Single ship | `output/experiments/single_ship/` |
| Multi-kapal | `output/experiments/multi_ship/` |

Setiap folder berisi `experiment_results.json`, `experiment_summary.csv`, grafik eksperimen, dan gambar Initial/Final State. JSON mencakup konfigurasi, riwayat objektif, state awal/akhir, serta ringkasan muatan per kapal. Jumlah kombinasi GA harus tetap 18. Untuk mengubah konfigurasi, dataset, seed, atau folder keluaran:

```bash
python -m src.run_experiments --help
```

## Visualisasi

Setelah eksperimen yang diinginkan selesai, visualisasikan semua file hasil single ship dan Multi-kapal yang tersedia:

```bash
python -m src.visualizer
```

Single ship ditampilkan sebagai satu geladak. Multi-kapal ditampilkan dalam satu gambar dengan geladak armada berdampingan. Tiap geladak mencantumkan jumlah kendaraan, shipping fee, dan berat muatan. Grafik riwayat eksperimen dan gambar grid disimpan di subfolder `plots/` pada folder hasil masing-masing.

Untuk visualisasi satu file JSON saja:

```bash
python -m src.visualizer output/experiments/multi_ship/experiment_results.json
python -m src.visualizer output/experiments/single_ship/experiment_results.json
```

## Modul utama

| File | Kegunaan |
| --- | --- |
| `src/main.py` | CLI interaktif untuk memilih dan menjalankan algoritma. |
| `src/models.py` | Model kapal, kendaraan, state, validasi, dan objective armada. |
| `src/run_experiments.py` | Runner 33 eksperimen dan ekspor JSON/CSV. |
| `src/visualizer.py` | Visualisasi grid kapal dan grafik dari JSON eksperimen. |
| `src/dataset.py` | Dataset bawaan serta baca/tulis dataset JSON. |
