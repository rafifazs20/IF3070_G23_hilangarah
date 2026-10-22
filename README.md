# IF3070 G23 - Hilangarah

## Menjalankan CLI interaktif

```bash
python -m src.main
```

Pilih satu dari empat varian Hill-Climbing, Simulated Annealing, atau Genetic Algorithm dari menu. Gunakan `--help` untuk melihat opsi konfigurasi atau `--algorithm` untuk langsung menjalankan satu pilihan.

Untuk menjalankan algoritma pada armada dari dataset bawaan, tentukan jumlah kapal, misalnya `python -m src.main --num-ships 3`. Setiap kapal memakai ukuran dan kapasitas dataset bawaan; kendaraan akan dialokasikan ke kapal mana pun yang masih memiliki ruang dan kapasitas. Dataset JSON juga dapat berisi properti `ships` berupa daftar kapal.

## Otomatisasi eksperimen dan visualisasi

```bash
python -m src.run_experiments
```

Runner menjalankan 3 run untuk setiap varian Hill-Climbing dan Simulated Annealing, lalu 18 kombinasi GA (populasi 20, 40, 60, 80, 100, 120 × generasi 50, 100, 150). Hasil tugas utama (satu kapal) disimpan di `output/experiments/single_ship/`; hasil bonus armada disimpan terpisah di `output/experiments/multi_ship/`. Masing-masing folder berisi JSON/CSV, plot riwayat eksperimen, dan gambar Initial/Final State. Gunakan `--help` untuk mengganti dataset, seed, batas iterasi, konfigurasi, dan folder keluaran. Jumlah kombinasi GA harus tetap 18.

Runner juga menerima `--num-ships 3` untuk dataset bawaan. Objective menghitung total shipping fee kendaraan yang diangkut seluruh armada, sementara batas berat dan ruang diperiksa per kapal.

Visualisasi langsung membaca dan memproses kedua JSON yang tersedia (tanpa opsi tambahan). Single ship ditampilkan sebagai satu geladak; Multi-kapal ditampilkan sebagai beberapa geladak berdampingan dalam satu gambar Initial State atau Final State. Judul gambar mencantumkan jumlah kendaraan, fee, dan berat per kapal.

```bash
python -m src.visualizer
```

Untuk file JSON lain, berikan path sebagai argumen: `python -m src.visualizer path/hasil.json`.
