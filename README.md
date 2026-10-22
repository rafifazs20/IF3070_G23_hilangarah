# IF3070 G23 - Hilangarah

## Menjalankan CLI interaktif

```bash
python -m src.main
```

Pilih satu dari empat varian Hill-Climbing, Simulated Annealing, atau Genetic Algorithm dari menu. Gunakan `--help` untuk melihat opsi konfigurasi atau `--algorithm` untuk langsung menjalankan satu pilihan.

## Otomatisasi eksperimen dan visualisasi

```bash
python -m src.run_experiments
```

Runner menjalankan 3 run untuk setiap varian Hill-Climbing dan Simulated Annealing, lalu 18 kombinasi GA (populasi 20, 40, 60, 80, 100, 120 × generasi 50, 100, 150). Hasil JSON/CSV, plot riwayat eksperimen, dan gambar Initial/Final State disimpan di `output/experiments/`. Gunakan `--help` untuk mengganti dataset, seed, batas iterasi, konfigurasi, dan folder keluaran. Jumlah kombinasi GA harus tetap 18.

Visualisasi langsung membaca JSON hasil runner (tanpa opsi tambahan):

```bash
python -m src.visualizer
```

Untuk file JSON lain, berikan path sebagai argumen: `python -m src.visualizer path/hasil.json`.
