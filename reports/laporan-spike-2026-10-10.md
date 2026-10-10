# Laporan Spike Embedding x Chunking — 2026-10-10

## Tujuan

Membandingkan grid 3x2 strategi representasi dokumen dan model embedding dengan metrik
Precision@5 pada lima kueri PRD §7.3. Setiap entri pangan tetap satu dokumen mandiri.

## Bahan dan Konfigurasi

- Dataset: `data/tkpi-gi-final.jsonl` (1146 entri)
- Strategi: fixed, recursive, late
- Model: text-embedding-3-large (3072), text-embedding-3-small (1536)
- Kueri: 5 kueri PRD §7.3
- Bukti mentah: `reports/logs/spike-raw.log`

## Hasil

| Strategi | Model | Dimensi | Rata-rata P@5 | Latensi kueri rerata (ms) |
|---|---|---:|---:|---:|
| fixed | `text-embedding-3-large` | 3072 | 0.720 | 500.93 |
| fixed | `text-embedding-3-small` | 1536 | 0.600 | 369.63 |
| recursive | `text-embedding-3-large` | 3072 | 0.800 | 581.96 |
| recursive | `text-embedding-3-small` | 1536 | 0.680 | 1041.89 |
| late | `text-embedding-3-large` | 3072 | 0.600 | 726.37 |
| late | `text-embedding-3-small` | 1536 | 0.600 | 489.33 |

## Keputusan

Pemenang: **recursive x text-embedding-3-large** dengan rata-rata P@5
**0.800**. Jika selisih skor tertinggi kurang dari 0,02,
model small dipilih sesuai PRD §7.2 untuk biaya dan latensi lebih rendah.

Konfigurasi ini mengikat pengembangan berikutnya dan telah ditulis ke `.env`.
