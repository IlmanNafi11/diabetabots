# Laporan Evaluasi Retrieval Precision@5

**Tanggal:** 2026-10-11  
**Kegiatan:** Evaluasi retrieval iterasi 2  
**Acuan:** PRD §7.2, §7.3, §7.4; SPEC-rag-retrieval.md  
**Kriteria lolos:** rata-rata Precision@5 tidak lebih rendah dari hasil spike Iterasi 0 (0,800).

## Apa yang Diuji

Lima kueri retrieval PRD §7.3 dievaluasi terhadap koleksi pangan TKPI+GI. Untuk setiap kueri, lima nama pangan teratas dibandingkan dengan istilah relevan yang ditetapkan oleh harness evaluasi. Metrik adalah Precision@5; keputusan lolos menggunakan rata-rata lima skor.

## Bahan Pengujian

| Parameter | Nilai |
|---|---|
| Dataset | `data/tkpi-gi-final.jsonl` |
| Koleksi | `tkpi_gi` |
| Jumlah dokumen | 1.146 |
| Strategi chunking | `recursive` |
| Model embedding | `text-embedding-3-large` |
| Dimensi embedding | 3.072 |
| Top-k | 5 |
| Ambang spike | 0,800 |
| Kueri | Lima kueri PRD §7.3, verbatim |
| Bukti mentah | `reports/logs/retrieval-raw.log` |

Log mentah berisi keluaran evaluator aktual dengan timestamp, kueri, lima nama hasil, dan skor Precision@5. Rekaman terakhir untuk masing-masing kueri dipakai agar evaluasi tidak menghitung pengulangan run sebelumnya.

## Hasil Pengujian

| # | Kueri | Precision@5 |
|---:|---|---:|
| 1 | Berapa kandungan gizi nasi putih? | 0,800 |
| 2 | Gizi ayam goreng segini banyak apa? | 1,000 |
| 3 | Tahu dan tempe mengandung protein berapa? | 1,000 |
| 4 | Kalori bayam rebus berapa? | 1,000 |
| 5 | Indeks glikemik pisang ambon tinggi tidak? | 0,200 |
| **Rata-rata** |  | **0,800** |

**Keputusan:** **LOLOS** — rata-rata Precision@5 = 0,800, sama dengan ambang spike Iterasi 0. Semua lima kueri mengembalikan lima hasil.

## Reproduksibilitas dan Kondisi Lingkungan

Harness uji repositori untuk evaluasi ini adalah
`tests/test_integration/test_retrieval_5queries.py`; percobaan ulang langsung
dilakukan saat verifikasi final. Penyedia embedding yang dikonfigurasi saat ini
menolak permintaan dengan HTTP 401 (`Authentication Error, Key is blocked`)
setelah sebelumnya mengembalikan HTTP 429 (budget terlampaui). Karena itu
laporan ini memuat bukti mentah yang sudah tersimpan dan secara eksplisit
membedakan bukti tersebut dari percobaan ulang langsung yang terblokir.

## Kesimpulan

Bukti retrieval tersimpan memenuhi PRD §1.3 AC2 dan menyertakan rujukan log
mentah yang diwajibkan. Tidak ada isi log mentah yang diedit.
