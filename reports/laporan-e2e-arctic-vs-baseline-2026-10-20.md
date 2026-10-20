# Laporan Perbandingan E2E: snowflake-arctic-embed2 vs Baseline OpenAI (2026-10-20)

## Tujuan

Membandingkan hasil 10 skenario E2E Telegram (PRD §7.3, 5 kategori, 8 profil)
menggunakan kandidat embedding self-hosted `snowflake-arctic-embed2` dengan
baseline produksi `text-embedding-3-large` (laporan 2026-10-11, 10/10 PASS).

## Bahan Pengujian

| Parameter | Arctic (eksperimen) | Baseline (produksi) |
|---|---|---|
| Koleksi ChromaDB | `tkpi_gi_arctic` | `tkpi_gi` |
| Jumlah dokumen terindeks | 1146 | 1146 |
| Model embedding | snowflake-arctic-embed2 (Ollama lokal) | text-embedding-3-large (OpenAI) |
| Dimensi embedding | 1024 | 3072 |
| Model chat (LLM) | `diabetabot` (alias) | `coding-daily` |
| Basis data | 127.0.0.1:55432/diabetabot_test | localhost:5432/diabetabot_test |
| Strategi chunking | recursive | recursive |
| RAG top-k | 5 | 5 |
| RAG kandidat (name-grounding) | 20 | 20 |

Bukti mentah arctic: `reports/logs/arctic-e2e/e2e-scenario-01.log` … `-10.log`.
Baseline: `reports/logs/e2e-scenario-01.log` … `-10.log` (2026-10-11).

## Hasil

| # | Kategori | Profil | Skenario | Arctic | Baseline |
|---|---|---|---|---|---|
| 1 | Pokok | P1 | Tanya gizi nasi putih 150 g (tersedia, ber-GI) | ✅ Lulus | ✅ Lulus |
| 2 | Pokok | P3 | Makan makaroni 1 mangkuk → klarifikasi gram → konfirmasi konsumsi | ✅ Lulus | ✅ Lulus |
| 3 | Lauk hewani | P2 | Telur rebus 2 butir → klarifikasi gram per butir | ✅ Lulus | ✅ Lulus |
| 4 | Lauk hewani | P4 | Ikan kembung 100 g → konfirmasi konsumsi | ✅ Lulus | ✅ Lulus |
| 5 | Lauk nabati | P5 | Tahu 100 g → konfirmasi konsumsi | ✅ Lulus | ✅ Lulus |
| 6 | Lauk nabati | P6 | Oncom (tersedia, tanpa IG → respons transparan) | ✅ Lulus | ✅ Lulus |
| 7 | Sayuran | P7 | Kangkung 100 g (tersedia, tanpa IG → respons transparan) | ✅ Lulus | ✅ Lulus |
| 8 | Sayuran | P8 | Sayur aster (tidak tersedia di TKPI → respons jelas) | ✅ Lulus | ✅ Lulus |
| 9 | Buah | P1 | Semangka 200 g (tersedia & ber-GI → GL aktual dihitung) | ✅ Lulus | ✅ Lulus |
| 10 | Buah | P3 | Buah naga merah (tersedia, tanpa IG, BDD 66,5%) | ✅ Lulus | ✅ Lulus |

**Ringkasan: Arctic 10/10 PASS, setara baseline 10/10 PASS.**

## Catatan Non-Determinisme

Provider LLM (`diabetabot` alias) diketahui mengembalikan `504 Gateway
Time-out` nginx pada 60 s untuk sebagian permintaan dengan prompt nyata
(dokumentasi: `reports/laporan-verifikasi-ac4-timeout-2026-10-13.md`, tingkat
kegagalan kira-kira 1/3). Pada run pertama 2026-10-20, skenario 5/6/7 gagal
karena 504; pada run final (10 skenario, dengan retry otomatis terbatas pada
skenario gagal), seluruh 10 PASS. Kegagalan 504 bukan bukti terhadap kualitas
retrieval arctic: pada setiap kasus, retrieval top-1 benar (Tahu goreng untuk
"tahu 100g", Oncom, Kangkung segar).

## Keputusan

- **E2E arctic-embed2 setara baseline:** seluruh 10 skenario PRD §7.3 PASS
  dengan `tkpi_gi_arctic` (1024 dim, biaya 0) maupun `tkpi_gi` (3072 dim,
  OpenAI).
- **Layak menggantikan OpenAI?** Data pendukung saat ini: P@5 = 0.800 (5
  kueri, menyamai baseline 0.800) + E2E 10/10 (setara baseline). Keterbatasan:
  set kueri kecil (5) dan 10 skenario tetap sampel terbatas; keputusan final
  sebaiknya disertai evaluasi kueri lebih luas sebelum cutover produksi.
- **Skrip eksperimen:** `scripts/run_e2e_arctic.py` (koleksi eksperimen
  `tkpi_gi_arctic`; produksi `tkpi_gi` tidak disentuh).