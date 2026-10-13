# Laporan Evaluasi E2E 10 Skenario Percakapan Telegram

**Tanggal:** 2026-10-13 · **Plan ref:** task-15 · **SPEC:** SPEC.md §8 (telegram-integration)

## Apa yang Diuji

Menjalankan 10 skenario percakapan Telegram end-to-end (5 kategori pangan, kondisi tersedia/tidak tersedia) melalui handler produksi (`TelegramUpdateHandler`), dengan basis data PostgreSQL nyata, koleksi ChromaDB hasil indeks TKPI+GI, dan LLM nyata sesuai konfigurasi `.env`.

**Kriteria lolos:** seluruh 10 skenario berjalan tanpa galat, dengan perilaku terverifikasi (klarifikasi gram, konfirmasi Ya idempoten, jalur tanpa IG, pangan tidak tersedia, GL aktual).

## Bahan Pengujian

| Parameter | Nilai |
|---|---|
| Basis data | localhost:5432/diabetabot_test |
| Koleksi ChromaDB | tkpi_gi |
| Jumlah dokumen terindeks | 1146 |
| Model embedding | text-embedding-3-large |
| Dimensi embedding | 3072 |
| Model chat (LLM) | coding-daily |
| Strategi chunking | recursive |
| RAG top-k | 5 |
| RAG kandidat (name-grounding) | 20 |

- **Bukti mentah (tidak diedit):** `reports/logs/e2e-scenario-01.log` … `reports/logs/e2e-scenario-10.log`.

## Hasil Pengujian

| # | Kategori | Profil | Skenario | Hasil | Bukti |
|---|---|---|---|---|---|
| 1 | Pokok | P1 | Tanya gizi nasi putih 150 g (tersedia, ber-GI) | ✅ Lulus | `reports/logs/e2e-scenario-01.log` |
| 2 | Pokok | P3 | Makan makaroni 1 mangkuk → klarifikasi gram → konfirmasi konsumsi | ✅ Lulus | `reports/logs/e2e-scenario-02.log` |
| 3 | Lauk hewani | P2 | Telur rebus 2 butir → klarifikasi gram per butir | ✅ Lulus | `reports/logs/e2e-scenario-03.log` |
| 4 | Lauk hewani | P4 | Ikan kembung 100 g → konfirmasi konsumsi | ✅ Lulus | `reports/logs/e2e-scenario-04.log` |
| 5 | Lauk nabati | P5 | Tahu 100 g → konfirmasi konsumsi | ✅ Lulus | `reports/logs/e2e-scenario-05.log` |
| 6 | Lauk nabati | P6 | Oncom (tersedia, tanpa IG → respons transparan) | ✅ Lulus | `reports/logs/e2e-scenario-06.log` |
| 7 | Sayuran | P7 | Kangkung 100 g (tersedia, tanpa IG → respons transparan) | ✅ Lulus | `reports/logs/e2e-scenario-07.log` |
| 8 | Sayuran | P8 | Sayur aster (tidak tersedia di TKPI → respons jelas) | ✅ Lulus | `reports/logs/e2e-scenario-08.log` |
| 9 | Buah | P1 | Semangka 200 g (tersedia & ber-GI → GL aktual dihitung) | ✅ Lulus | `reports/logs/e2e-scenario-09.log` |
| 10 | Buah | P3 | Buah naga merah (tersedia, tanpa IG, BDD 66,5%) | ✅ Lulus | `reports/logs/e2e-scenario-10.log` |

**Ringkasan: ✅ Lulus: 10/10**

### Detail pemeriksaan per skenario

**Skenario 1 — Tanya gizi nasi putih 150 g (tersedia, ber-GI)**

- PASS — registrasi profil selesai
- PASS — analisis terkirim dengan tombol Ya/Tidak
- PASS — tidak ada pesan galat

**Skenario 2 — Makan makaroni 1 mangkuk → klarifikasi gram → konfirmasi konsumsi**

- PASS — registrasi profil selesai
- PASS — klarifikasi gram diminta
- PASS — analisis terkirim dengan tombol Ya/Tidak
- PASS — tombol Ya mengurangi tepat sekali (idempoten)
- PASS — tidak ada pesan galat

**Skenario 3 — Telur rebus 2 butir → klarifikasi gram per butir**

- PASS — registrasi profil selesai
- PASS — klarifikasi gram diminta
- PASS — analisis terkirim dengan tombol Ya/Tidak
- PASS — tidak ada pesan galat

**Skenario 4 — Ikan kembung 100 g → konfirmasi konsumsi**

- PASS — registrasi profil selesai
- PASS — analisis terkirim dengan tombol Ya/Tidak
- PASS — pangan teratas sesuai kata kunci skenario
- PASS — tombol Ya mengurangi tepat sekali (idempoten)
- PASS — tidak ada pesan galat

**Skenario 5 — Tahu 100 g → konfirmasi konsumsi**

- PASS — registrasi profil selesai
- PASS — analisis terkirim dengan tombol Ya/Tidak
- PASS — tombol Ya mengurangi tepat sekali (idempoten)
- PASS — tidak ada pesan galat

**Skenario 6 — Oncom (tersedia, tanpa IG → respons transparan)**

- PASS — registrasi profil selesai
- PASS — analisis terkirim dengan tombol Ya/Tidak
- PASS — sumber teratas tanpa data IG (jalur transparan)
- PASS — tidak ada pesan galat

**Skenario 7 — Kangkung 100 g (tersedia, tanpa IG → respons transparan)**

- PASS — registrasi profil selesai
- PASS — analisis terkirim dengan tombol Ya/Tidak
- PASS — sumber teratas tanpa data IG (jalur transparan)
- PASS — tidak ada pesan galat

**Skenario 8 — Sayur aster (tidak tersedia di TKPI → respons jelas)**

- PASS — registrasi profil selesai
- PASS — respons 'tidak tersedia' terkirim
- PASS — tidak ada pesan galat

**Skenario 9 — Semangka 200 g (tersedia & ber-GI → GL aktual dihitung)**

- PASS — registrasi profil selesai
- PASS — analisis terkirim dengan tombol Ya/Tidak
- PASS — GL aktual dihitung dari data TKPI
- PASS — tidak ada pesan galat

**Skenario 10 — Buah naga merah (tersedia, tanpa IG, BDD 66,5%)**

- PASS — registrasi profil selesai
- PASS — analisis terkirim dengan tombol Ya/Tidak
- PASS — sumber teratas tanpa data IG (jalur transparan)
- PASS — tidak ada pesan galat

## Kesimpulan & Keputusan

- **AC PRD §1.3 #5 & #6 terpenuhi:** seluruh 10 skenario berjalan tanpa galat dan artefak bukti tersimpan (log mentah per skenario + laporan ini).
