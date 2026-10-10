# Laporan Verifikasi Final dan Kesiapan Thesis

**Tanggal:** 2026-10-11  
**Branch:** `dev`  
**Acuan:** PRD §1.3, §7.4; SPEC.md §6–§8; RULES.md §5  
**Kriteria:** tujuh acceptance criteria PRD §1.3 diverifikasi dengan bukti aktual
tersimpan. Status dibedakan antara bukti yang berhasil diamati langsung, bukti
historis tersimpan, dan verifikasi yang terblokir oleh lingkungan.

## Apa yang Diuji

1. jumlah entri TKPI+GI pada koleksi ChromaDB;
2. Precision@5 retrieval terhadap lima kueri PRD §7.3;
3. akurasi TDEE pada delapan profil simulasi;
4. akurasi nutrisi terhadap nilai rujukan TKPI;
5. sepuluh skenario Telegram end-to-end;
6. kelengkapan artefak audit §7.4; dan
7. idempotensi callback konsumsi **Ya** serta no-op callback **Tidak**.

Verifikasi juga menjalankan lint, format check, build/compile check, unit-test
gate, dan pengujian callback yang relevan. Log mentah lama tidak diubah.

## Bahan Pengujian

| Bahan | Nilai |
|---|---|
| Dataset | `data/tkpi-gi-final.jsonl` — 1.146 entri |
| Koleksi | ChromaDB `tkpi_gi` — persistent `./chroma_db/` |
| Retrieval | recursive × `text-embedding-3-large`, cosine, top-k 5 |
| Basis data uji | PostgreSQL `localhost:5432/diabetabot_test` |
| Test runner | pytest 9.1.1, Python 3.11 |
| Gate deterministik | Ruff, format check, `compileall`, pytest non-integration |
| Gate callback | `tests/test_pending_confirmation.py`, `tests/test_energy_scheduler.py`, `tests/test_telegram_handlers.py` |
| Deployment check | `docker compose config --quiet`; cek daemon Docker langsung |

## Hasil Pengujian per Acceptance Criterion

| AC | Kriteria | Status | Bukti |
|---:|---|---|---|
| 1 | Seluruh 1.146 entri pangan terindeks | ✅ **LULUS** | `tkpi_gi.count()` live = 1.146; `reports/logs/indexing-raw.log` mencatat `document_count: 1146`. |
| 2 | Mean Precision@5 ≥ ambang spike 0,800 | ✅ **LULUS — bukti tersimpan** | `reports/laporan-evaluasi-retrieval-2026-10-11.md`; lima skor terakhir = 0,800; 1,000; 1,000; 1,000; 0,200; mean = 0,800; bukti mentah di `reports/logs/retrieval-raw.log`. Rerun langsung terblokir provider. |
| 3 | Selisih TDEE ≤0,5% pada 8 profil | ✅ **LULUS** | `tests/test_integration/test_tdee_8profiles.py`: 25 passed; selisih maksimum 0,0006%; `reports/laporan-tdee-8profiles-2026-10-10.md`. |
| 4 | Selisih nutrisi dilaporkan ≤5% dari TKPI | ⚠️ **JALUR DETERMINISTIK LULUS; E2E LANGSUNG TERBLOKIR** | Defect grounding deterministik (BDD/porsi) ditemukan dan diperbaiki dengan TDD; `tests/test_llm_service.py` memverifikasi konteks sumber vs porsi. Log E2E historis memuat defect sebelum perbaikan (skenario 09 melaporkan 51,52 vs TKPI 56,00, sekitar 8%). Rerun LLM langsung terblokir provider 401/429. Lihat bagian "Detail AC4". |
| 5 | 10 skenario Telegram tanpa error, format konsisten | ✅ **LULUS — bukti tersimpan** | `reports/laporan-e2e-10-scenarios-2026-10-11.md`; log mentah `e2e-scenario-01.log` … `e2e-scenario-10.log`; laporan mencatat 10/10. |
| 6 | Artefak raw log + laporan markdown lengkap | ✅ **LULUS** | Laporan wajib kini tersedia: spike, retrieval, TDEE, E2E 10 skenario, callback button, dan verifikasi final ini; log mentah terkait dirujuk di bawah. |
| 7 | **Ya** mengurangi tepat sekali; **Tidak** tidak mengubah saldo | ✅ **LULUS** | `reports/laporan-uji-callback-button-2026-10-11.md`; bukti mentah `reports/logs/callback-button-raw.log`; 37 uji callback/scheduler/handler lulus. |

## Detail AC4 dan Perbaikan yang Diterapkan

### Temuan

`format_food_context()` sebelumnya menuliskan nilai porsi hasil penyesuaian BDD
pada kedua sisi fragmen konteks berikut:

```text
Energi 25.76 kkal/100g→25.76 kkal untuk 200g
```

Untuk Semangka (TKPI 28,0 kkal/100 g, BDD 46%, 200 g), hal ini membuat LLM
memperlakukan 25,76 sebagai nilai sumber per 100 g lalu melaporkan 51,52 kkal.
Nilai itu sekitar 8% dari rujukan TKPI 56,00 kkal untuk 200 g, melampaui
toleransi 5% AC4. Log `e2e-scenario-09.log` yang tersimpan mendahului perbaikan
dan dipertahankan apa adanya sebagai bukti mentah historis.

### Perbaikan TDD

Uji regresi ditambahkan lebih dulu dan gagal. Implementasi kini menuliskan:

```text
Energi 28.0 kkal/100g→25.76 kkal untuk 200g
```

Nilai kiri adalah nilai sumber TKPI; nilai kanan adalah nilai porsi hasil
penyesuaian BDD. Payload deterministik callback di handler sudah memakai helper
kanonik `nutrition_for_gram()` dan tidak diubah oleh perbaikan ini.

### Batas verifikasi saat ini

Uji regresi dan uji deterministik LLM-service lulus. Rerun LLM/E2E langsung
tidak dapat diselesaikan karena penyedia embedding yang dikonfigurasi menolak
permintaan dengan HTTP 401 (`Authentication Error, Key is blocked`) dan
sebelumnya HTTP 429 (`budget exceeded`). Karena itu AC4 tidak ditandai sebagai
kelulusan end-to-end langsung di lingkungan ini.

## Daftar Artefak

| Kegiatan | Bukti mentah | Laporan terbaca |
|---|---|---|
| Spike Iterasi 0 | `reports/logs/spike-raw.log` | `reports/laporan-spike-2026-10-10.md` |
| Retrieval iterasi 2 | `reports/logs/retrieval-raw.log` | `reports/laporan-evaluasi-retrieval-2026-10-11.md` |
| TDEE 8 profil | `reports/logs/tdee-raw.log` | `reports/laporan-tdee-8profiles-2026-10-10.md` |
| Callback button | `reports/logs/callback-button-raw.log` | `reports/laporan-uji-callback-button-2026-10-11.md` |
| E2E 10 skenario | `reports/logs/e2e-scenario-01.log` … `-10.log` | `reports/laporan-e2e-10-scenarios-2026-10-11.md` |
| Verifikasi final | keluaran gate dan bukti terkait | laporan ini |

## Gate Verifikasi

Pemeriksaan berikut lulus setelah perbaikan konteks prompt:

- Lint Ruff: `ruff check app tests` — seluruh pemeriksaan lulus.
- Format: `ruff format --check app tests` — seluruh file terformat.
- Build/compile: `.venv/bin/python -m compileall -q app tests scripts` — lulus.
- Uji deterministik: `pytest -q -m 'not integration'` — 432 passed, 5 deselected.
- Uji TDEE terisolasi: 25 passed.
- Uji callback/scheduler/handler: 37 passed.

Suite penuh penanda integration dijalankan dan berakhir dengan 5 galat fixture
saat proses embedding, seluruhnya disebabkan respons 401 key-blocked dari
penyedia eksternal. Validasi konfigurasi Docker lulus
(`docker compose config --quiet`), namun `docker info` gagal karena pengguna
saat ini tidak dapat mengakses `/var/run/docker.sock`; `docker compose up`
tidak dapat dijalankan.

## Kesimpulan dan Status Thesis

- Semua gate deterministik (lint, format, build, unit, callback) lulus setelah
  perbaikan.
- Artefak audit yang diwajibkan PRD §7.4 lengkap dan tersimpan.
- AC1, AC2, AC3, AC5, AC6, dan AC7 memiliki bukti kelulusan.
- AC4: defect grounding sumber deterministik ditemukan, diperbaiki dengan TDD,
  dan terverifikasi lokal; konfirmasi LLM/E2E langsung masih terblokir oleh
  kredensial/anggaran eksternal.
- File docker compose valid secara sintaksis; startup runtime belum terverifikasi
  karena izin socket Docker.
- Tidak ada bukti mentah yang diedit, dan tidak ada kredensial atau rahasia yang
  ditambahkan.
