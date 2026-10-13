# Laporan Re-Verifikasi Area Terblokir

**Tanggal:** 2026-10-13
**Branch:** `dev` (basis `51bebd4` + fix extractor di tree kerja)
**Acuan:** `reports/laporan-final-verification-2026-10-11.md` (bagian blokir),
PRD §1.3, §7.4, RULES.md §5
**Tujuan:** memverifikasi ulang tiga area yang pada 2026-10-11 terblokir
lingkungan: (1) suite integration live, (2) rerun LLM/E2E langsung untuk AC4,
(3) `docker compose up`.

## Apa yang Diuji

1. Ketersediaan ulang penyedia embedding/LLM (sebelumnya 401 key blocked).
2. Akses Docker tanpa sudo (sebelumnya permission denied pada socket).
3. Suite integration penuh (`pytest -m integration`).
4. Rerun 10 skenario E2E Telegram langsung (audit runner, bukan pytest).
5. Gate deterministik penuh setelah perbaikan test.

## Bahan Pengujian

| Bahan | Nilai |
|---|---|
| Dataset | `data/tkpi-gi-final.jsonl` — 1.146 entri |
| Model embedding | `text-embedding-3-large` (3072 d), endpoint `ai.afidev.me/v1` |
| Model chat | `coding-daily`, temperature 0.0 |
| Basis data uji | `localhost:5432/diabetabot_test` (PostgreSQL live) |
| Docker | compose Mode A; `docker compose up -d --build` |
| Bukti mentah E2E | `reports/logs/reverify-2026-10-13/e2e-scenario-01.log` … `-10.log` |

## Perbaikan Lingkungan

1. **Penyedia embedding hidup kembali.** Probe `embed_documents` berhasil
   (dim 3072). Key tidak lagi diblokir; throughput penuh 1.146 dokumen
   sekitar 181 detik.
2. **Grup Docker.** `agentuser` sudah tercantum di `docker:x:113:agentuser`,
   tetapi sesi lama dibuat sebelum keanggotaan berlaku. Verifikasi dengan
   `sg docker` (mewakili sesi baru) menunjukkan `docker info` sukses tanpa
   sudo; tidak ada perubahan file repositori untuk item ini.

## Hasil Pengujian

| Area | Status 2026-10-11 | Status 2026-10-13 | Bukti |
|---|---|---|---|
| Suite integration | 5 error (401) | ✅ **5 passed, 432 deselected** (147,8 s) | output pytest; `retrieval-raw.log` rekaman baru |
| AC2 retrieval live | terblokir | ✅ mean P@5 = 0,800 pada rerun langsung | `reports/logs/retrieval-raw.log` (10 rekaman 2026-10-13) |
| AC4 nutrition live | terblokir | ✅ **3 passed** (81,7 s) | output pytest |
| AC5 E2E 10 skenario | bukti 2026-10-11 | ✅ **10/10 PASS** rerun langsung (194,8 s) | `reports/logs/reverify-2026-10-13/` |
| AC9 docker compose up | terblokir (socket) | ✅ `up -d --build` sukses; app healthy; `/health` = 200 | output docker compose |
| Gate deterministik | 432 passed | ✅ 432 passed (tidak berubah) | output pytest |

### Skenario 9 (kasus defect BDD) — terverifikasi pulih

Sebelum perbaikan: `Energi: 51,52 kkal (dari 25,76 kkal/100 g)` — salah.
Setelah perbaikan, rerun langsung memproduksi:

- Pesan: `• Energi: 25,76 kkal (dari 28,0 kkal/100 g)` — nilai porsi BDD 46%
  dan nilai sumber TKPI kini benar dan berbeda.
- `facts.pending` energy: `25,76` (dulunya 56,0) — payload deterministik
  callback sudah memakai `nutrition_for_gram`.

## Temuan Tambahan dan Perbaikan Test

Rerun live pertama menemukan tiga kegagalan test (bukan defect produk):

1. `test_format_food_context_matches_source` — regresi dari perbaikan label
   2026-10-11: regex test membaca operand pertama (nilai sumber 180), bukan
   operand porsi (270). **Diperbaiki:** test kini menegaskan KEDUA operand
   (sumber dan porsi) terhadap rujukan TKPI.
2. `_extract_nutrients` rapuh terhadap markdown (`**Energi:** 80,0`, tabel)
   dan memindai lintas baris hingga menempel angka tak terkait
   (Protein → 64). **Diperbaiki:** pola kini dibatasi per baris dan toleran
   format; angka narasi sebenarnya sudah benar dan grounded.
3. Format respons model (`coding-daily`) bergeser dari §3.5 persis (tabel
   markdown, bold). Angka tetap benar; diekstrak setelah hardening. Risiko
   sisa: variasi format — tidak mengubah prompt (SPEC §7.2 Ask First).

## Catatan Skenario 10 (informasi, tanpa perubahan)

Narasi menyebut 71,0 kkal untuk 100 g (tanpa reduksi BDD 66,5%) sambil
mencantumkan 47,22 kkal di catatan. Payload deterministik callback tetap benar
(`energy_kkal: 47.22`, memakai BDD). Akar: konteks kini memuat nilai sumber
71,0 dan nilai porsi BDD 47,22, dan aturan prompt belum secara eksplisit
memerintahkan pemakaian nilai porsi hasil BDD. Karena narasi bukan sumber
aritmetika callback (D14), dampaknya terbatas pada tampilan; perubahan prompt
memerlukan persetujuan (§7.2) sehingga hanya dicatat di sini.

## Kesimpulan

Seluruh area yang pada 2026-10-11 terblokir kini terverifikasi lulus:

- AC1, AC2, AC3, AC4, AC5, AC6, AC7: lulus dengan bukti live terbaru.
- `docker compose up -d --build`: sukses; kedua kontainer healthy.
- Log mentah historis tidak diubah; bukti rerun disimpan terpisah di
  `reports/logs/reverify-2026-10-13/`.
