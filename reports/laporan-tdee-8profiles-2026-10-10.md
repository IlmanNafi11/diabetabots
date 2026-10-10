# Laporan Uji TDEE 8 Profil Simulasi (TDD)

**Tanggal:** 2026-10-10 · **Plan ref:** task-11 · **SPEC:** SPEC-tdee-calculation.md
**Acuan PRD:** §4 (8 profil simulasi), §1.3 AC3 (selisih TDEE ≤ 0,5%)

## Apa yang Diuji

Verifikasi akurasi kalkulasi BMR/TDEE (Mifflin-St Jeor + PAL) untuk 8 profil simulasi PRD §4 melalui pipeline lengkap registrasi:

```
conversation state machine → RegistrationService → tdee_service (BMR/TDEE)
    → User row (upsert PostgreSQL) → verifikasi TDEE ≤ 0,5% dari acuan
```

**Metrik:** selisih relatif TDEE sistem vs TDEE acuan PRD.
**Kriteria lolos:** selisih ≤ 0,5% untuk seluruh 8 profil (PRD §1.3 AC3).

## Bahan Pengujian

- **Dataset referensi:** tabel 8 profil simulasi PRD §4 (acuan manual Mifflin-St Jeor).
- **Konfigurasi:** formula BMR = 10·BB + 6,25·TB − 5·usia + 5 [L] / −161 [P]; TDEE = round(BMR × PAL, 2); PAL: Sedentari 1,2 · Ringan 1,375 · Sedang 1,55 · Berat 1,725 · SangatBerat 1,9.
- **Basis kode:** commit `ebeabab` (task-10) + task-11 (RegistrasiService baru).
- **Modul yang diuji:**
  - `app/services/tdee_service.py` (matematika murni, tanpa LLM)
  - `app/services/registration_service.py` (jembatan conversation → DB, upsert idempoten)
  - `app/bot/conversation.py` (state machine registrasi 5 langkah)
- **Test runner:** pytest 9.1.1, Python 3.11.16, SQLAlchemy async + PostgreSQL test database.
- **Bukti mentah:** `reports/logs/tdee-raw.log` (output pytest tidak diedit tangan).

## Hasil Pengujian

### Tabel akurasi TDEE per profil

| ID | Gender | Usia | BB (kg) | TB (cm) | Aktivitas | PAL | BMR (kkal) | TDEE Sistem | TDEE Acuan | Selisih (%) |
|----|--------|------|---------|---------|-----------|-----|-----------|-------------|------------|-------------|
| P1 | Laki-laki | 30 | 70 | 170 | Ringan | 1,375 | 1.617,50 | 2.224,06 | 2.224,06 | 0,0000% |
| P2 | Perempuan | 28 | 60 | 160 | Ringan | 1,375 | 1.299,00 | 1.786,12 | 1.786,13 | 0,0006% |
| P3 | Laki-laki | 65 | 95 | 170 | Sedentari | 1,2 | 1.692,50 | 2.031,00 | 2.031,00 | 0,0000% |
| P4 | Perempuan | 63 | 85 | 155 | Sedentari | 1,2 | 1.342,75 | 1.611,30 | 1.611,30 | 0,0000% |
| P5 | Laki-laki | 32 | 100 | 175 | Sedentari | 1,2 | 1.938,75 | 2.326,50 | 2.326,50 | 0,0000% |
| P6 | Perempuan | 30 | 90 | 158 | Sedentari | 1,2 | 1.576,50 | 1.891,80 | 1.891,80 | 0,0000% |
| P7 | Laki-laki | 60 | 68 | 168 | Sedang | 1,55 | 1.435,00 | 2.224,25 | 2.224,25 | 0,0000% |
| P8 | Perempuan | 58 | 58 | 152 | Sedang | 1,55 | 1.079,00 | 1.672,45 | 1.672,45 | 0,0000% |

**Selisih maksimum: 0,0006%** (P2, pembulatan acuan PRD 2 desimal) — jauh di bawah ambang 0,5%. **Kesimpulan: LOLOS** untuk seluruh 8 profil.

### Hasil pipeline registrasi (persistensi)

Untuk setiap profil, tes integrasi juga memverifikasi:

1. **Persistensi User row** — `register_profile` menulis baris `users` dengan `jenis_kelamin`, `usia_tahun`, `berat_kg`, `tinggi_cm`, `tingkat_aktivitas`, `pal`, `bmr_kkal`, `tdee_kkal` yang sesuai (25/25 tes lolos).
2. **Upsert idempoten** — pemanggilan `register_profile` dua kali untuk `telegram_id` sama menghasilkan tepat **1** baris (tidak ada duplikat).
3. **Pipeline conversation → TDEE** — simulasi alur Telegram penuh (`/start` → AWAIT_GENDER → AWAIT_AGE → AWAIT_WEIGHT → AWAIT_HEIGHT → AWAIT_ACTIVITY → CONFIRMED → persist) menghasilkan TDEE ≤ 0,5% acuan.

### Ringkasan pytest

```
25 passed in 1.26s
- test_tdee_within_0_5_percent[P1..P8]     8/8 PASSED
- test_registration_persists_user_row[...] 8/8 PASSED
- test_registration_upsert_idempotent[...] 8/8 PASSED
- test_conversation_to_tdee_pipeline       1/1 PASSED
```

Bukti lengkap (output pytest mentah, tidak diedit): `reports/logs/tdee-raw.log`.

## Kesimpulan & Keputusan

- **AC PRD §1.3 #3 terpenuhi:** selisih TDEE ≤ 0,5% pada 8 profil uji (tercapai 0,0006% maksimum).
- Formula dan tabel PAL **tidak diubah**; implementasi cocok persis dengan SPEC.md §5.3.
- Selisih 0,0006% pada P2 murni efek pembulatan acuan PRD ke 2 desimal; sistem mengikuti aturan pembulatan yang sama (`round(BMR × PAL, 2)`).
- TDEE dihitung penuh **tanpa defisit/surplus** sesuai keputusan D1 PRD.
- Sisa energi harian (`DailyEnergy`) akan di-seed dari `users.tdee_kkal` — modul ini menjadi basis `energy-tracking` (task-13).
