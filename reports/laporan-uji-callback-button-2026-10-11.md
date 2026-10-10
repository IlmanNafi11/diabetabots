# Laporan Uji Callback Button Konfirmasi Konsumsi

**Tanggal:** 2026-10-11  
**Kegiatan:** Uji callback inline button Ya/Tidak  
**Acuan:** PRD §1.3 AC7, §3.3 langkah 4, §6 D14, §7.4; SPEC-energy-tracking.md; SPEC-telegram-integration.md  
**Kriteria lolos:** menekan **Ya** mengurangi sisa energi tepat sekali (idempoten terhadap penekanan ganda); menekan **Tidak** tidak mengubah sisa energi; callback kedaluwarsa/asing ditolak tanpa mutasi.

## Apa yang Diuji

Perilaku konfirmasi konsumsi melalui inline button Telegram pada pesan analisis
("Catat sebagai konsumsi?" → [Ya] [Tidak]):

1. **Ya** mengurangi `sisa_energi_kkal` tepat sekali, menulis baris `consumption_log`, dan menandai `pending_analyses.resolved`.
2. Penekanan **Ya** ganda (double tap) tidak mengurangi dua kali — status kedua `ALREADY_RESOLVED`.
3. **Tidak** hanya menandai baris `resolved` tanpa mengubah saldo dan tanpa baris `consumption_log`.
4. Callback lintas hari (basi) → `STALE`; `pending_id` tak dikenal/milik user lain → `NOT_FOUND`; keduanya tanpa mutasi.
5. Tombol dibersihkan (keyboard dihapus) setelah hasil dikirim.
6. Saldo boleh negatif dengan peringatan eksplisit.
7. Reset tengah malam WIB berjalan idempoten via job APScheduler pada event loop async nyata.

## Bahan Pengujian

| Parameter | Nilai |
|---|---|
| Runner | pytest 9.1.1, Python 3.11.16 |
| Basis data | PostgreSQL uji (`diabetabot_test`, asyncpg) |
| Perangkat uji | `tests/test_pending_confirmation.py`, `tests/test_energy_scheduler.py`, `tests/test_telegram_handlers.py` |
| Modul yang diuji | `app/services/energy_service.py`, `app/services/scheduler.py`, `app/bot/handlers.py` |
| Bukti mentah | `reports/logs/callback-button-raw.log` |

## Hasil Pengujian

| # | Perilaku yang diuji | Sumber uji | Hasil |
|---:|---|---|---|
| 1 | "Ya" mengurangi tepat sekali + catat `consumption_log` | `test_pending_confirmation.py` | ✅ Lulus |
| 2 | Double tap "Ya" → `ALREADY_RESOLVED`, saldo tetap | `test_pending_confirmation.py` | ✅ Lulus |
| 3 | "Tidak" → `DECLINED`, tanpa mutasi saldo/log | `test_pending_confirmation.py` | ✅ Lulus |
| 4 | Callback basi → `STALE`; asing/tak dikenal → `NOT_FOUND` | `test_pending_confirmation.py` | ✅ Lulus |
| 5 | Saldo negatif diberi peringatan (`warned=True`) | `test_pending_confirmation.py` | ✅ Lulus |
| 6 | Reset tengah malam WIB idempoten (job APScheduler) | `test_energy_scheduler.py` | ✅ Lulus |
| 7 | Keyboard dibersihkan setelah hasil; wiring handler | `test_telegram_handlers.py` | ✅ Lulus |

**Ringkasan: 37 uji lulus, 0 gagal** (`reports/logs/callback-button-raw.log`).

## Kesimpulan

PRD §1.3 AC7 terpenuhi: "Ya" mengurangi tepat sekali (idempoten), "Tidak" tidak
mengubah sisa energi, dan callback basi/asing ditolak tanpa mutasi. Log mentah
tersimpan sebagai bukti dan tidak diedit tangan.
