# Laporan Verifikasi AC4 — Timeout Klien LLM

**Tanggal:** 2026-10-13
**Branch:** `dev`
**Acuan:** PRD §1.3 AC4, SPEC.md §6.2, §7.1, §7.4; RULES.md §5
**Tujuan:** menyelesaikan kegagalan AC4 (selisih nutrisi ≤5% dari TKPI) yang
tersisa pada verifikasi final, dengan bukti live.

## Gejala

Suite penuh (termasuk penanda `integration`) berakhir `1 failed, 437 passed`:

```
tests/test_integration/test_nutrition_accuracy.py::test_nutrition_within_5pct_of_source[nasi-putih-100g]
langchain_openai.chat_models.base.OpenAITimeoutError: Request timed out.
```

Rerun terisolasi memberi hasil identik, jadi bukan flake sesaat suite.

## Diagnosa

Pengukuran langsung ke endpoint `ai.afidev.me` (2026-10-13):

| Percobaan | Isi permintaan | Hasil |
|---|---|---|
| 1 | prompt sistem nyata (~2,3 KB, Bahasa Indonesia) | HTTP 504 nginx pada **60,0 s** |
| 2 | prompt sistem nyata | **200 OK pada 37,9 s** |
| 3 | prompt sistem nyata | **200 OK pada 34,0 s** |
| 4 | pengisi ASCII sepanjang prompt nyata | 200 OK pada 5,7 s |
| 5 | prompt nyata, panah `→` diganti `->` | 200 OK pada 41,0 s |

Temuan:

1. **Penyebab proksimal ada di sisi klien.** Dua respons sukses (34,0 s dan
   37,9 s) berada di atas `request_timeout=30.0` pada
   `app/services/llm_service.py`. Klien membatalkan permintaan yang sebenarnya
   akan dijawab penyedia — tanda tangan galat `OpenAITimeoutError` (bukan
   HTTP 504) konsisten dengan pembatalan klien.
2. **Penyebab sekunder ada di sisi penyedia.** Gateway mengembalikan
   `504` nginx tepat pada 60,0 s untuk variasi prompt nyata. Ini infrastruktur
   pihak ketiga; tidak dapat diperbaiki dari repositori ini.
3. Bukan terkait ukuran atau karakter non-ASCII: pengisi ASCII maupun prompt
   dengan `→` diganti keduanya melewati gateway.

## Perbaikan (TDD)

1. **RED** — uji konstruksi baru
   `test_llm_client_timeout_covers_provider_tail_latency` menegaskan
   `request_timeout` ChatOpenAI > 30 s. Gagal dengan `assert 30.0 > 30.0`.
2. **GREEN** — `request_timeout` dinaikkan ke `90.0` (komentar sumber mencatat
   alasan dan risiko sisa).
3. Verifikasi ulang live.

## Bukti Verifikasi Kembali

| Pemeriksaan | Hasil |
|---|---|
| `tests/test_llm_service.py` | 25 passed |
| `test_nutrition_accuracy.py` (3 uji, live) | **3 passed** (137,27 s) |
| `test_format_food_context_matches_source` (live) | lulus |
| Regresi deterministik `-m "not integration"` | **434 passed**, 5 deselected (17,09 s) |
| `ruff check app tests scripts` | All checks passed |
| `ruff format --check app tests scripts` | 66 files already formatted |
| `python -m compileall -q app tests scripts` | lulus |

Durasi suite AC4 137 s untuk 3 uji menunjukkan setidaknya satu permintaan
berjalan jauh di atas 30 s dan tetap berhasil — perilaku yang tidak mungkin
sebelum perbaikan. Ini konfirmasi langsung alasan pembatalan yang dilaporkan.

## Risiko Sisa

Gateway penyedia mengembalikan `504` pada 60 s untuk sebagian permintaan
dengan prompt nyata (1 dari 3 percobaan langsung). Kegagalan ini muncul
sebagai galat gateway, bukan lagi sebagai timeout klien, dan tidak dapat
dimigasi dari sisi repositori. Uji integrasi AC4 yang bergantung pada penyedia
karena itu tetap berpeluang gagal sesekali; tingkat kegagalannya dikuantifikasi
kasar (≈1/3) di sini, bukan disembunyikan dengan rerun sampai hijau.

## Kesimpulan

AC4 kini lulus verifikasi live melalui jalur klien nyata. Tidak ada perubahan
skema, dependensi, maupun struktur prompt pada perbaikan ini sehingga tidak
memicu aturan SPEC.md §7.2. Perubahan dapat dibalik dengan `git revert`.
