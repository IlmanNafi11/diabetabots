# Spec: embedding-selfhost-benchmark

**Status:** Draft (menunggu review) · **Worktree:** `eksperimental` · **Intent ref:**
`docs/intent/embedding-self-host-benchmark.md`

## Objective

Mengukur 4 kandidat embedding open-source yang di-self-host di host ini (2 vCPU,
~5 GB RAM free, 11 GB disk, tanpa GPU) pada lima kueri PRD §7.3 dengan metrik
Precision@5 + latensi, strategi `recursive`, dataset `data/tkpi-gi-final.jsonl`
(1146 entri). Output: tabel perbandingan vs baseline OpenAI
(`text-embedding-3-large`, P@5 = 0.800 — angka tersimpan, key mati, tidak
di-re-run) plus rekomendasi model pengganti. **Tidak mengubah konfigurasi
produksi.**

## Grid Kandidat

| # | Model | Runtime | Dimensi | Ukuran | Prefix query |
|---|---|---|---|---|---|
| 1 | Qwen3-Embedding-0.6B | fastembed ONNX (Qdrant official) | 1024 | 2.38 GB fp32 | `Instruct: …\nQuery: …` |
| 2 | Qwen3-Embedding-4B | Ollama GGUF (`qwen3-embedding:4b`) | 2560 | 2.50 GB Q4 | `Instruct: …\nQuery: …` |
| 3 | BGE-M3 | Ollama (`bge-m3:latest`) | 1024 | 1.16 GB | tidak ada (official dense usage) |
| 4 | multilingual-e5-large | fastembed ONNX (Qdrant official) | 1024 | 2.24 GB fp32 | `query: ` / `passage: ` |

> **Keputusan teknis:** BGE-M3 via Ollama, bukan fastembed — fastembed Python
> 0.9.0 tidak mendukung BGE-M3 secara native (hanya fastembed-rs); jalur custom
> ONNX (`add_custom_model`, file 2.2 GB via xet-bridge) lebih berisiko daripada
> model resmi di library Ollama. Model yang diuji tidak berubah.
> ONNX Qwen3-4B fp32 = 16 GB → tidak muat; GGUF Q4 2.5 GB adalah satu-satunya
> jalur viable (sesuai intent).

## Public API

- `app/scripts/run_selfhost_benchmark.py` (standalone,
  `uv run python -m app.scripts.run_selfhost_benchmark`):
  - `CANDIDATES` — tuple kandidat (id, label, runtime, dim, estimasi ukuran)
  - `FastembedClient` / `OllamaClient` — implementasi `EmbeddingClient`
    Protocol (`embed_documents`, `embed_query`); prefix query diterapkan di
    sini sesuai best practice model
  - CLI: `--dataset PATH`, `--reports-dir PATH`, `--candidates ID[,ID...]`
    (default: semua), `--keep-models` (default: model dihapus setelah
    evaluasi)
  - Menulis `reports/logs/selfhost-raw.log` (bukti mentah, tidak diedit) dan
    `reports/laporan-selfhost-YYYY-MM-DD.md`
- `app/evaluation/retrieval.py` — dipakai ulang (single source): `QUERY_CASES`,
  `precision_at_k`, `mean_precision_at_5`
- `app/data/parser.py` + `app/data/embedder.py` — dipakai ulang:
  `load_records`, `build_documents(records, "recursive")`
- `tests/test_selfhost_benchmark.py` — unit test bagian murni (renderer
  laporan, parsing raw log, pemilihan kandidat); **tidak** men-download model

## Dependencies

- Dataset `data/tkpi-gi-final.jsonl` (1146 — provided, read-only)
- `chromadb` (ephemeral client per eksperimen, collection per model,
  `metadata={"hnsw:space": "cosine"}`) — sudah ada di requirements
- `fastembed==0.9.0` + `onnxruntime>=1.23` — **di-install ke venv benchmark
  terpisah (`.venv-bench`), bukan ke requirements.txt produksi**
- Ollama binary (user-level, tanpa systemd/root) + model
  `qwen3-embedding:4b`, `bge-m3:latest` dari registry Ollama
- Download ONNX via hf-mirror (`HF_ENDPOINT=https://hf-mirror.com` — HF
  langsung 429 di host ini)

## Data Flow

```
tkpi-gi-final.jsonl
  → load_records → FoodRecord (1 per line)
  → build_documents(records, strategy="recursive") → IndexDocument (1146)
  → per kandidat (BERURUTAN, satu model di disk pada satu waktu):
      load model (fastembed ONNX | ollama pull + serve)
        → embed 1146 dokumen (ukur embedding_latency_ms)
        → ChromaDB ephemeral collection `selfhost_<id>` (cosine)
        → 5 kueri: embed_query (dengan prefix) → query top-5 → P@5
          (ukur latency_ms per kueri; raw log di-flush per kueri)
        → append hasil kandidat ke `selfhost-raw.log`
        → HAPUS model dari disk (fastembed cache | ollama rm)
        → kandidat berikutnya
  → render laporan final: tabel P@5 mean, latensi kueri, latensi embed,
    footprint (disk/RAM) + kolom referensi baseline OpenAI 0.800
  → rekomendasi eksplisit
```

**Hemat disk (strategi default):** satu model dievaluasi penuh → hasilnya
dicatat (raw log + append laporan) → model dihapus dari disk → kandidat
berikutnya. Maksimal satu model di disk pada satu waktu (~2.5 GB terbesar),
bukan semua kandidat sekaligus. `--keep-models` untuk inspeksi manual.

## Edge Cases

- **OOM / gagal load Qwen3-4B** → dicatat sebagai hasil `GAGAL` di raw log +
  laporan, run lanjut ke kandidat berikutnya (tidak abort seluruh grid)
- **Download gagal** (HF 429, registry timeout) → retry dengan backoff;
  gagal terus → `GAGAL` terdokumentasi
- **Prefix model** diterapkan sesuai best practice masing-masing model
  (e5: `query:`/`passage:`; bge-m3: tanpa prefix; qwen3:
  `Instruct: …\nQuery: …`); verifikasi
  dicatat di raw log
- **Disk tidak cukup** → model selesai dihapus sebelum pull berikutnya
  (default); `--keep-models` untuk inspeksi
- **Dimensi berbeda antar kandidat** (1024 vs 2560) → collection terpisah per
  model; tidak ada konflik
- **Baseline OpenAI tidak bisa di-re-run** (key mati) → angka 0.800 dari
  `reports/laporan-spike-2026-10-10.md`, dicatat sebagai referensi, bukan
  hasil baru

## Acceptance Criteria

- [ ] Empat kandidat dievaluasi (atau `GAGAL` terdokumentasi) pada 5 kueri
      PRD §7.3, strategi `recursive`, dataset 1146; setiap kandidat selesai
      dicatat lalu modelnya dihapus sebelum kandidat berikutnya dimuat
- [ ] Raw log unedited `reports/logs/selfhost-raw.log`: timestamp, model,
      kueri, retrieved names, distances, P@5, latency, prefix yang dipakai
- [ ] Laporan `reports/laporan-selfhost-YYYY-MM-DD.md`: tabel per kandidat
      (P@5 mean, latensi kueri rerata, latensi embed batch, footprint
      RAM/disk) + kolom referensi baseline OpenAI 0.800
- [ ] Rekomendasi eksplisit: kandidat dengan mean P@5 ≥ 0.800 → "layak
      menggantikan OpenAI" (dengan nomor); tidak ada → "tetap OpenAI" +
      alasan
- [ ] Unit test bagian murni pass tanpa download model
      (`uv run pytest tests/test_selfhost_benchmark.py -q`)
- [ ] Konfigurasi produksi tidak tersentuh: `.env`, koleksi `tkpi_gi`,
      `requirements.txt`, `run_spike.py` — verified via `git status`
      (hanya file benchmark baru + laporan + intent/spec)

## Commands

```bash
# Setup benchmark venv (terpisah dari runtime produksi)
uv venv .venv-bench
uv pip install --python .venv-bench fastembed==0.9.0 onnxruntime

# Install Ollama user-level (tanpa root/systemd)
curl -fsSL https://ollama.com/install.sh | sh

# Run benchmark (satu model di disk pada satu waktu, hapus otomatis setelah)
HF_ENDPOINT=https://hf-mirror.com \
  uv run python -m app.scripts.run_selfhost_benchmark \
  --dataset data/tkpi-gi-final.jsonl --reports-dir reports

# Test bagian murni (tidak butuh model)
uv run pytest tests/test_selfhost_benchmark.py -q

# Lint
ruff check app/scripts/run_selfhost_benchmark.py tests/test_selfhost_benchmark.py
```

## Boundaries

**Always:** raw log tidak diedit; laporan diturunkan dari raw log saja;
satu dokumen per entri pangan (never split); model dihapus setelah evaluasi
(kecuali `--keep-models`); prefix model didokumentasikan di raw log;
konfigurasi produksi tidak berubah.

**Ask first:** menambah kandidat di luar 4; mengubah metrik/kueri/strategi
chunking; mengubah baseline (0.800); install Ollama ke sistem
(`/usr/local`, systemd service); menyentuh `requirements.txt` produksi.

**Never:** menulis ke `.env`; re-index koleksi `tkpi_gi`; mengubah file di
`data/`; mengubah `run_spike.py`, `app/data/embedder.py`, atau
`app/evaluation/retrieval.py` (dipakai ulang apa adanya); fabricate angka;
men-commit ChromaDB persistence atau model cache.

## Open Questions

1. Versi 0.6B: fp32 (2.38 GB) vs int8 `-Q` (1.12 GB)? → default **fp32**
   (fair vs baseline, muat RAM 5 GB); int8 sebagai catatan saja.
2. Prefix handling: fastembed/Ollama mungkin menerapkan prefix otomatis
   (model card masing-masing) — verifikasi empiris di implementasi, hasil
   dicatat di raw log.
3. Ollama serve: background process user-level (port 11434) selama benchmark;
   tidak dijadikan service.

## Verifikasi

Sebelum implementasi: spec di-review dan disetujui user. Setelah implementasi:
jalankan benchmark nyata, buktikan angka, tulis laporan, `git status` bersih
dari perubahan produksi.