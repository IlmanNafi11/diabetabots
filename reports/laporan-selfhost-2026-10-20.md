# Laporan Benchmark Embedding Self-Host — 2026-10-20

## Tujuan

Membandingkan kandidat embedding open-source yang di-self-host di host ini
(2 vCPU, tanpa GPU) dengan baseline OpenAI pada lima kueri PRD §7.3, strategi
`recursive`, dataset `data/tkpi-gi-final.jsonl`. Ambang lolos: mean P@5 >=
0.800.

## Bahan dan Konfigurasi

- Dataset: `data/tkpi-gi-final.jsonl` (1146 entri)
- Strategi chunking: recursive (satu dokumen per entri pangan)
- Kueri: 5 kueri PRD §7.3 (verbatim, `app/evaluation/retrieval.py`)
- Top-k: 5, ruang: cosine (ChromaDB ephemeral per kandidat)
- Baseline: text-embedding-3-large (OpenAI, key blocked — stored result) — mean P@5 = 0.800,
  latensi kueri rerata ~582 ms (angka tersimpan
  `reports/laporan-spike-2026-10-10.md`; key provider diblokir, tidak di-re-run)
- Bukti mentah: `reports/logs/selfhost-raw.log`

## Hasil

| Kandidat | Runtime | Dimensi | P@5 | Latensi kueri (ms) | Latensi embed (ms) |
|---|---|---:|---:|---:|---:|
| Qwen3-Embedding-0.6B | fastembed | 1024 | 0.760 | 724.19 | 2607962.20 |
| Qwen3-Embedding-4B (GGUF Q4) | ollama | 2560 | 0.600 | 44356.19 | 16719921.83 |
| BGE-M3 | ollama | 1024 | 0.680 | 37156.22 | 3565351.87 |
| multilingual-e5-large | fastembed | 1024 | 0.760 | 144.21 | 1146184.66 |
| nomic-embed-text-v2-moe | ollama | 768 | 0.560 | 12639.88 | 1205595.74 |
| snowflake-arctic-embed2 | ollama | 1024 | 0.800 | 6288.87 | 2177698.80 |
| multilingual-e5-large-instruct | fastembed | 1024 | 0.720 | 179.47 | 980990.80 |

## Keputusan

**Rekomendasi:** snowflake-arctic-embed2 (snowflake-arctic-embed2:latest) layak menggantikan OpenAI: mean P@5 = 0.800 >= 0.800 dengan biaya 0.

Catatan per kandidat (detail di raw log): status load model, prefix yang
dipakai, dimensi aktual, dan status hapus model pasca-evaluasi.
