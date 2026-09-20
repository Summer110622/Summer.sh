# Qwen3.5 MoE tuning notes

This document explains how to adjust the presets in Summer. The numbers are practical starting points, not hardware-independent optima.

## 1. Why MoE placement matters

Qwen3.5 MoE models do not use every parameter for every token like a dense model. Routed expert weights are a large part of the model, while only a subset of experts is active for each token.

That makes the upstream `llama.cpp` options `--cpu-moe` and `--n-cpu-moe` especially useful: attention and other tensors can remain GPU-offloaded while routed expert weights are moved to system RAM.

The trade-off is bandwidth. During generation, selected expert weights still need to be read. If RAM bandwidth or CPU-GPU transfer becomes the bottleneck, excessive CPU MoE offload can reduce tokens per second.

If the model already fits comfortably in VRAM, reducing CPU MoE offload will often improve generation speed.

## 2. Why the presets use `--fit`

Recent upstream `llama.cpp` builds can automatically adjust unspecified offload settings so the model fits in available accelerator memory.

Summer normally generates:

```text
--gpu-layers auto --fit on --fit-target <MiB>
```

`--fit-target` is the amount of accelerator memory to leave free.

A target that is too small can make the process fragile when the desktop, display driver, context allocation, or another application consumes VRAM. A target that is too large can reduce GPU offload unnecessarily.

Useful starting points:

| Situation | fit-target |
|---|---:|
| Extra headroom | 1280 MiB |
| Balanced | 1024 MiB |
| Tight VRAM budget | 768 MiB |

## 3. `balanced`

```text
ctx=8192
batch=2048
ubatch=256
K/V=q8_0/q8_0
fit-target=1024 MiB
```

Use this first. It provides a useful baseline for memory use, prompt processing and generation speed while leaving MoE placement to the `llama.cpp` fit logic.

## 4. `low-vram`

```text
ctx=8192
batch=1024
ubatch=128
K/V=q4_0/q4_0
--cpu-moe
fit-target=768 MiB
```

This profile prioritizes VRAM savings. All routed expert weights remain on CPU.

If it is too slow, do not immediately abandon CPU offload. Try partial MoE offload instead:

```bash
./Summer.sh \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  --profile balanced \
  --n-cpu-moe 20
```

Windows:

```powershell
.\Summer.ps1 --bin "C:\llama.cpp\llama-cli.exe" --model "D:\models\model.gguf" --profile balanced --n-cpu-moe 20
```

Reduce `N` to move more expert layers back to the GPU.

## 5. `gpu`

```text
ctx=16384
batch=2048
ubatch=256
K/V=q8_0/q8_0
fit-target=1024 MiB
```

This is a more GPU-oriented starting point. It does **not** force every tensor onto the GPU: `--fit` remains enabled so the launcher can avoid an unnecessary out-of-memory failure.

For controlled benchmarking, use raw `llama.cpp` options after `--` to compare a fully manual placement strategy.

## 6. `long-context`

```text
ctx=32768
batch=1024
ubatch=128
K/V=q4_0/q4_0
fit-target=1280 MiB
```

This profile spends more memory on context, so it reduces KV-cache precision and micro-batch size.

Do not jump directly to the model's maximum context unless the workload actually requires it. Increase context progressively, for example:

```text
32K -> 64K -> 128K
```

At each step, check:

- model-load stability
- prompt-processing speed
- generation speed
- total RAM/VRAM use
- whether the requested context leaves enough memory for the rest of the application

## 7. KV cache

`q8_0/q8_0` is the balanced starting point.

For lower VRAM use:

```text
--ctk q4_0 --ctv q4_0
```

For higher precision:

```text
--ctk f16 --ctv f16
```

With long contexts, KV-cache type has a direct effect on memory use, so tune it together with context length instead of treating it as an independent switch.

## 8. Batch and micro-batch

`-b` is the logical batch size and `-ub` is the physical micro-batch processed at once.

For a hybrid/recurrent architecture, a micro-batch that works well on a conventional dense Transformer is not automatically optimal.

If prompt processing is unstable or runs out of memory, reduce `ubatch` first:

```text
256 -> 128 -> 64
```

If that is not enough, reduce the main batch size.

This preserves as much batching as possible while reducing peak working memory.

## 9. MTP speculative decoding

Summer can use embedded MTP self-speculation when both the GGUF and the installed `llama.cpp` build expose compatible support.

Baseline:

```bash
./Summer.sh --bin /path/to/llama-cli --model /path/to/model.gguf --mtp off
```

MTP:

```bash
./Summer.sh --bin /path/to/llama-cli --model /path/to/model.gguf --mtp on --mtp-tokens 2
```

The same comparison on Windows:

```powershell
.\Summer.ps1 --bin "C:\llama.cpp\llama-cli.exe" --model "D:\models\model.gguf" --mtp off
.\Summer.ps1 --bin "C:\llama.cpp\llama-cli.exe" --model "D:\models\model.gguf" --mtp on --mtp-tokens 2
```

MTP is not automatically faster on every machine. Heavy CPU offload can increase verification cost enough to erase the speculative-decoding gain.

Benchmark with the same:

- GGUF
- prompt
- context length
- sampling parameters
- CPU/GPU placement

Compare both tokens/s and first-token latency.

## 10. Reasoning

Summer defaults to:

```text
--reasoning auto
```

You can explicitly request:

```text
--reasoning on
--reasoning off
```

For server workloads, also account for request-level sampling and output-token limits. A reasoning workload may require a substantially different output budget from a non-reasoning workload.

## 11. Sampling

Summer deliberately does not force a sampling preset. Sampling is workload-dependent and, in server mode, is usually controlled by each API request.

CLI example:

```bash
./Summer.sh \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  -- \
  --temp 1.0 --top-p 0.95 --top-k 20
```

PowerShell:

```powershell
.\Summer.ps1 --bin "C:\llama.cpp\llama-cli.exe" --model "D:\models\model.gguf" -- --temp 1.0 --top-p 0.95 --top-k 20
```

## 12. Recommended tuning order

Change one major variable at a time.

1. Establish a baseline with `balanced`.
2. If you run out of VRAM, increase `fit-target` or try `low-vram`.
3. If generation is too slow, reduce the number of MoE layers kept on CPU.
4. If you need more context, increase context while reducing KV-cache size and/or `ubatch`.
5. Benchmark MTP only after memory placement is stable.
6. Keep the configuration that improves your actual workload, not just model-load success.

If `llama-bench` is available, measure prompt processing and token generation separately. They respond differently to batch size, offload, memory bandwidth and speculative decoding.
