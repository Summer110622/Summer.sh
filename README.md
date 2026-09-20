# Summer.sh

A cross-platform Qwen3.5 MoE launcher and preset collection for upstream `llama.cpp`.

Summer reads GGUF metadata, verifies that the model uses the `qwen35moe` architecture, probes the installed `llama-cli` / `llama-server` options, and builds a practical command line around MoE CPU offload, automatic VRAM fitting, quantized KV cache, MTP speculative decoding, and reasoning controls.

No custom `llama.cpp` fork is required.

> The presets are starting points, not universal benchmark winners. Real performance depends on CPU/GPU architecture, RAM and VRAM bandwidth, model quantization, context length, and how much of the MoE model is offloaded.

## Windows: copy, paste, run

Requirements:

- Windows PowerShell
- Python 3
- a recent upstream `llama.cpp` Windows build
- a Qwen3.5 MoE GGUF

Paste this whole block into **PowerShell**:

```powershell
$dir = "$HOME\Summer-Qwen35"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Invoke-WebRequest "https://raw.githubusercontent.com/Summer110622/Summer.sh/main/Summer.ps1" -OutFile "$dir\Summer.ps1"
Invoke-WebRequest "https://raw.githubusercontent.com/Summer110622/Summer.sh/main/summer.py" -OutFile "$dir\summer.py"
Set-Location $dir
Set-ExecutionPolicy -Scope Process Bypass -Force
.\Summer.ps1
```

The script will ask for:

1. CLI or server mode
2. the path to `llama-cli.exe` or `llama-server.exe`
3. the path to the Qwen3.5 MoE GGUF
4. the preset profile

Example non-interactive PowerShell command:

```powershell
.\Summer.ps1 --bin "C:\llama.cpp\llama-cli.exe" --model "D:\models\Qwen3.5-35B-A3B-Q4_K_M.gguf" --profile balanced
```

Server mode:

```powershell
.\Summer.ps1 --mode server --bin "C:\llama.cpp\llama-server.exe" --model "D:\models\Qwen3.5-35B-A3B-Q4_K_M.gguf" --profile balanced
```

Preview the generated `llama.cpp` command without starting the model:

```powershell
.\Summer.ps1 --bin "C:\llama.cpp\llama-cli.exe" --model "D:\models\model.gguf" --dry-run
```

## Linux / macOS

```bash
git clone https://github.com/Summer110622/Summer.sh.git
cd Summer.sh

./Summer.sh \
  --bin /path/to/llama.cpp/build/bin/llama-cli \
  --model /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf
```

Server mode:

```bash
./Summer.sh \
  --mode server \
  --bin /path/to/llama.cpp/build/bin/llama-server \
  --model /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --profile balanced
```

## Supported models

The launcher is intentionally focused on Qwen3.5 MoE GGUF files whose GGUF metadata reports:

```text
general.architecture = qwen35moe
```

Typical targets include:

- Qwen3.5-35B-A3B
- Qwen3.5-122B-A10B
- compatible derivatives using the same GGUF architecture

For split GGUF models, pass the **first shard**:

```text
Qwen3.5-122B-A10B-Q4_K_M-00001-of-0000N.gguf
```

Summer rejects later shards because tuning decisions are based on metadata from the first shard.

## Profiles

| Profile | Goal | Context | KV cache | MoE placement | Batch / uBatch |
|---|---|---:|---|---|---:|
| `balanced` | Recommended starting point | 8K | Q8_0 / Q8_0 | automatic `--fit` | 2048 / 256 |
| `low-vram` | Minimize VRAM use | 8K | Q4_0 / Q4_0 | `--cpu-moe` | 1024 / 128 |
| `gpu` | More GPU-heavy starting point | 16K | Q8_0 / Q8_0 | GPU + `--fit` | 2048 / 256 |
| `long-context` | Context-first starting point | 32K | Q4_0 / Q4_0 | automatic `--fit` | 1024 / 128 |

Linux/macOS:

```bash
./Summer.sh --bin /path/to/llama-cli --model /path/to/model.gguf --profile low-vram
```

Windows:

```powershell
.\Summer.ps1 --bin "C:\llama.cpp\llama-cli.exe" --model "D:\models\model.gguf" --profile low-vram
```

The `low-vram` profile keeps routed expert weights on CPU. This can save substantial VRAM, but may reduce token-generation speed when system RAM bandwidth or CPU-GPU transfer becomes the bottleneck.

## MoE CPU offload

Keep all expert weights on CPU:

```text
--cpu-moe
```

Keep only the first N MoE layers on CPU:

```text
--n-cpu-moe 20
```

Examples:

```bash
./Summer.sh --bin /path/to/llama-cli --model /path/to/model.gguf --cpu-moe
./Summer.sh --bin /path/to/llama-cli --model /path/to/model.gguf --n-cpu-moe 20
```

`--cpu-moe` and `--n-cpu-moe` cannot be enabled at the same time.

## MTP speculative decoding

MTP can be enabled when both conditions are true:

- the GGUF declares an embedded MTP head
- the installed `llama.cpp` build exposes compatible `draft-mtp` options

Automatic:

```text
--mtp auto
```

Require MTP support:

```text
--mtp on --mtp-tokens 2
```

The default is `off`. MTP is hardware-sensitive: benchmark it against normal decoding using the same model, prompt, context and sampling settings.

## Reasoning

The default is:

```text
--reasoning auto
```

You can explicitly select:

```text
--reasoning on
--reasoning off
```

This leaves reasoning behavior under `llama.cpp` and the model's chat template instead of forcing one mode for every workload.

## KV cache

Balanced:

```text
--ctk q8_0 --ctv q8_0
```

Lower VRAM use:

```text
--ctk q4_0 --ctv q4_0
```

Higher-precision cache:

```text
--ctk f16 --ctv f16
```

When quantized KV cache is selected, Summer requires a backend that exposes Flash Attention support.

## Context, batch and micro-batch

Example:

```bash
./Summer.sh \
  --mode server \
  --bin /path/to/llama-server \
  --model /path/to/model.gguf \
  --ctx 32768 \
  --batch 2048 \
  --ubatch 128 \
  --fit-target 1024
```

PowerShell uses the same flags:

```powershell
.\Summer.ps1 --mode server --bin "C:\llama.cpp\llama-server.exe" --model "D:\models\model.gguf" --ctx 32768 --batch 2048 --ubatch 128 --fit-target 1024
```

Do not assume that the maximum model context is the best local-inference setting. Larger context increases memory requirements and can change prompt-processing performance considerably.

## Passing raw llama.cpp options

Everything after `--` is appended to the generated `llama.cpp` command.

Linux/macOS:

```bash
./Summer.sh \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  -- \
  --temp 1.0 --top-p 0.95 --top-k 20
```

Windows:

```powershell
.\Summer.ps1 --bin "C:\llama.cpp\llama-cli.exe" --model "D:\models\model.gguf" -- --temp 1.0 --top-p 0.95 --top-k 20
```

Later options may override an automatically generated option depending on how that `llama.cpp` build parses duplicate arguments.

## OpenAI-compatible server

Linux/macOS:

```bash
./Summer.sh \
  --mode server \
  --bin /path/to/llama-server \
  --model /models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --profile balanced \
  --host 127.0.0.1 \
  --port 8080 \
  --alias qwen35
```

Windows:

```powershell
.\Summer.ps1 --mode server --bin "C:\llama.cpp\llama-server.exe" --model "D:\models\Qwen3.5-35B-A3B-Q4_K_M.gguf" --profile balanced --host 127.0.0.1 --port 8080 --alias qwen35
```

Endpoint:

```text
http://127.0.0.1:8080/v1
```

## How it works

The repository now has one shared launcher core:

```text
summer.py     cross-platform configuration logic
Summer.sh     Bash wrapper for Linux/macOS
Summer.ps1    PowerShell wrapper for Windows
```

The launcher reads only the GGUF metadata needed for configuration. It does not load model tensors itself.

It then probes `llama-cli --help` or `llama-server --help` and uses supported upstream options such as:

- `--cpu-moe` / `--n-cpu-moe`
- `--gpu-layers auto`
- `--fit on` / `--fit-target`
- `--flash-attn on`
- `--cache-type-k` / `--cache-type-v`
- `--spec-type draft-mtp`
- `--reasoning auto|on|off`
- `--jinja` for server mode

The default path intentionally avoids private patches and fork-only expert-routing environment variables.

For the rationale behind the presets, see [docs/TUNING.md](docs/TUNING.md).

## References

- llama.cpp: https://github.com/ggml-org/llama.cpp
- llama.cpp CLI: https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md
- llama.cpp server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Qwen3.5-35B-A3B: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- Qwen3.5-122B-A10B: https://huggingface.co/Qwen/Qwen3.5-122B-A10B
- Qwen3.5-35B-A3B GGUF: https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF

## License

MIT. See [LICENSE](LICENSE).
