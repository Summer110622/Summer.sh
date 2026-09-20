# Summer.sh

Qwen3.5 MoE を upstream `llama.cpp` で動かすための設定ランチャーです。

`Summer.sh` は GGUF のメタデータを読み、Qwen3.5 MoE (`qwen35moe`) であることを確認したうえで、MoE CPU offload、VRAM 自動 fit、KV cache、MTP、reasoning などを一つのコマンドにまとめます。モデル本体や `llama.cpp` の改変版は同梱しません。

> プリセットは万能なベンチマーク結果ではなく、安全な開始点です。最終的な速度は CPU、GPU、RAM/VRAM 帯域、量子化、context length に大きく依存します。

## 対象

主な対象は Qwen3.5 の MoE 系 GGUF です。

- Qwen3.5-35B-A3B: 35B total / 3B activated、256 experts、8 routed experts + 1 shared expert、40 layers
- Qwen3.5-122B-A10B: 122B total / 10B activated、256 experts、8 routed experts + 1 shared expert、48 layers
- GGUF `general.architecture` が `qwen35moe` の派生モデル

Qwen3.5 は Gated DeltaNet と full attention を混ぜた hybrid architecture です。通常の dense Transformer と同じ感覚で VRAM、KV cache、micro-batch を決めるより、MoE と hybrid layer を意識した設定が有効です。

## 必要なもの

- Python 3
- upstream `llama.cpp` の比較的新しい `llama-cli` または `llama-server`
- Qwen3.5 MoE の GGUF

`Summer.sh` は起動時に `llama-cli --help` / `llama-server --help` を確認し、利用するオプションがその build に存在するか検査します。

## Quick start

まず `llama.cpp` と GGUF を用意します。公式の `ggml-org/Qwen3.5-35B-A3B-GGUF` も利用できます。

```bash
git clone https://github.com/Summer110622/Summer.sh.git
cd Summer.sh

bash Summer.sh \
  --bin /path/to/llama.cpp/build/bin/llama-cli \
  --model /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf
```

OpenAI-compatible server を起動する場合:

```bash
bash Summer.sh \
  --mode server \
  --bin /path/to/llama.cpp/build/bin/llama-server \
  --model /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --profile balanced
```

生成される `llama.cpp` コマンドだけ確認する場合は `--dry-run` を付けます。

```bash
bash Summer.sh \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  --dry-run
```

## Profiles

| Profile | 狙い | Context | KV cache | MoE placement | Batch / uBatch |
|---|---|---:|---|---|---:|
| `balanced` | まず試す既定値 | 8K | Q8_0 / Q8_0 | `--fit` に任せる | 2048 / 256 |
| `low-vram` | VRAM を節約 | 8K | Q4_0 / Q4_0 | `--cpu-moe` | 1024 / 128 |
| `gpu` | VRAM に余裕がある構成 | 16K | Q8_0 / Q8_0 | GPU + `--fit` | 2048 / 256 |
| `long-context` | context 優先の開始点 | 32K | Q4_0 / Q4_0 | `--fit` に任せる | 1024 / 128 |

例:

```bash
bash Summer.sh \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  --profile low-vram
```

`low-vram` は routed expert weights を CPU に置くため VRAM を大きく節約できますが、token generation が遅くなる場合があります。

## MoE を明示的に CPU へ置く

全 expert weights を CPU に置く:

```bash
bash Summer.sh \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  --cpu-moe
```

先頭 N layer だけ CPU に置く:

```bash
bash Summer.sh \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  --n-cpu-moe 20
```

`--cpu-moe` と `--n-cpu-moe` は同時指定できません。

## MTP speculative decoding

Qwen3.5 MoE の GGUF が embedded MTP head を持ち、`llama.cpp` build が `draft-mtp` を公開している場合だけ有効化できます。

```bash
bash Summer.sh \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  --mtp auto
```

強制する場合:

```bash
bash Summer.sh \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  --mtp on \
  --mtp-tokens 2
```

既定値は `off` です。MTP は hardware と offload 構成によって速くも遅くもなるため、同じ prompt で baseline と比較してから常用してください。

## Thinking / reasoning

既定値は `--reasoning auto` です。Qwen3.5 の chat template と `llama.cpp` の判定に任せます。

```bash
# thinking を明示的に有効化
bash Summer.sh ... --reasoning on

# non-thinking
bash Summer.sh ... --reasoning off
```

Qwen3.5 は `/think` / `/nothink` の soft switch を公式には前提としていないため、起動設定または API 側の reasoning 設定を使う方が明確です。

## KV cache

Qwen3.5 の hybrid architecture では、KV cache を無制限に高精度化するより、VRAM と context のバランスを取る方が実用的です。

```bash
# 精度寄り
bash Summer.sh ... --ctk q8_0 --ctv q8_0

# VRAM 寄り
bash Summer.sh ... --ctk q4_0 --ctv q4_0
```

量子化 KV cache を使う場合、ランチャーは `--flash-attn` が利用できる build を要求します。

## Context / batch を上書きする

```bash
bash Summer.sh \
  --bin /path/to/llama-server \
  --mode server \
  --model /path/to/model.gguf \
  --ctx 32768 \
  --batch 2048 \
  --ubatch 128 \
  --fit-target 1024
```

Qwen3.5 の official model card は native context 262,144 tokens を示していますが、実際に確保できる context は量子化、KV cache、VRAM/RAM、backend に依存します。最初から最大 context を指定することは推奨しません。

## llama.cpp の追加オプション

`--` より後ろはそのまま `llama.cpp` に渡します。

```bash
bash Summer.sh \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  -- \
  --temp 1.0 --top-p 0.95 --top-k 20
```

後段の引数が同じ設定を指定した場合、`llama.cpp` 側の解釈によってはランチャーの値を上書きできます。

## Server example

```bash
bash Summer.sh \
  --mode server \
  --bin /path/to/llama-server \
  --model /models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --profile balanced \
  --host 127.0.0.1 \
  --port 8080 \
  --alias qwen35
```

API endpoint:

```text
http://127.0.0.1:8080/v1
```

## Split GGUF

分割 GGUF では必ず最初の shard を渡してください。

```text
Qwen3.5-122B-A10B-Q4_K_M-00001-of-0000N.gguf
```

`Summer.sh` は `split.no != 0` の shard を拒否します。

## 設計方針

このリポジトリは upstream `llama.cpp` の公開オプションだけを既定経路で使います。特定 fork だけに存在する expert-routing 環境変数や private patch は標準プリセットに含めません。

主に利用する upstream 機能:

- `--cpu-moe` / `--n-cpu-moe`
- `--gpu-layers auto`
- `--fit on` / `--fit-target`
- `--flash-attn on`
- `--cache-type-k` / `--cache-type-v`
- `--spec-type draft-mtp`
- `--reasoning auto|on|off`
- `--jinja` (server)

詳しいチューニング理由は [docs/TUNING.md](docs/TUNING.md) を参照してください。

## References

- llama.cpp: https://github.com/ggml-org/llama.cpp
- llama.cpp CLI options: https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md
- llama.cpp server options: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Qwen3.5-35B-A3B: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- Qwen3.5-122B-A10B: https://huggingface.co/Qwen/Qwen3.5-122B-A10B
- ggml-org Qwen3.5-35B-A3B GGUF: https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF

## License

MIT. See [LICENSE](LICENSE).
