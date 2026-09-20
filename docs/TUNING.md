# Qwen3.5 MoE tuning notes

この文書は `Summer.sh` の既定値を、どの方向に動かすべきか判断するためのメモです。数値はハードウェア横断の最適値ではありません。

## 1. Qwen3.5 MoE で重要な点

Qwen3.5-35B-A3B と 122B-A10B は、全 parameter を毎 token 使う dense model ではありません。両モデルとも 256 routed experts を持ち、各 token では 8 routed experts と shared expert が使われます。

そのため、`llama.cpp` の `--cpu-moe` / `--n-cpu-moe` は特に有用です。attention やその他の tensor を GPU に残しながら、大きな expert weights を CPU RAM 側へ逃がせます。

一方で、generation 中は選択された experts を読む必要があるため、CPU RAM bandwidth や PCIe transfer が bottleneck になり得ます。VRAM が足りるなら、CPU MoE を減らした方が速い場合が多くあります。

## 2. `--fit` を既定にする理由

最近の upstream `llama.cpp` は `--fit` により、VRAM に収まるよう未指定の offload parameter を調整できます。

`Summer.sh` は原則として:

```text
--gpu-layers auto --fit on --fit-target <MiB>
```

を使います。

`--fit-target` は accelerator memory に残す余白です。小さすぎると desktop / display / driver / context 増加で OOM しやすくなり、大きすぎると GPU offload が減ります。

開始点:

| 状況 | fit-target |
|---|---:|
| 余裕を優先 | 1280 MiB |
| 標準 | 1024 MiB |
| VRAM が厳しい | 768 MiB |

## 3. `balanced`

```text
ctx=8192
batch=2048
ubatch=256
K/V=q8_0/q8_0
fit-target=1024 MiB
```

最初に throughput と memory 使用量を見るための profile です。MoE placement は `llama.cpp` の fit logic に任せます。

## 4. `low-vram`

```text
ctx=8192
batch=1024
ubatch=128
K/V=q4_0/q4_0
--cpu-moe
fit-target=768 MiB
```

VRAM を最優先で節約します。全 MoE expert weights を CPU に残すため、GPU が小さい環境でも model を扱いやすくなります。

速度が低すぎる場合は `--cpu-moe` ではなく `--n-cpu-moe N` を使い、CPU に置く layer 数を減らしてください。

例:

```bash
bash Summer.sh ... --profile balanced --n-cpu-moe 20
```

## 5. `gpu`

```text
ctx=16384
batch=2048
ubatch=256
K/V=q8_0/q8_0
fit-target=1024 MiB
```

GPU 側により多く置ける環境向けです。ただし profile 名は「全 tensor を必ず GPU に固定する」という意味ではありません。OOM 回避のため `--fit` を残しています。

完全に手動で placement を固定したい場合は、`--` 以降に upstream `llama.cpp` のオプションを渡して比較してください。

## 6. `long-context`

```text
ctx=32768
batch=1024
ubatch=128
K/V=q4_0/q4_0
fit-target=1280 MiB
```

context を増やす代わりに KV cache と micro-batch を抑えます。Qwen3.5 の official native context は 262,144 tokens ですが、ローカル推論でその長さをそのまま確保する必要はありません。

32K → 64K → 128K のように段階的に増やし、起動時メモリだけでなく prompt processing と generation の安定性も確認してください。

## 7. KV cache

`q8_0/q8_0` はバランスの良い開始点です。VRAM が厳しい場合は `q4_0/q4_0` に落とせます。

高精度側へ戻す場合:

```bash
bash Summer.sh ... --ctk f16 --ctv f16
```

長 context では KV cache の型が memory 使用量へ直接効くため、context length とセットで調整してください。

## 8. Batch と micro-batch

`-b` は logical batch、`-ub` は実際に一度に処理する micro-batch です。

Qwen3.5 の hybrid/recurrent 部分を考えると、dense model で問題なかった大きな `ubatch` が最適とは限りません。OOM や prompt processing の不安定さが出たら、まず `ubatch` を半分にします。

順序としては:

```text
256 -> 128 -> 64
```

を試し、それでも厳しければ `batch` を下げます。

## 9. MTP

Qwen3.5 は MTP training を使っており、current upstream `llama.cpp` には Qwen3.5 MoE の embedded MTP graph があります。

ただし self-speculation の損益は hardware 依存です。CPU offload が大きい構成では verification cost が増え、MTP が逆効果になることがあります。

比較方法:

```bash
# baseline
bash Summer.sh ... --mtp off --dry-run

# MTP
bash Summer.sh ... --mtp on --mtp-tokens 2 --dry-run
```

実測時は同一 model、同一 context、同一 sampling、同一 prompt で tokens/s と first-token latency を比較してください。

## 10. Reasoning

`Summer.sh` は reasoning を勝手に無効化せず `auto` を既定にします。

```bash
bash Summer.sh ... --reasoning auto
bash Summer.sh ... --reasoning on
bash Summer.sh ... --reasoning off
```

Server 利用時は client request の sampling / max_tokens と合わせて設計してください。Thinking を切ると必要 output budget も変わります。

## 11. Sampling

ランチャーは sampling parameter を固定しません。理由は、Qwen3.5 の thinking と non-thinking で推奨例が異なること、`llama-server` では request 単位の parameter が優先されるためです。

CLI で試す場合は `--` 以降に渡せます。

```bash
bash Summer.sh ... -- --temp 1.0 --top-p 0.95 --top-k 20
```

## 12. 変更する順番

速度または memory を詰める場合、複数の軸を一度に変えない方が原因を追いやすくなります。

1. `balanced` で baseline を取る
2. OOM なら `fit-target` を増やす、または `low-vram`
3. 遅いなら `--n-cpu-moe` を減らして GPU 側へ戻す
4. context が必要なら KV cache と `ubatch` を下げながら増やす
5. 最後に MTP を比較する

`llama-bench` を使える環境では、profile を変える前後で prompt processing と token generation を分けて測ると判断しやすくなります。
