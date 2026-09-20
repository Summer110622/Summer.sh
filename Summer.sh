#!/usr/bin/env bash
# Qwen3.5 MoE launcher for upstream llama.cpp.
# Bash is only used as a tiny bootstrap; the implementation uses Python stdlib.
set -euo pipefail
command -v python3 >/dev/null 2>&1 || { printf 'ERROR: python3 is required.\n' >&2; exit 127; }
exec python3 - "$@" <<'PY'
import argparse
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

VERSION = "2.0.0"
ARCH = "qwen35moe"


def fail(message):
    raise ValueError(message)


def as_int(value, name, low=0, high=2_147_483_647):
    text = str(value)
    if not re.fullmatch(r"[0-9]+", text):
        fail(f"{name}: expected an integer, got {value!r}")
    number = int(text)
    if not low <= number <= high:
        fail(f"{name}: expected {low}..{high}, got {number}")
    return number


class GGUF:
    """Minimal GGUF metadata reader; tensor data is never loaded."""
    FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def __init__(self, stream):
        self.f = stream
        self.size = os.fstat(stream.fileno()).st_size
        self.endian = "<"

    def read(self, count):
        if count < 0 or count > self.size - self.f.tell():
            fail("Truncated GGUF file")
        data = self.f.read(count)
        if len(data) != count:
            fail("Could not read GGUF metadata")
        return data

    def skip(self, count):
        if count < 0 or count > self.size - self.f.tell():
            fail("Invalid GGUF metadata length")
        self.f.seek(count, 1)

    def number(self, fmt):
        return struct.unpack(self.endian + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, keep=False):
        size = self.number("Q")
        if size > self.size - self.f.tell():
            fail("Invalid GGUF string length")
        if keep:
            if size > 1024 * 1024:
                fail("GGUF metadata string is too large")
            return self.read(size).decode("utf-8")
        self.skip(size)
        return None

    def value(self, kind, keep=False):
        if kind in self.FORMATS:
            return self.number(self.FORMATS[kind])
        if kind == 8:
            return self.string(keep)
        if kind == 9:
            subtype = self.number("I")
            count = self.number("Q")
            if keep:
                fail("Required GGUF metadata unexpectedly uses an array")
            if subtype in self.FORMATS:
                self.skip(count * struct.calcsize(self.FORMATS[subtype]))
            elif subtype == 8:
                for _ in range(count):
                    self.string(False)
            else:
                fail(f"Unsupported GGUF array type: {subtype}")
            return None
        fail(f"Unsupported GGUF value type: {kind}")

    def metadata(self):
        if self.read(4) != b"GGUF":
            fail("Not a GGUF file")
        raw = self.read(4)
        version = struct.unpack("<I", raw)[0]
        if version not in (2, 3):
            self.endian = ">"
            version = struct.unpack(">I", raw)[0]
        if version not in (2, 3):
            fail(f"Unsupported GGUF version: {version}")
        self.number("Q")  # tensor_count
        n_kv = self.number("Q")
        if n_kv > min(1_000_000, self.size // 12):
            fail("Invalid GGUF metadata entry count")
        wanted = {
            "general.architecture",
            "general.name",
            "split.no",
            "split.count",
            f"{ARCH}.expert_count",
            f"{ARCH}.expert_used_count",
            f"{ARCH}.block_count",
            f"{ARCH}.context_length",
            f"{ARCH}.nextn_predict_layers",
        }
        out = {}
        for _ in range(n_kv):
            key = self.string(True)
            keep = key in wanted
            value = self.value(self.number("I"), keep)
            if keep:
                out[key] = value
        return out


def model_info(path):
    with path.open("rb") as stream:
        meta = GGUF(stream).metadata()
    arch = meta.get("general.architecture")
    if arch != ARCH:
        fail(f"Expected Qwen3.5 MoE GGUF (general.architecture={ARCH!r}), got {arch!r}")
    info = {"name": meta.get("general.name") or path.name, "architecture": arch}
    for key in ("expert_count", "expert_used_count", "block_count", "context_length"):
        value = meta.get(f"{ARCH}.{key}")
        if type(value) is not int:
            fail(f"GGUF is missing integer metadata: {ARCH}.{key}")
        info[key] = as_int(value, key, 1)
    if info["expert_used_count"] > info["expert_count"]:
        fail("GGUF expert_used_count exceeds expert_count")
    mtp = meta.get(f"{ARCH}.nextn_predict_layers", 0)
    if type(mtp) is not int:
        fail(f"GGUF has invalid {ARCH}.nextn_predict_layers")
    info["mtp_layers"] = as_int(mtp, "nextn_predict_layers", 0, info["block_count"])
    if meta.get("split.no", 0) != 0:
        fail("For split GGUF models, pass the first shard (-00001-of-....gguf)")
    return info


def binary_path(value, mode):
    if not value:
        fail(f"Specify --bin /path/to/llama-{mode}")
    path = Path(value).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        fail(f"Backend is not executable: {path}")
    return str(path)


def clean_environment():
    # The launcher should be reproducible: stale LLAMA_ARG_* values must not
    # silently change a published preset.
    return {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}


def cpu_counts():
    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    physical = logical
    try:
        ids = set()
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else range(logical)
        for cpu in cpus:
            base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            ids.add(((base / "physical_package_id").read_text().strip(), (base / "core_id").read_text().strip()))
        if ids:
            physical = len(ids)
    except OSError:
        pass
    return max(1, physical), max(1, logical)


def backend_capabilities(binary, env):
    try:
        result = subprocess.run([binary, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", timeout=15, env=env)
    except subprocess.TimeoutExpired as exc:
        fail(f"{Path(binary).name} --help timed out: {exc}")
    if result.returncode:
        fail(f"{Path(binary).name} --help failed with exit code {result.returncode}")
    flags = set(re.findall(r"(?<![\w-])--?[A-Za-z][A-Za-z0-9-]*", result.stdout))
    return flags, result.stdout


PROFILE_DEFAULTS = {
    "balanced": {"ctx": 8192, "batch": 2048, "ubatch": 256, "ctk": "q8_0", "ctv": "q8_0", "fit_target": 1024, "cpu_moe": False},
    "low-vram": {"ctx": 8192, "batch": 1024, "ubatch": 128, "ctk": "q4_0", "ctv": "q4_0", "fit_target": 768, "cpu_moe": True},
    "gpu": {"ctx": 16384, "batch": 2048, "ubatch": 256, "ctk": "q8_0", "ctv": "q8_0", "fit_target": 1024, "cpu_moe": False},
    "long-context": {"ctx": 32768, "batch": 1024, "ubatch": 128, "ctk": "q4_0", "ctv": "q4_0", "fit_target": 1280, "cpu_moe": False},
}


def main():
    p = argparse.ArgumentParser(
        prog="Summer.sh",
        allow_abbrev=False,
        description="Launch upstream llama.cpp with Qwen3.5-MoE-oriented defaults.",
        epilog="Pass additional llama.cpp options after --. Later options may override the generated command.")
    p.add_argument("--version", action="version", version=f"Summer.sh {VERSION}")
    p.add_argument("--mode", choices=("cli", "server"), default="cli")
    p.add_argument("--bin", required=True, help="path to llama-cli or llama-server")
    p.add_argument("--model", required=True, help="Qwen3.5 MoE GGUF (first shard for split models)")
    p.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="balanced")
    p.add_argument("--ctx")
    p.add_argument("--batch")
    p.add_argument("--ubatch")
    p.add_argument("--threads")
    p.add_argument("--batch-threads")
    p.add_argument("--fit-target", help="MiB to keep free per accelerator; default comes from profile")
    p.add_argument("--n-cpu-moe", help="keep routed-expert weights for the first N layers on CPU")
    p.add_argument("--cpu-moe", action="store_true", help="keep all MoE expert weights on CPU")
    p.add_argument("--ctk", choices=("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"))
    p.add_argument("--ctv", choices=("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"))
    p.add_argument("--mtp", choices=("off", "auto", "on"), default="off",
                   help="embedded MTP self-speculation; off by default because speedups are hardware dependent")
    p.add_argument("--mtp-tokens", default="2", help="maximum MTP draft tokens, 1..8")
    p.add_argument("--reasoning", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="8080")
    p.add_argument("--alias", default="qwen35-moe")
    p.add_argument("--dry-run", action="store_true")

    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        index = argv.index("--")
        argv, extra = argv[:index], argv[index + 1:]
    args = p.parse_args(argv)

    model = Path(args.model).expanduser().absolute()
    if not model.is_file():
        fail(f"Model not found: {model}")
    info = model_info(model)
    binary = binary_path(args.bin, args.mode)
    env = clean_environment()
    flags, help_text = backend_capabilities(binary, env)
    supports = flags.__contains__

    d = PROFILE_DEFAULTS[args.profile]
    physical, logical = cpu_counts()
    ctx = as_int(args.ctx or min(d["ctx"], info["context_length"]), "ctx", 1)
    batch = as_int(args.batch or d["batch"], "batch", 1)
    ubatch = as_int(args.ubatch or min(d["ubatch"], batch), "ubatch", 1, batch)
    threads = as_int(args.threads or physical, "threads", 1)
    batch_threads = as_int(args.batch_threads or logical, "batch-threads", 1)
    fit_target = as_int(args.fit_target or d["fit_target"], "fit-target", 0)
    ctk = args.ctk or d["ctk"]
    ctv = args.ctv or d["ctv"]
    port = as_int(args.port, "port", 1, 65535)

    command = [binary, "-m", str(model), "-c", str(ctx), "-b", str(batch), "-ub", str(ubatch),
               "-t", str(threads), "-tb", str(batch_threads)]
    notes = []

    if supports("--gpu-layers"):
        command += ["--gpu-layers", "auto"]
    if supports("--fit"):
        command += ["--fit", "on"]
        if supports("--fit-target"):
            command += ["--fit-target", str(fit_target)]
    else:
        notes.append("backend has no --fit; accelerator placement is left to llama.cpp defaults")

    want_cpu_moe = args.cpu_moe or d["cpu_moe"]
    if args.n_cpu_moe is not None and want_cpu_moe:
        fail("Use either --cpu-moe or --n-cpu-moe, not both")
    if args.n_cpu_moe is not None:
        n_cpu_moe = as_int(args.n_cpu_moe, "n-cpu-moe", 0, info["block_count"])
        if not supports("--n-cpu-moe"):
            fail("This llama.cpp build does not support --n-cpu-moe")
        command += ["--n-cpu-moe", str(n_cpu_moe)]
    elif want_cpu_moe:
        if not supports("--cpu-moe"):
            fail("This llama.cpp build does not support --cpu-moe")
        command += ["--cpu-moe"]
        notes.append("all MoE expert weights stay on CPU; VRAM use drops but token generation may slow down")

    if supports("--flash-attn"):
        command += ["--flash-attn", "on"]
    elif ctk not in ("f32", "f16", "bf16") or ctv not in ("f32", "f16", "bf16"):
        fail("Quantized KV cache requested, but this build does not expose --flash-attn")

    for flag, value in (("--cache-type-k", ctk), ("--cache-type-v", ctv)):
        if not supports(flag):
            fail(f"This llama.cpp build does not support {flag}")
        command += [flag, value]

    mtp_enabled = False
    mtp_tokens = as_int(args.mtp_tokens, "mtp-tokens", 1, 8)
    required_mtp = {"--spec-type", "--spec-draft-n-max", "--spec-draft-n-min", "--spec-draft-p-min"}
    mtp_supported = info["mtp_layers"] > 0 and required_mtp <= flags and "draft-mtp" in help_text
    if args.mtp == "on" and not mtp_supported:
        fail("MTP was required, but the GGUF/backend does not expose compatible embedded draft-MTP support")
    if args.mtp != "off" and mtp_supported:
        mtp_enabled = True
        command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(mtp_tokens),
                    "--spec-draft-n-min", "0", "--spec-draft-p-min", "0"]
    elif supports("--spec-type"):
        command += ["--spec-type", "none"]

    if supports("--reasoning"):
        command += ["--reasoning", args.reasoning]
    elif args.reasoning != "auto":
        fail("This llama.cpp build does not support --reasoning")

    if args.mode == "cli":
        command += ["-n", "-1"]
    else:
        if supports("--jinja"):
            command += ["--jinja"]
        command += ["--host", args.host, "--port", str(port), "--alias", args.alias, "-np", "1"]

    command += extra

    print(f"Summer.sh {VERSION} — Qwen3.5 MoE / llama.cpp")
    print(f"  model          : {info['name']}")
    print(f"  architecture   : {info['architecture']}")
    print(f"  routed experts : {info['expert_used_count']} / {info['expert_count']}")
    print(f"  layers / ctx   : {info['block_count']} / {info['context_length']} (GGUF)")
    print(f"  profile        : {args.profile}")
    print(f"  ctx / b / ub   : {ctx} / {batch} / {ubatch}")
    print(f"  threads        : {threads} generation / {batch_threads} prompt")
    print(f"  KV K/V         : {ctk} / {ctv}")
    print(f"  MTP            : {'on' if mtp_enabled else 'off'} (GGUF heads={info['mtp_layers']})")
    print(f"  reasoning      : {args.reasoning}")
    if args.n_cpu_moe is not None:
        print(f"  CPU MoE        : first {args.n_cpu_moe} layers")
    else:
        print(f"  CPU MoE        : {'all' if want_cpu_moe else 'auto/off'}")
    for note in notes:
        print(f"  note           : {note}")
    if extra:
        print("  note           : arguments after -- are appended verbatim and can override generated settings")
    print("\n" + shlex.join(command), flush=True)

    if not args.dry_run:
        os.execve(binary, command, env)


try:
    main()
except KeyboardInterrupt:
    print("\nInterrupted.", file=sys.stderr)
    raise SystemExit(130)
except (OSError, UnicodeError, ValueError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2)
PY
