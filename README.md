# memfit

**RAM budget for GGUF models on CPU-only machines — with measured calibration.**

Answers the question *"will this model run on my box?"* and then **checks its
own answer** by actually running the model and comparing.

Developed and validated on a memory-starved Windows machine:
Ryzen 3 2200G, **5.95 GB RAM**, no discrete GPU, CPU-only llama.cpp.

[中文说明](README.zh-CN.md)

---

## The gap this fills

Most existing tools compute **whether a model fits in VRAM**:

| Tool | Question it answers |
|---|---|
| [llmfit](https://github.com/AlexsJones/llmfit) | "does this model fit your machine" |
| [gguf-vram-calculator](https://github.com/mrsaynothing/gguf-vram-calculator) | **"does it fit in VRAM"** |
| [gekro config builder](https://github.com/drajb/gekro) | recommends a config for your hardware |

They assume you have a GPU. If you are doing CPU inference on a small box,
your questions are different:

- Is there enough **RAM** (not VRAM)?
- **How large a context can I afford?** — almost nobody computes this.
- And the one that actually decides it: **how much runtime overhead is there?**

They also hand you a single number with no way to check it. `memfit --verify`
runs the model for real and prints prediction next to measurement. When the
prediction is wrong, you can see it.

---

## The memory model, and what is actually verified

```
peak = weights + KV cache + runtime overhead

weights        = GGUF file size                 (exact, no estimation)
KV cache       = 2 * n_layer * n_head_kv * head_dim * 2 bytes * ctx
runtime overhead = compute buffers + runtime + fragmentation
                   (no formula exists -- must be measured)
```

### KV cache formula: validated

Four context sizes on a 1.5B model, slope from the two extremes:

```
      ctx     peak GB   theory KV   non-KV
      512       1.589      0.014      1.576
     2048       1.631      0.055      1.576
     4096       1.686      0.109      1.576
     8192       1.794      0.219      1.575

measured slope      27.95 KB/token
theoretical         28.00 KB/token
measured/theoretical     1.00        <- agrees
```

The non-KV portion is constant across all four: **spread 0.001 GB**.

### The overhead term is the whole point

For the 1.5B model the file is **1.041 GB** but actual peak is **1.576 GB**:

```
runtime overhead = 1.576 - 1.041 = 0.535 GB
```

**That 0.535 GB is not in any file.** No amount of staring at model sizes
will reveal it. On a 5.95 GB machine it is the entire distance between
"runs fine" and "swaps itself to death".

### Accuracy after calibration

```
                      predict   actual      diff
qwen2.5-1.5b            1.630    1.630    -0.000     -0.0%
DeepSeek-R1-Distill-1.5B 1.630   1.631    +0.000     +0.0%
qwen2.5-3b              2.566    2.576    +0.010     +0.4%
```

---

## What is NOT verified (read this)

**"The overhead constant is independent of model size" cannot be proven on
this machine.**

The 3B model needs about 2.57 GB and this box usually has ~2.5 GB free. It
pages. The measured slope came out at **7.5x the theoretical value**
(269 vs 36 KB/token) — obviously garbage. The tool detects this (slow load,
low free RAM) and **refuses to calibrate with it**.

Consequences:

- **For 28-layer / hidden-1536 models on this machine, prediction is exact.**
- **Every other shape uses an extrapolated value, and the tool labels it.**
- To make it accurate for your models, run `--verify` once on each.

This is not laziness. For a quantity with no formula, labelling the guess is
the only honest thing to do.

---

## Usage

```console
$ python memfit.py                              # all models, default ctx 2048
$ python memfit.py --ctx 8192                   # pick a context size
$ python memfit.py --model qwen2.5-1.5b-instruct-q4_k_m.gguf
$ python memfit.py --dir "E:\your\models"
$ python memfit.py --verify --ctx 2048          # actually run and calibrate
```

### Output

```
  machine   5.95 GB total   free 2.53 GB   (56% used)

  qwen2.5-1.5b-instruct
    file (=weights)1.041 GB  exact
    KV cache    0.055 GB  28.0 KB/token x 2048
    overhead    0.535 GB
                -> measured here (qwen2.5-1.5b-instruct-q4_k_m.gguf)
    ------------
    PREDICTED   1.630 GB

    OK  fits with headroom  predicted 1.63GB, free 2.53GB
    max ctx  21392  (15% headroom; trained max 32768)

  qwen2.5-3b-instruct
    file (=weights)1.960 GB  exact
    KV cache    0.070 GB  36.0 KB/token x 2048
    overhead    0.535 GB
                -> borrowed from the smallest measured -- a guess
    ------------
    PREDICTED   2.566 GB

    NO  does not fit  predicted 2.57GB > free 2.53GB -> short by 0.04GB
```

`--verify` adds the comparison table:

```
                    predict    actual      diff
    weights            1.041    (file)
    KV                 0.055    (formula)
    overhead           0.535    (from table)
    TOTAL              1.630     1.630    -0.000

    prediction is good (off by -0.0%)
      overhead recorded: 0.535 GB (key 28L-1536H)
```

---

## How calibration works

Stored in `data/calibration.json`, keyed by **model shape** (`28L-1536H` =
28 layers, hidden 1536) rather than filename, because different fine-tunes of
the same architecture share the same overhead.

**Every trusted measurement is recorded**, averaged in with a weighted mean
so repeated runs converge instead of oscillating. **Untrusted measurements
are refused** — that rule is deliberate: paging-induced garbage would
otherwise poison the whole table.

---

## Layout

```
memfit/
  memfit.py            entry point (pure ASCII)
  core/
    gguf.py            GGUF metadata reader (header only, no weights)
    measure.py         launch llama-server, measure peak RSS
  measure_sweep.py     sweep context sizes, derive the slope
  data/
    calibration.json   overhead table + measurement log
```

**Why a hand-written GGUF parser:** a few hundred KB of header gives every
structural parameter needed. There is no reason to touch the 2 GB of weights.
Existing libraries either carry heavy dependencies or do not expose the
fields required here.

**Why the trust check is not optional:** on a memory-starved box the numbers
come out wrong while looking perfectly normal. Only comparing the slope
against theory exposes it — the failure mode is a confident wrong answer,
not an error.

---

## Bugs found while building this

| Bug | Symptom | Cause |
|---|---|---|
| **Wrong slope units** | reported "130.32 MB/token" alongside "0.1 KB/token" | divided by `1024**2` instead of `1024**3`; GB→KB needs two factors of 1024 |
| **Cold-cache contamination** | first run loaded in 88s, second in 10s | first read hits disk; needs a warm-up pass |
| **Previous run lingering** | ctx=512 peaked *higher* than ctx=2048 | the prior model was still in page cache; needs a settle delay |
| **Paging inverted the slope** | 3B produced a *negative* slope | working set evicted to disk and read back; measurement meaningless |
| **Single inference under-reports** | peak kept climbing after the first request | compute buffers are allocated during the first inference; needs two |
| **Push reported false success** | helper said "pushed" while the repo was empty | git's exit code was never inspected — see `push_to_github.py` |
| **`--force-with-lease` refused with "stale info"** | push rejected after an amend | the lease needs a known remote-tracking ref, so a `fetch` must come first |

The negative-slope one is the valuable lesson: **the data did not error, it
calmly returned a wrong answer.** It only surfaced by comparing against theory.

---

## Requirements

- Python 3.8+ (standard library only)
- A `llama-server` binary (for `--verify`; prediction works without it)
- Windows, Linux or macOS — the measurement layer uses Windows APIs for
  process memory, so `--verify` is Windows-only for now while `predict`
  is portable

---

## Roadmap

1. **Batch calibration** — measure every model once, fill the table
2. **Inverse recommendation** — "I want 8K context, which quantisation fits?"
3. **History** — how the overhead term changes across llama.cpp builds
4. **More machines** — right now the validated sample is exactly one
5. **Report export** — markdown/HTML for pasting into an issue

---

## Licence

MIT
