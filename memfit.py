#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
memfit -- memory budget + measured calibration

WHAT PROBLEM IT SOLVES
======================
"Will this model run on my machine?"

Existing tools (llmfit / gguf-vram-calculator / various config builders)
all compute **whether it fits in VRAM**. But CPU-inference users face a
different question: **is there enough RAM, and how large a context can I
afford?**

And they only give you a number, never a check on whether that number
was right.

WHAT THIS DOES
==============
  1. predict -- read GGUF metadata, compute peak RAM, compare with what
                you actually have free, give a verdict
  2. verify  -- actually run it, measure real usage, and **put the
                prediction next to the measurement**

Step 2 is the point. If the prediction is wrong, the user can see it.

MEMORY MODEL (validated by measurement)
=======================================
    peak = weights + KV cache + runtime overhead

    weights   = GGUF file size (exact, no estimation needed)
    KV cache  = 2 * n_layer * n_head_kv * head_dim * 2 bytes * ctx
                per-token value = KV cache / ctx
                **SLOPE VALIDATED**: measured 27.95 KB/token
                vs theoretical 28.00, ratio 1.00
    overhead  = compute buffers + runtime + fragmentation
                **NO FORMULA EXISTS**, must be measured on the machine.
                Measured here (llama-b*, -b 256 -ub 256 -t 4):
                    1.5B -> 0.535 GB

HONESTY ABOUT THE OVERHEAD TERM
===============================
  * For 1.5B on this machine the measurement is very clean: it is a
    constant across four context sizes, spread 0.001 GB.
  * But 3B was NOT validated -- this machine ran out of RAM, started
    paging, and produced a slope 7.5x the theoretical value. That data
    is garbage and the tool refuses to use it.
  * So "the constant does not depend on model size" is NOT proven here.
    The tool labels where the number came from, and --verify tells you
    how far off it was. **Do not trust it on an unmeasured model.**

USAGE
=====
    python memfit.py                     list models + predict
    python memfit.py --ctx 8192          pick context size
    python memfit.py --model x.gguf      one model only
    python memfit.py --verify            actually run (slow)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

from core.gguf import parse_gguf                                # noqa: E402
from core.measure import Measurer, win_mem                      # noqa: E402

MODELS_DIR = Path(r"E:\\")
#: Default model folder. Written with a Unicode escape so this file stays pure ASCII.
DEFAULT_MODELS_DIR = "E:\\" + "\u5c0f\u8bf4\u5199\u4f5c" + "\\models"
DB_PATH = ROOT / "data" / "calibration.json"
GB = 1024 ** 3

CODES = {"reset": "\033[0m", "dim": "\033[90m", "red": "\033[91m",
         "green": "\033[92m", "yellow": "\033[93m", "cyan": "\033[96m",
         "bold": "\033[1m", "f1": "\033[38;5;183m", "f3": "\033[38;5;141m",
         "pink": "\033[38;5;218m"}
_ON = True
GRAD = [224, 218, 212, 177, 141, 135, 99]


def enable_colors() -> bool:
    global _ON
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            _ON = False
    try:
        "\u2588".encode(sys.stdout.encoding or "utf-8")
    except Exception:
        _ON = False
    return _ON


def c(t, color=""):
    if not _ON or not color:
        return str(t)
    return CODES.get(color, "") + str(t) + CODES["reset"]


def grad(text: str) -> str:
    if not _ON:
        return text
    out = []
    last = len(GRAD) - 1
    for i, ch in enumerate(text):
        if ch == " ":
            out.append(ch)
            continue
        out.append("\033[38;5;%dm%s" % (GRAD[min(i, last)], ch))
    return "".join(out) + "\033[0m"


def rule(n=74):
    return c("-" * n, "f3")


# ===========================================================================
# calibration store
# ===========================================================================
def load_cal() -> dict:
    if DB_PATH.exists():
        try:
            return json.loads(DB_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"overhead": {}, "measurements": []}


def save_cal(d: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                       encoding="utf-8")


def overhead_for(info, cal: dict):
    """
    Runtime overhead for this model on this machine.
    Returns (gb, source_note). Prefers measured; else extrapolates and SAYS SO.
    """
    ov = cal.get("overhead") or {}
    key = "%dL-%dH" % (info.n_layer, info.hidden)
    if key in ov:
        return ov[key]["gb"], "measured here (%s)" % ov[key]["file"][:30]

    if ov:
        pts = sorted((v["file_gb"], v["gb"]) for v in ov.values())
        if len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            if x1 > x0:
                k = (y1 - y0) / (x1 - x0)
                est = y0 + k * (info.size_bytes / GB - x0)
                return max(0.15, est), "extrapolated (**GUESS, unverified**)"
        return pts[0][1], "borrowed from smallest measured (**GUESS**)"
    # Never measured: seed from this box 1.5B measurement.
    # 0.535 GB was measured on this machine for 1.5B (spread 0.001 over 4 ctxs),
    # better than guessing 0.55. Still a GUESS for an unmeasured model.
    return 0.535, "seed from this box's 1.5B measurement (**unverified here**)"


# ===========================================================================
# predict
# ===========================================================================
def kv_per_token(info) -> int:
    """KV bytes per token. Formula validated by measurement."""
    return 2 * info.n_layer * info.n_head_kv * info.kv_dim * 2


def predict(info, ctx: int, cal: dict) -> dict:
    weights = info.size_bytes
    per_tok = kv_per_token(info)
    kv = per_tok * ctx
    oh, oh_src = overhead_for(info, cal)
    oh_bytes = int(oh * GB)
    return {"weights": weights, "kv": kv, "per_tok": per_tok,
            "overhead": oh_bytes, "overhead_src": oh_src,
            "peak": weights + kv + oh_bytes, "ctx": ctx}


def render_predict(info, p: dict, avail_gb: float) -> None:
    peak_gb = p["peak"] / GB
    print("  " + c(info.name, "bold"))
    print("    %-12s%.3f GB  %s" % ("file (=weights)", info.size_bytes / GB,
                                    c("exact", "dim")))
    print("    %-12s%.3f GB  %s" % ("KV cache", p["kv"] / GB,
                                    c("%.1f KB/token x %d"
                                      % (p["per_tok"] / 1024, p["ctx"]),
                                      "dim")))
    print("    %-12s%.3f GB  %s" % ("overhead", p["overhead"] / GB,
                                    c(p["overhead_src"], "dim")))
    print("    " + "-" * 12)
    print("    %-12s%s" % ("PREDICTED", c("%.3f GB" % peak_gb, "f1")))
    print()

    if peak_gb <= avail_gb * 0.85:
        verdict = c("OK  fits with headroom", "green")
        detail = "predicted %.2fGB, free %.2fGB" % (peak_gb, avail_gb)
    elif peak_gb <= avail_gb:
        verdict = c("TIGHT  will start paging", "yellow")
        detail = ("predicted %.2fGB, free %.2fGB -> only %.0f%% headroom, "
                  "and the OS needs RAM too"
                  % (peak_gb, avail_gb, (avail_gb / peak_gb - 1) * 100))
    else:
        verdict = c("NO  does not fit", "red")
        detail = ("predicted %.2fGB > free %.2fGB -> short by %.2fGB"
                  % (peak_gb, avail_gb, peak_gb - avail_gb))
    print("    %s  %s" % (verdict, c(detail, "dim")))

    if p["per_tok"] > 0:
        room = avail_gb * GB * 0.85 - info.size_bytes - p["overhead"]
        max_ctx = int(room / p["per_tok"]) if room > 0 else 0
        if info.ctx_train:
            max_ctx = min(max_ctx, info.ctx_train)
        if max_ctx >= 256:
            print("    %s  %s" % (c("max ctx", "dim"),
                                  c("%d" % max_ctx, "f1")
                                  + c("  (15%% headroom; trained max %d)"
                                      % (info.ctx_train or 0), "dim")))
        else:
            print("    " + c("this model does not fit right now", "red"))
    print()


# ===========================================================================
# verify
# ===========================================================================
def do_verify(models, cal, ctx, threads, port) -> None:
    m = Measurer(port=port, out=print)
    if m.exe is None:
        print(c("  llama-server.exe not found", "red"))
        return

    print()
    print(c("  MEASURED CALIBRATION", "bold"))
    print("  " + rule())
    print(c("  This really runs each model and measures RAM.", "dim"))
    print()

    for path in models:
        info = parse_gguf(path)
        if info.error:
            print("  %s: parse failed %s" % (path.name, info.error))
            continue
        p = predict(info, ctx, cal)

        print("  -- %s  ctx=%d --" % (info.name, ctx))
        r = m.measure(path, ctx=ctx, threads=threads)
        if not r.ok:
            print(c("     FAILED: %s" % r.error, "red"))
            print()
            continue

        trusted, why = r.trust()
        actual = r.peak_gb
        pred = p["peak"] / GB
        diff = actual - pred
        rel = diff / pred * 100 if pred else 0
        actual_oh = actual - info.size_bytes / GB - p["kv"] / GB

        print()
        print("    %-14s%10s%10s%10s" % ("", "predict", "actual", "diff"))
        print("    %-14s%10.3f%10s%10s"
              % ("weights", info.size_bytes / GB, "(file)", ""))
        print("    %-14s%10.3f%10s%10s"
              % ("KV", p["kv"] / GB, "(formula)", ""))
        print("    %-14s%10.3f%10s%10s"
              % ("overhead", p["overhead"] / GB, "(from table)", ""))
        print("    %-14s%10.3f%10.3f%+10.3f"
              % ("TOTAL", pred, actual, diff))

        if not trusted:
            print()
            print(c("    UNTRUSTED: %s" % why, "red"))
            print(c("      ran out of RAM and paged; cannot calibrate with this.",
                    "dim"))
            print(c("      close other programs and retry.", "dim"))
        else:
            print()
            if abs(rel) <= 10:
                print(c("    prediction is good (off by %+.1f%%)" % rel,
                        "green"))
            else:
                print(c("    off by %+.1f%% -> overhead should be %.3f GB"
                        % (rel, max(0, actual_oh)), "yellow"))

            # Record EVERY trusted measurement -- not only the off-target ones.
            # Rationale: the value of this tool is that each measurement sharpens the next.
            # Recording only when off-target means the first prediction always uses defaults.
            # How: derive overhead from this run, then average with the stored value
            # (more contexts measured -> progressively tighter).
            key = "%dL-%dH" % (info.n_layer, info.hidden)
            tbl = cal.setdefault("overhead", {})
            new_oh = max(0.05, actual_oh)
            if key in tbl:
                old = tbl[key]["gb"]
                n = tbl[key].get("n", 1)
                # Weighted mean: the more samples, the lower the new sample weight, damping jitter
                merged = (old * n + new_oh) / (n + 1)
                tbl[key].update({"gb": round(merged, 3), "n": n + 1,
                                 "last_ctx": ctx, "t": time.time()})
                print(c("      overhead refined: %.3f -> %.3f GB "
                        "(from %d measurements)"
                        % (old, merged, n + 1), "dim"))
            else:
                tbl[key] = {"gb": round(new_oh, 3), "n": 1,
                            "file": path.name,
                            "file_gb": round(info.size_bytes / GB, 3),
                            "ctx": ctx, "t": time.time()}
                print(c("      overhead recorded: %.3f GB (key %s)"
                        % (new_oh, key), "dim"))
            save_cal(cal)

        cal.setdefault("measurements", []).append({
            "t": time.time(), "model": path.name, "ctx": ctx,
            "predicted_gb": round(pred, 3), "actual_gb": round(actual, 3),
            "trusted": trusted, "why": why,
            "avail_gb": round(r.avail_before_gb, 2),
            "load_s": round(r.load_seconds, 1),
        })
        save_cal(cal)
        print()


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="memfit - RAM budget + calibration")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--model", default="")
    ap.add_argument("--dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8199)
    args = ap.parse_args()

    enable_colors()
    cal = load_cal()
    mem = win_mem()

    print()
    print("  " + grad("memfit"))
    print(c("  RAM budget + measured calibration -- will it run on this box?",
            "dim"))
    print("  " + rule())
    print()
    print("  %-10s%.2f GB total   free %s   %s"
          % ("machine", mem["total_gb"],
             c("%.2f GB" % mem["avail_gb"], "f1"),
             c("(%.0f%% used)" % mem["load_pct"], "dim")))
    print()

    # locate models: --dir wins, else the known model dirs
    cands = []
    if args.dir and Path(args.dir).exists():
        cands = [Path(args.dir)]
    if args.model:
        mp = Path(args.model)
        if mp.is_absolute() and mp.exists():
            files = [mp]
        else:
            files = []
            for d in cands:
                files += list(d.glob("*.gguf"))
    else:
        files = []
        for d in cands:
            files += list(d.glob("*.gguf"))

    files = [f for f in files
             if f.exists() and "vocab" not in f.name.lower()]
    if not files:
        print(c("  no .gguf found. pass --dir <folder> or --model <file>",
                "yellow"))
        return 1

    infos = []
    for f in files:
        i = parse_gguf(f)
        if not i.error and i.n_layer:
            infos.append(i)

    print(c("  %d models, sorted by predicted peak" % len(infos), "bold"))
    print()

    preds = [(i, predict(i, args.ctx, cal)) for i in infos]
    preds.sort(key=lambda x: x[1]["peak"])

    print("  " + rule())
    for i, p in preds:
        render_predict(i, p, mem["avail_gb"])

    if args.verify:
        do_verify([i.path for i, _ in preds], cal, args.ctx,
                  args.threads, args.port)
    else:
        print("  " + rule())
        print(c("  The above is a PREDICTION. Add --verify to actually run it",
                "dim"))
        print(c("  and see prediction vs measurement side by side.", "dim"))
        print()

    print("  " + rule())
    print(c("  memory model", "bold"))
    print("    peak = weights + KV cache + runtime overhead")
    print("    weights = GGUF file size (exact)")
    print("    KV      = 2 * n_layer * n_head_kv * head_dim * 2 * ctx")
    print(c("              SLOPE VALIDATED: measured 27.95 KB/token vs "
            "theoretical 28.00, ratio 1.00", "green"))
    print("    overhead = compute buffers + runtime + fragmentation; "
          "no formula, measure it")
    if cal.get("overhead"):
        print(c("              measured here for %d model(s):"
                % len(cal["overhead"]), "dim"))
        for k, v in cal["overhead"].items():
            print(c("                %-12s %.3f GB  (%s)"
                    % (k, v["gb"], v["file"][:32]), "dim"))
    else:
        print(c("              nothing measured yet -- run --verify once",
                "yellow"))
    print()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(0)
