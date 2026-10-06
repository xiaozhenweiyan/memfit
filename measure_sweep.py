#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
实测扫描：同一个模型，不同上下文各跑一次，量峰值内存

目的：把内存拆成两部分
    总量(ctx) = 固定开销 + 斜率 x ctx

    * 固定开销 —— 权重 + 计算缓冲 + 运行时
    * 斜率       —— 主要是 KV cache，理论值能从 GGUF 元数据算出来

    实测斜率 vs 理论 KV/token：
        对得上 -> 公式是对的，可以拿去预测
        对不上 -> 公式漏了东西，先别发布

用法：
    python measure_sweep.py                   扫 1.5B，ctx 512~8192
    python measure_sweep.py --big             扫 3B
    python measure_sweep.py --ctx 512,2048    指定上下文
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

MODELS = Path(r"E:\小说写作\models")
OUT = ROOT / "data" / "measurements.json"


def load_db() -> dict:
    if OUT.exists():
        try:
            return json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"measurements": []}


def save_db(db: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(db, ensure_ascii=False, indent=2),
                   encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="用 3B 模型")
    ap.add_argument("--ctx", default="", help="逗号分隔的上下文长度")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8199)
    args = ap.parse_args()

    if args.big:
        path = MODELS / "qwen2.5-3b-instruct-q4_k_m.gguf"
    else:
        path = MODELS / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    if not path.exists():
        print(f"  找不到模型：{path}")
        return 1

    if args.ctx:
        ctxs = [int(x) for x in args.ctx.split(",") if x.strip()]
    else:
        ctxs = [512, 2048, 4096, 8192]

    info = parse_gguf(path)
    per_tok = 2 * info.n_layer * info.n_head_kv * info.kv_dim * 2

    mem = win_mem()
    print("=" * 76)
    print("  实测扫描")
    print("=" * 76)
    print(f"  模型　　{info.name}")
    print(f"  文件　　{path.name}　{info.size_bytes/1024**3:.3f} GB")
    print(f"  结构　　{info.n_layer} 层　hidden {info.hidden}　"
          f"KV头 {info.n_head_kv}×{info.kv_dim}")
    print(f"  理论 KV {per_tok/1024:.1f} KB/token")
    print(f"  整机内存 {mem['total_gb']:.2f} GB（可用 {mem['avail_gb']:.2f} GB，"
          f"已占 {mem['load_pct']:.0f}%）")
    print()
    if mem["avail_gb"] < 2.5:
        print("  ▲ 可用内存不到 2.5GB，实测可能失败。先关掉别的东西。")
        print()

    m = Measurer(port=args.port, out=print)
    if m.exe is None:
        print("  找不到 llama-server.exe")
        return 1
    print(f"  llama-server：{m.exe}")
    print()

    db = load_db()
    results = []
    for ctx in ctxs:
        print(f"  ── ctx = {ctx} ──")
        r = m.measure(path, ctx=ctx, threads=args.threads)
        if not r.ok:
            print(f"     ✘ 失败：{r.error}")
            results.append({"ctx": ctx, "ok": False, "error": r.error})
            continue
        expect_kv = per_tok * ctx
        trusted, why = r.trust()
        print(f"     ✔ 峰值工作集 {r.peak_gb:.3f} GB　"
              f"提交 {r.commit_gb:.3f} GB　加载 {r.load_seconds:.1f}s")
        print(f"       其中理论 KV {expect_kv/1024**3:.3f} GB　"
              f"→ 非 KV 部分 {(r.peak_gb - expect_kv/1024**3):.3f} GB")
        if trusted:
            print("       可信度：✔ 可信")
        else:
            print(f"       可信度：✘ 不可信 —— {why}")
            print("         （内存不够导致换页，这个数字不能拿去做结论）")
        results.append({
            "ctx": ctx, "ok": True, "trusted": trusted, "untrusted_why": why,
            "avail_before_gb": round(r.avail_before_gb, 2),
            "min_avail_gb": round(r.min_avail_gb, 2),
            "peak_gb": round(r.peak_gb, 4),
            "commit_gb": round(r.commit_gb, 4),
            "load_s": round(r.load_seconds, 1),
            "theory_kv_gb": round(expect_kv / 1024 ** 3, 4),
            "non_kv_gb": round(r.peak_gb - expect_kv / 1024 ** 3, 4),
        })
        print()

    # ---- 求斜率 ----
    allok = [x for x in results if x.get("ok")]
    good = [x for x in allok if x.get("trusted")]
    rejected = [x for x in allok if not x.get("trusted")]
    if rejected:
        print(f"\n  ▲ 有 {len(rejected)} 个测量因为换页不可信，"
              f"求斜率时已排除：")
        for x in rejected:
            print(f"     ctx={x['ctx']}  {x['untrusted_why']}")
    print("=" * 76)
    print("  结果")
    print("=" * 76)
    print(f"  {'ctx':>7}{'峰值GB':>10}{'理论KV':>10}{'非KV':>10}")
    for x in good:
        print(f"  {x['ctx']:>7}{x['peak_gb']:>10.3f}{x['theory_kv_gb']:>10.3f}"
              f"{x['non_kv_gb']:>10.3f}")

    if len(good) >= 2:
        # 用最小和最大 ctx 两点求斜率。
        # 单位换算要小心（这里踩过一次）：
        #   d_mem 是 GB，d_ctx 是 token
        #   slope 的单位是 GB/token
        #   换成 MB/token 要乘 1024；换成 KB/token 要乘 1024*1024
        a, b = good[0], good[-1]
        d_ctx = b["ctx"] - a["ctx"]
        d_mem = b["peak_gb"] - a["peak_gb"]
        slope_gb = d_mem / d_ctx if d_ctx else 0        # GB / token
        slope_mb = slope_gb * 1024                       # MB / token
        slope_kb = slope_gb * 1024 * 1024                # KB / token
        print()
        print(f"  实测斜率　{slope_mb:.3f} MB/token"
              f"　（{slope_kb:.2f} KB/token）")
        print(f"  理论 KV  {per_tok/1024:.2f} KB/token")
        if per_tok:
            ratio = slope_kb / (per_tok / 1024)
            print(f"  实测/理论 = {ratio:.2f}")
            if 0.8 <= ratio <= 1.25:
                print("  ✔ 斜率吻合 —— KV cache 公式是对的")
            else:
                print("  ▲ 斜率对不上，说明还有别的随上下文增长的开销")
                print("    （可能是计算缓冲随 batch/ctx 变化，或者 KV 量化）")
        # 固定开销 = 峰值 - 斜率*ctx
        fixed = a["peak_gb"] - slope_gb * a["ctx"]
        print()
        print(f"  推出固定开销 ≈ {fixed:.3f} GB")
        print(f"    其中权重 {info.size_bytes/1024**3:.3f} GB，"
              f"其余（计算缓冲+运行时）{(fixed - info.size_bytes/1024**3):.3f} GB")
        # 顺带记下"非 KV 开销"的均值，供预测器用
        nonkv = [x["non_kv_gb"] for x in good]
        print(f"  各 ctx 的「非 KV 部分」："
              + "、".join(f"{v:.3f}" for v in nonkv))
        print(f"    均值 {sum(nonkv)/len(nonkv):.3f} GB　"
              f"极差 {max(nonkv)-min(nonkv):.3f} GB")
        if max(nonkv) - min(nonkv) > 0.15:
            print("    ▲ 极差偏大 —— 说明「非 KV 部分」不是常数，")
            print("      预测器不能简单用「权重+固定值」，得按缓冲公式算")

    db["measurements"].append({
        "t": time.time(),
        "model": path.name,
        "arch": info.arch,
        "n_layer": info.n_layer,
        "hidden": info.hidden,
        "n_head_kv": info.n_head_kv,
        "kv_dim": info.kv_dim,
        "file_gb": round(info.size_bytes / 1024 ** 3, 4),
        "kv_kb_per_token": round(per_tok / 1024, 2),
        "sys_total_gb": round(mem["total_gb"], 2),
        "threads": args.threads,
        "results": results,
    })
    save_db(db)
    print(f"\n  已存进 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
