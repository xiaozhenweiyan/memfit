# -*- coding: utf-8 -*-
r"""
实测：在真机上跑起来，量真实内存

为什么必须实测，不能只算公式
============================
手算出来 3B 在 ctx=2048 只要 2.03GB（权重 1.96 + KV 0.07），
但这台机器上实测工作集是 **2.93~3.17GB** —— 差了约 1GB。
那 1GB 是计算缓冲、运行时开销、内存碎片。
在 5.95GB 的机器上，**这 1GB 就是"能跑"和"爆内存"的分界线**。

而这个开销：
    * 不算在模型文件里（所以看文件大小没用）
    * 不同 llama.cpp 版本不一样
    * 不同机器/编译选项不一样
    * **没有现成工具帮你算**

所以唯一的办法是：**在这台机器上真跑一遍，量出来。**

怎么把两部分分开
================
    总量 = 固定开销 + 上下文相关部分

    * 固定开销：权重 + 计算缓冲 + 运行时。用小 ctx 测，KV 小到可忽略
    * 上下文相关：主要是 KV cache。用**不同 ctx 各测一次**，
      用两点连线求出斜率，再跟理论 KV/token 对比 ——
      对得上说明模型是对的，对不上说明公式漏了什么

为什么不用 --no-mmap
====================
mmap 让权重从页面缓存映射，工作集统计会比较复杂。
但默认行为就是 mmap，而我们要预测的正是**默认行为下的实际占用**。
所以照默认来，不人为改。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: llama.cpp 可执行文件候选位置
LLAMA_DIRS = [
    Path(r"E:\novel-ai\build\bin"),
    Path(r"E:\小说写作\bin"),
    Path(r"E:\小说写作\llama.cpp\build\bin"),
]


def find_llama(name: str = "llama-server.exe") -> Optional[Path]:
    for d in LLAMA_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
def win_mem() -> Dict[str, float]:
    """整机内存状态（GB）。"""
    import ctypes

    class MEM(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEM()
    m.dwLength = ctypes.sizeof(MEM)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    GB = 1024 ** 3
    return {
        "total_gb": m.ullTotalPhys / GB,
        "avail_gb": m.ullAvailPhys / GB,
        "load_pct": float(m.dwMemoryLoad),
        "commit_gb": (m.ullTotalPageFile - m.ullAvailPageFile) / GB,
    }


def proc_tree_rss(pid: int) -> Tuple[int, int]:
    """
    进程树的常驻内存。返回 (工作集字节, 提交字节)。

    为什么要整个进程树：llama-server 可能派生子进程，
    只看主进程会漏。
    """
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32
    # 收集所有进程的 父->子 关系不方便，先直接累加同名相关进程
    # 这里用 tasklist 拿不到父子，改用 toolhelp 太复杂 ——
    # 实际上 llama-server 是单进程干活，所以只测自己和直接子进程。
    total_ws = 0
    total_commit = 0
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"$p = Get-Process -Id {pid} -EA SilentlyContinue; "
             f"if($p){{ $ws=0; $vm=0; "
             f"Get-Process -EA SilentlyContinue | "
             f"Where-Object {{ $_.Id -eq {pid} }} | "
             f"ForEach-Object {{ $ws += $_.WorkingSet64; $vm += $_.PrivateMemorySize64 }}; "
             f"Write-Output \"$ws $vm\" }}"],
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace")
        parts = (r.stdout or "").strip().split()
        if len(parts) >= 2:
            total_ws = int(parts[0])
            total_commit = int(parts[1])
    except Exception:
        pass
    return total_ws, total_commit


def _ps_mem_of_tree(pid: int) -> Tuple[int, int]:
    """
    用一次 PowerShell 拿整棵进程树的内存（更准）。

    为什么花这个代价：单看主进程会漏掉 llama.cpp 的内存映射部分，
    实测差别能到几百 MB。
    """
    script = (
        "$root = " + str(pid) + ";"
        "$all = Get-CimInstance Win32_Process;"
        "$ids = New-Object System.Collections.ArrayList;"
        "[void]$ids.Add($root);"
        "$changed = $true;"
        "while($changed){"
        "  $changed = $false;"
        "  foreach($p in $all){"
        "    if($ids -contains $p.ParentProcessId -and -not ($ids -contains $p.ProcessId)){"
        "      [void]$ids.Add($p.ProcessId); $changed = $true } } }"
        "$ws=0; $vm=0;"
        "foreach($id in $ids){"
        "  $pr = Get-Process -Id $id -EA SilentlyContinue;"
        "  if($pr){ $ws += $pr.WorkingSet64; $vm += $pr.PrivateMemorySize64 } }"
        "Write-Output \"$ws $vm\""
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                           capture_output=True, text=True, timeout=25,
                           encoding="utf-8", errors="replace")
        parts = (r.stdout or "").strip().split()
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 0


@dataclass
class MeasureResult:
    model: str = ""
    ctx: int = 0
    ngl: int = 0
    threads: int = 0
    batch: int = 0
    ubatch: int = 0
    peak_ws: int = 0            # 峰值工作集（字节）
    peak_commit: int = 0        # 峰值提交
    baseline_ws: int = 0        # 启动前的整机提交
    avail_before_gb: float = 0.0   # 启动前的可用内存（GB）
    min_avail_gb: float = 0.0      # 跑的时候最低可用内存 —— 判断有没有换页
    load_seconds: float = 0.0
    ok: bool = False
    error: str = ""
    sys_total_gb: float = 0.0

    @property
    def peak_gb(self) -> float:
        return self.peak_ws / 1024 ** 3

    @property
    def commit_gb(self) -> float:
        return self.peak_commit / 1024 ** 3

    @property
    def paged(self) -> bool:
        """
        跑的时候系统是不是快没内存了。

        为什么关心：一旦开始换页，工作集会**缩水**（被换到磁盘），
        测出来的峰值反而偏低 —— 数据不可信，得标出来。
        """
        return self.min_avail_gb > 0 and self.min_avail_gb < 0.25

    @property
    def slow_load(self) -> bool:
        """
        加载异常慢也是换页的信号。

        实测参考：1.5B 缓存热了之后 3.5~5 秒；
        3B 在内存紧张时 34~53 秒。差一个数量级。
        """
        return self.load_seconds > 25.0

    def trust(self) -> tuple:
        """
        这次测量可不可信。返回 (可信, 原因)。

        为什么要这个：内存不够的机器上，测出来的数字是**失真的**，
        直接拿去做结论会得出错误的公式。宁可标出来不用。
        3B 那次实测斜率是理论值的 7.5 倍，就是这个情况。
        """
        reasons = []
        if self.paged:
            reasons.append(
                f"跑的时候最低可用内存 {self.min_avail_gb:.2f}GB（换页了）")
        if self.avail_before_gb > 0 and self.peak_gb > 0 \
                and self.avail_before_gb < self.peak_gb * 1.05:
            reasons.append(
                f"启动前可用 {self.avail_before_gb:.2f}GB < 峰值 "
                f"{self.peak_gb:.2f}GB")
        if self.slow_load:
            reasons.append(f"加载 {self.load_seconds:.0f}s（异常慢）")
        return (not reasons), "；".join(reasons)


class Measurer:
    """启动 llama-server，盯着它的内存，跑一次推理，记峰值。"""

    def __init__(self, port: int = 8189, out=print):
        self.port = port
        self.out = out
        self.exe = find_llama("llama-server.exe")

    # ------------------------------------------------------------------
    def _wait_ready(self, proc, timeout: float = 180.0) -> bool:
        """等 /health 通过。"""
        t0 = time.time()
        url = f"http://127.0.0.1:{self.port}/health"
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status in (200, 503):
                        # 503 = 还在加载；200 = 好了
                        if r.status == 200:
                            return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def measure(self, model: Path, ctx: int = 2048, ngl: int = 0,
                threads: int = 4, batch: int = 256, ubatch: int = 256,
                n_predict: int = 8, extra: Optional[List[str]] = None,
                settle: float = 3.0, warmup: bool = True,
                ) -> MeasureResult:
        """
        跑一次测量。

        settle: 启动前等多久，让上一个进程的内存真正被系统回收。
            **这个不能省**（实测踩过）：不等待的话，上一个模型
            还挂在页面缓存里，下一次测出来的峰值会偏高，
            而且 ctx 小的反而比 ctx 大的高 —— 数据自相矛盾。
        warmup: 先空跑一次推理，让计算缓冲分配出来，
            再取峰值。不然第一次推理的峰值会偏低。
        """
        r = MeasureResult(model=model.name, ctx=ctx, ngl=ngl,
                          threads=threads, batch=batch, ubatch=ubatch)
        if self.exe is None:
            r.error = "找不到 llama-server.exe"
            return r

        # 启动前先等系统回收内存（否则上轮的页面缓存会污染本轮）
        if settle > 0:
            time.sleep(settle)

        base = win_mem()
        r.sys_total_gb = base["total_gb"]
        r.baseline_ws = int(base["commit_gb"] * 1024 ** 3)
        r.avail_before_gb = base["avail_gb"]

        cmd = [str(self.exe), "-m", str(model),
               "-c", str(ctx), "-t", str(threads),
               "-b", str(batch), "-ub", str(ubatch),
               "-np", "1", "--port", str(self.port), "--jinja",
               "--no-warmup"]
        if ngl:
            cmd += ["-ngl", str(ngl)]
        if extra:
            cmd += extra

        self.out("    启动：" + " ".join(cmd[1:6]) + " …")
        t0 = time.time()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    creationflags=getattr(
                                        subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as e:
            r.error = f"启动失败：{e}"
            return r

        peak_ws = 0
        peak_commit = 0
        min_avail = 999.0
        stop = threading.Event()

        def poll():
            nonlocal peak_ws, peak_commit, min_avail
            while not stop.is_set():
                ws, vm = _ps_mem_of_tree(proc.pid)
                if ws > peak_ws:
                    peak_ws = ws
                if vm > peak_commit:
                    peak_commit = vm
                m = win_mem()
                if m["avail_gb"] < min_avail:
                    min_avail = m["avail_gb"]
                time.sleep(0.25)

        th = threading.Thread(target=poll, daemon=True)
        th.start()

        try:
            ready = self._wait_ready(proc)
            r.load_seconds = time.time() - t0
            if not ready:
                rc = proc.poll()
                r.error = f"启动超时或退出（返回码 {rc}）"
                return r

            # 跑推理。warmup=True 时跑两次，第二次之后才取峰值 ——
            # 第一次推理会额外分配计算缓冲，只跑一次峰值会偏低
            rounds = 2 if warmup else 1
            for i in range(rounds):
                try:
                    body = json.dumps({
                        "prompt": "你好，简单介绍一下你自己。",
                        "n_predict": n_predict, "temperature": 0.3,
                        "stream": False,
                    }).encode()
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{self.port}/completion", data=body,
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=240) as resp:
                        resp.read()
                    time.sleep(0.8)
                except Exception as e:
                    r.error = f"推理失败（第{i+1}轮）：{type(e).__name__}: {e}"
                    return r

            r.peak_ws = peak_ws
            r.peak_commit = peak_commit
            r.min_avail_gb = min_avail if min_avail < 999 else 0.0
            r.ok = peak_ws > 0
        finally:
            stop.set()
            try:
                proc.terminate()
                proc.wait(timeout=20)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            # 退出后也要等内存回收，不然下一个 ctx 会被污染
            time.sleep(max(2.0, settle))

        return r
