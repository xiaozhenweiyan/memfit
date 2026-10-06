# -*- coding: utf-8 -*-
r"""
GGUF 元数据解析 —— 只读文件头，不加载权重

为什么需要它
============
要预测"这个模型在我的机器上能不能跑"，必须知道模型的**结构参数**：
层数、hidden size、KV 头数、词表大小、上下文上限、量化类型。
这些全在 GGUF 文件的头部元数据里，读头几百 KB 就够了，
不需要把 2GB 的权重全读进来。

GGUF 格式（简化）
=================
    magic      "GGUF"          4 字节
    version    uint32
    n_tensors  uint64          张量个数
    n_kv       uint64          元数据键值对个数
    --- 然后 n_kv 组 (key, value_type, value) ---
    --- 然后 n_tensors 组张量描述 ---

value_type 是个枚举，要按类型分支读。
字符串是 (uint64 长度, 字节)。
数组是 (type, uint64 个数, 元素...)。

关键字段（llama.cpp 用的键名）
==============================
    general.architecture              llama / qwen2 / qwen3 ...
    <arch>.block_count                层数 n_layer
    <arch>.attention.head_count       Q 头数 n_head
    <arch>.attention.head_count_kv    KV 头数 n_head_kv（GQA 关键）
    <arch>.embedding_length           hidden size
    <arch>.context_length             训练时的上下文上限
    <arch>.feed_forward_length        FFN 维度（算激活值要用）
    tokenizer.ggml.tokens             词表（很大，跳过内容只记个数）

**为什么 KV 头数最重要**：Qwen2.5 用的 GQA，KV 头数远小于 Q 头数。
写错这个，KV cache 会算大好几倍 —— 而 KV cache 恰恰是低内存机器
爆内存的主因。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: GGUF 值类型枚举
GGUF_TYPE = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32",
    5: "int32", 6: "float32", 7: "bool", 8: "string", 9: "array",
    10: "uint64", 11: "int64", 12: "float64",
}

#: 定长类型的 struct 格式
_FIXED = {
    "uint8": ("<B", 1), "int8": ("<b", 1),
    "uint16": ("<H", 2), "int16": ("<h", 2),
    "uint32": ("<I", 4), "int32": ("<i", 4),
    "float32": ("<f", 4), "bool": ("<?", 1),
    "uint64": ("<Q", 8), "int64": ("<q", 8), "float64": ("<d", 8),
}

#: 元数据里可能很大的数组，只记个数不存内容
_BIG_KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges",
             "tokenizer.ggml.token_type", "tokenizer.ggml.scores")


@dataclass
class GgufInfo:
    path: Path
    size_bytes: int = 0
    version: int = 0
    n_tensors: int = 0
    n_kv: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    tensor_types: Dict[str, int] = field(default_factory=dict)
    error: str = ""

    # ------------------------------------------------------------ 便捷取值
    @property
    def arch(self) -> str:
        return str(self.meta.get("general.architecture", "") or "?")

    def _a(self, suffix: str, default: Any = None) -> Any:
        """按架构前缀取字段。"""
        return self.meta.get(f"{self.arch}.{suffix}", default)

    @property
    def n_layer(self) -> int:
        v = self._a("block_count")
        if v is None:
            # 有些模型用不同键名
            for k in ("llama.block_count", "qwen2.block_count"):
                if k in self.meta:
                    v = self.meta[k]
                    break
        return int(v or 0)

    @property
    def n_head(self) -> int:
        return int(self._a("attention.head_count") or 0)

    @property
    def n_head_kv(self) -> int:
        """
        KV 头数。**GQA 的关键**。

        很多模型不显式写这个字段 —— 那说明没做 GQA，
        KV 头数 = Q 头数。别默认成 1，那会算小几十倍。
        """
        v = self._a("attention.head_count_kv")
        return int(v) if v else self.n_head

    @property
    def hidden(self) -> int:
        return int(self._a("embedding_length") or 0)

    @property
    def ffn(self) -> int:
        return int(self._a("feed_forward_length") or 0)

    @property
    def ctx_train(self) -> int:
        return int(self._a("context_length") or 0)

    @property
    def kv_dim(self) -> int:
        """
        KV 每个头的维度。

        通常是 hidden / n_head。有的模型单独给 key_length，
        那就用它（MLA 类结构会不一样）。
        """
        kl = self._a("attention.key_length")
        if kl:
            return int(kl)
        return self.hidden // self.n_head if self.n_head else 0

    @property
    def name(self) -> str:
        n = self.meta.get("general.name")
        if n:
            return str(n)
        return self.path.stem

    def quant(self) -> str:
        """
        猜量化类型。

        不直接读文件里的声明（不一定有），而是**从文件大小反推**：
        知道结构和层数，就能算出 fp16 该多大，实际大小 / 理论大小
        就是大致位宽。这比读元数据可靠 —— 元数据里的
        file_type 经常是 None 或者不准。
        """
        if self.n_layer and self.hidden:
            # 粗略参数量：每层 4*hidden^2（注意力）+ 3*hidden*ffn（FFN）
            per_layer = 4 * self.hidden ** 2 + 3 * self.hidden * (self.ffn or 4 * self.hidden)
            total = per_layer * self.n_layer + 2 * self.hidden * 32000  # 词表粗估
            fp16_bytes = total * 2
            if fp16_bytes > 0:
                ratio = self.size_bytes / fp16_bytes
                bits = ratio * 16
                return f"约 {bits:.1f} bit/权重"
        return "?"

    def summary(self) -> List[Tuple[str, str]]:
        return [
            ("文件", self.path.name),
            ("架构", self.arch),
            ("名字", self.name),
            ("大小", f"{self.size_bytes/1024**3:.2f} GB"),
            ("层数", str(self.n_layer)),
            ("hidden", str(self.hidden)),
            ("FFN", str(self.ffn)),
            ("Q 头 / KV 头", f"{self.n_head} / {self.n_head_kv}"
                            + ("（GQA）" if self.n_head_kv < self.n_head else "")),
            ("KV 头维度", str(self.kv_dim)),
            ("训练上下文", str(self.ctx_train)),
            ("词表", str(self.meta.get("tokenizer.ggml.tokens"
                                       if False else "_n_vocab") or "?")),
            ("张量数", str(self.n_tensors)),
            ("量化（推算）", self.quant()),
        ]


class _Reader:
    """小端二进制读取器。"""

    def __init__(self, f):
        self.f = f

    def read(self, fmt: str, size: int):
        data = self.f.read(size)
        if len(data) != size:
            raise EOFError("文件提前结束了")
        return struct.unpack(fmt, data)[0]

    def u32(self) -> int:
        return self.read("<I", 4)

    def u64(self) -> int:
        return self.read("<Q", 8)

    def string(self) -> str:
        n = self.u64()
        if n > 64 * 1024 * 1024:
            raise ValueError(f"字符串长度异常：{n}")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, vtype: int, key: str = "") -> Any:
        name = GGUF_TYPE.get(vtype)
        if name is None:
            raise ValueError(f"未知值类型 {vtype}")
        if name == "string":
            return self.string()
        if name == "array":
            et = self.u32()
            n = self.u64()
            ename = GGUF_TYPE.get(et, "?")
            # 大数组（词表之类）只记个数，别把几 MB 读进内存
            if key in _BIG_KEYS or n > 100000:
                if ename == "string":
                    for _ in range(n):
                        ln = self.u64()
                        self.f.seek(ln, 1)
                else:
                    fmt, sz = _FIXED.get(ename, ("<B", 1))
                    self.f.seek(sz * n, 1)
                return f"<数组 {ename}[{n}]>"
            return [self.value(et, key) for _ in range(min(n, 4096))]
        fmt, sz = _FIXED.get(name, ("<B", 1))
        return self.read(fmt, sz)


def parse_gguf(path, max_kv: int = 4096) -> GgufInfo:
    """
    只读头部元数据。读失败不抛异常，返回带 error 的 GgufInfo。
    """
    p = Path(path)
    info = GgufInfo(path=p)
    try:
        info.size_bytes = p.stat().st_size
    except Exception as e:
        info.error = f"取不到文件大小：{e}"
        return info

    try:
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                info.error = f"不是 GGUF 文件（magic={magic!r}）"
                return info
            r = _Reader(f)
            info.version = r.u32()
            info.n_tensors = r.u64()
            info.n_kv = r.u64()

            for _ in range(min(info.n_kv, max_kv)):
                key = r.string()
                vtype = r.u32()
                try:
                    val = r.value(vtype, key)
                except Exception as e:
                    info.error = f"读元数据 {key!r} 失败：{e}"
                    break
                info.meta[key] = val
                # 词表个数单独记一下（summary 里要用）
                if key == "tokenizer.ggml.tokens" and isinstance(val, str):
                    info.meta["_n_vocab"] = val

            # 张量描述：名字 + 维度数 + 维度 + 类型 + 偏移
            try:
                for _ in range(min(info.n_tensors, 2000)):
                    tname = r.string()
                    ndim = r.u32()
                    dims = [r.u64() for _ in range(ndim)]
                    ttype = r.u32()
                    r.u64()          # offset
                    info.tensor_types[tname] = ttype
            except Exception:
                pass             # 张量表读不全没关系，元数据已经够了
    except Exception as e:
        info.error = f"{type(e).__name__}: {e}"
    return info


#: GGML 张量类型 -> (每块元素数, 每块字节数)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}


def tensor_bpw(ttype: int) -> Optional[float]:
    """张量类型 -> 每权重平均字节数。用于精确算权重内存。"""
    t = GGML_TYPES.get(ttype)
    if not t:
        return None
    _, n_el, n_bytes = t
    return n_bytes / n_el
