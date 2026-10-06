# memfit

**内存预算 + 实测校准 —— 这个模型在我这台机器上跑得起来吗？**

跑在 CPU 上、内存紧张的 Windows 机器（本项目在一台 5.95GB、
无独显的 Ryzen 3 2200G 上开发和验证）。

---

## 一、它和现有工具有什么不一样

| 现有工具 | 它们算的是 |
|---|---|
| [llmfit](https://github.com/AlexsJones/llmfit) / [llamafit](https://pypi.org/project/llamafit/0.1.0/) | "这个模型能不能装进你的机器" |
| [gguf-vram-calculator](https://github.com/mrsaynothing/gguf-vram-calculator) | **"能不能装进 VRAM"** |
| [gekro 配置生成器](https://github.com/drajb/gekro) | 按机器给推荐配置 |

**它们都假设你有 GPU**，核心问题是显存够不够。

而 CPU 推理的用户面对的是另一个问题：

- 内存够不够（不是显存）
- **上下文能开多大**（这个几乎没人算）
- 以及最要命的：**那个看不见的运行时开销有多少**

**而且它们只给一个数，不给验证。** memfit 的 `--verify` 会真跑一遍，
把预测和实测摆在一起。预测错了，用户自己能看见。

---

## 二、内存模型（已实测验证）

```
峰值 = 权重 + KV cache + 运行时开销

权重        = GGUF 文件大小                （精确，不用估）
KV cache    = 2 x 层数 x KV头数 x 头维度 x 2字节 x ctx
运行时开销  = 计算缓冲 + 运行时 + 内存碎片   （没有公式，只能实测）
```

### KV cache 公式已经验证

在 1.5B 模型上跑了四个上下文长度，用两点求斜率：

```
      ctx      峰值GB      理论KV       非KV
      512     1.589     0.014     1.576
     2048     1.631     0.055     1.576
     4096     1.686     0.109     1.576
     8192     1.794     0.219     1.575

实测斜率        27.95 KB/token
理论值          28.00 KB/token
实测/理论       1.00          <- 吻合
```

**"非 KV 部分"在四个 ctx 下极差只有 0.001 GB** —— 说明它是常数。

### 那个看不见的开销才是关键

1.5B 的模型文件是 **1.041 GB**，但实际要用 **1.576 GB**：

```
运行时开销 = 1.576 - 1.041 = 0.535 GB
```

**这 0.535 GB 不在模型文件里**，看文件大小完全看不出来。
在一台 5.95GB 的机器上，这 0.5GB 就是"能跑"和"爆内存"的分界线。

### 校准后的精度

```
                    预测      实测      差
1.5B  qwen2.5       1.630    1.630    -0.000     偏差 -0.0%
1.5B  R1-Distill    1.630    1.631    +0.000     偏差 +0.0%
3B    qwen2.5       2.566    2.576    +0.010     偏差 +0.4%
```

---

## 三、我没验证的部分（重要）

**"这个开销跟模型大小无关"这句话，在这台机器上无法证实。**

原因很直接：3B 需要约 2.57GB，而这台机器日常只剩 2.5GB 左右。
一跑就换页，测出来的斜率是理论值的 **7.5 倍**（269 vs 36 KB/token）——
明显失真。工具会把这个测量标成 `UNTRUSTED` 并拒绝用它做校准。

所以：

- **1.5B 在 28 层 / hidden 1536 这个规格上，预测是精确的**
- **其他规格用的是外推值，工具会标 `**GUESS**`**
- 想让它准，就对自己常用的模型跑一次 `--verify`

这不是偷懒，是这类工具唯一诚实的做法：
**没有公式的东西，就别假装有公式。**

---

## 四、用法

```cmd
:: 列出所有模型 + 预测（默认 ctx=2048）
python memfit.py

:: 指定上下文
python memfit.py --ctx 8192

:: 只看一个模型
python memfit.py --model qwen2.5-1.5b-instruct-q4_k_m.gguf

:: 指定模型目录
python memfit.py --dir "E:\你的模型目录"

:: 真跑一遍校准（慢，1.5B 约 10 秒，3B 约 40 秒）
python memfit.py --verify --ctx 2048
```

### 输出长什么样

```
  machine   5.95 GB total   free 2.53 GB   (56% used)

  qwen2.5-1.5b-instruct
    file (=weights)1.041 GB  exact
    KV cache    0.055 GB  28.0 KB/token x 2048
    overhead    0.535 GB  measured here (qwen2.5-1.5b-instruct-q4_k_m.g)
    ------------
    PREDICTED   1.630 GB

    OK  fits with headroom  predicted 1.63GB, free 2.53GB
    max ctx  21392  (15% headroom; trained max 32768)

  qwen2.5-3b-instruct
    file (=weights)1.960 GB  exact
    KV cache    0.070 GB  36.0 KB/token x 2048
    overhead    0.535 GB  borrowed from smallest measured (**GUESS**)
    ------------
    PREDICTED   2.566 GB

    NO  does not fit  predicted 2.57GB > free 2.53GB -> short by 0.04GB
```

`--verify` 会额外打印一张对比表：

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

## 五、校准表怎么工作

存在 `data/calibration.json`。键是**模型结构**（`28L-1536H` = 28 层、
hidden 1536），不是文件名 —— 因为同一个结构的不同微调版开销一样。

**每次可信的实测都会记进去**，用加权平均逐渐收敛：

```
第 1 次测：overhead = 0.535 GB  (n=1)
第 2 次测：overhead = 0.535 GB  (n=2)   <- 两次一致
```

**不可信的测量被拒绝写入。** 这是设计上的硬规则：
换页导致的数据失真，写进去会污染整个校准表。

---

## 六、目录结构

```
E:\memfit\
  memfit.py              主程序（纯 ASCII）
  core\
    gguf.py              GGUF 元数据解析（只读文件头，不加载权重）
    measure.py           实测：启动 llama-server，量峰值内存
  measure_sweep.py       批量扫描：同一个模型测多个 ctx，求斜率
  data\
    calibration.json     校准表（overhead + 测量记录）
    measurements.json    扫描记录
```

**为什么 GGUF 解析要自己写**：读头几百 KB 就能拿到全部结构参数，
不需要把 2GB 权重读进来。现有的 Python 库要么依赖太重，
要么读不了我们需要的字段。

**为什么"可信度"判定这么重要**：见第三节。内存不够的机器上，
测出来的数字是失真的，而且**表面上看起来完全正常** ——
只有斜率跟理论值差 7.5 倍这种对比才能暴露它。

---

## 七、开发时踩的坑

| 坑 | 现象 | 原因 |
|---|---|---|
| **斜率单位算错** | 报 "130.32 MB/token" 又同时说 "0.1 KB/token" | 除以 `1024**2` 而不是 `1024**3`，GB→KB 要乘两次 1024 |
| **冷缓存污染** | 第一次测加载 88 秒，第二次 10 秒 | 第一次读盘慢，页面缓存还没热。要 warmup |
| **上一轮残留** | ctx=512 的峰值比 ctx=2048 还高 | 上一个模型还挂在页面缓存里没回收。要 settle |
| **换页导致斜率反向** | 3B 斜率算出负数 | 内存不够，工作集被换到磁盘又读回，测量完全失真 |
| **只测一次峰值偏低** | 首次推理后峰值还在涨 | 计算缓冲是第一次推理时分配的。要跑两轮 |

**第五条那个"斜率变负"是这篇文章里最有价值的一条**：
数据不会报错，它会**平静地给你一个错误答案**。
只有拿它跟理论值对一下（7.5 倍），才发现不对。

---

## 八、下一步可以做什么

现在只做了"预测 + 校准"。可以加的：

1. **批量校准模式** —— 一次把所有模型都测了，把表填满
2. **推荐参数** —— 反过来解："我要 8K 上下文，该选哪个量化"
3. **历史趋势** —— 同一个模型在不同 llama.cpp 版本下开销的变化
4. **更多机器上的验证** —— 现在只有一个数据点，样本太少
5. **导出报告** —— markdown/html，方便贴到 issue 或博客

---

## 九、一句话

**它把"够不够"从猜测变成测量，并且明确告诉你哪些是测出来的、哪些是猜的。**

在一台内存紧张的机器上，这个区别很实在：
`0.535 GB` 的运行时开销看文件大小永远看不出来，
但它是"能跑"和"爆内存"之间的全部差距。
