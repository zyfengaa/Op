# 编译器与 SASS：确认实际生成了什么

源码写的是「我想算什么」，SASS 里才是「硬件实际做什么」。两者经常不一致：循环没展开、分支没谓词化、该用 Tensor Core 却用了标量、寄存器不够而 spill 到 local memory。这一页讲怎么从 CUDA C 一路看到真实指令，以及常见编译选项的陷阱。

> **重要限制声明。**
> 本仓当前环境**没有 `nvcc`，也没有 GPU**（见[能力矩阵](../00-start/environment.md)）。因此本页所述工具链**未经在本仓验证**：命令形式来自官方文档，输出示例是**示意性**的（只表达格式，不是本仓实测值）。所有实际输出、spill 计数、指令名都必须**在有设备的环境里重新确认**，包括工具版本差异带来的指令名变化。无 GPU 时，本页只能作为「知道有这层、知道该查什么」的阅读材料，不能据此填写任何实测数字。

## 1. 从 CUDA C 到 SASS 的路径

~~~text
CUDA C / C++  ──nvcc──▶  PTX  ──ptxas──▶  SASS  ──▶  硬件执行
   （源码）              （虚拟 ISA）        （设备汇编）
~~~

| 层 | 是什么 | 谁生成 | 谁读 |
|---|---|---|---|
| CUDA C / C++ | 你写的源码 | — | 人 |
| PTX | 虚拟指令集，寄存器数量近似无限，跨代可移植 | `nvcc` 前端 | 人（排查编译器行为）|
| SASS | 具体架构的设备汇编，寄存器数量受硬件限制 | `ptxas` | 人（确认实际指令）、硬件 |

关键点：**PTX 不是最终指令。** 它是中间表示，一条 PTX 可能被 `ptxas` 展开成多条 SASS，也可能被融合。要确认「实际用了什么指令、寄存器够不够」，必须看 SASS。

工具分工：

| 工具 | 作用 |
|---|---|
| `nvcc` | 顶层驱动：编译、链接、生成 PTX 与 SASS |
| `ptxas` | PTX → SASS 的汇编器；`-v` 输出寄存器与 spill 统计 |
| `cuobjdump -sass` | 从已编译的目标文件 / 可执行文件里反汇编出 SASS |
| `cuobjdump -ptx` | 同理，导出 PTX |
| `nvdisasm` | 反汇编 cubin / SASS，可加控制流与行号信息 |

常用命令形式（来自官方文档；**命令能否跑通取决于本机是否装了对应 Toolkit**）：

~~~bash
# 直接看一个目标文件的 SASS
nvcc -arch=sm_80 -O3 -cubin kernel.cu -o kernel.cubin
cuobjdump -sass kernel.cubin
nvdisasm kernel.cubin

# 生成并保留 PTX，对比中间表示
nvcc -arch=sm_80 -O3 -ptx kernel.cu -o kernel.ptx

# 让 ptxas 打印寄存器与 spill 统计
ptxas -v -arch=sm_80 kernel.ptx -o kernel.cubin
~~~

`-cubin` 只生成设备代码（cubin），不链接；对分析单个 kernel 很方便。想从已有可执行文件里反汇编，直接用 `cuobjdump -sass ./app`。

## 2. 为什么要看 SASS

profiler 告诉你「慢」，SASS 告诉你「为什么慢的指令级原因」。几个只有 SASS 能确认的问题：

| 想确认 | 在 SASS 里看什么 | 在 `ptxas -v` 输出里看什么 |
|---|---|---|
| 是否真的用了 Tensor Core | 是否出现 `HMMA` / `IMMA` / `QMMA` 之类矩阵乘加指令 | — |
| 是否发生寄存器 spill | 是否出现 `LDL` / `STL`（local load / store）| spill stores / spill loads 字节数 |
| 循环有没有展开 | 循环体是否重复出现、是否有回跳分支 | — |
| 分支有没有谓词化 | 是否被编译成带 `@P` 谓词的单指令，而非跳转 | — |
| 寄存器压力 | 实际使用的寄存器数（`nvdisasm` 可给出）| Used N registers |

**SASS 指令名随架构和工具版本变化。** 例如矩阵乘加指令在不同代上叫法不同，`nvdisasm` 的显示也可能不同。判断时以「本机 `cuobjdump` / `nvdisasm` 实际印出的名字」为准，不要照抄旧文档里的助记符。

## 3. 常见编译选项与其陷阱

| 选项 | 作用 | 陷阱 |
|---|---|---|
| `-O3` | 打开大部分优化：展开、内联、调度 | 降到 `-O0` / `-O1` 会明显减少优化，SASS 变长变慢；通常只在排查编译器问题时才降 |
| `--use_fast_math` | 放松精度换速度（含 `--ftz=true`、`--prec-div=false`、`--prec-sqrt=false`、`--fmad=true`）| **会改变数值结果，属于契约变更**，见下 |
| `-arch=sm_XX` | 指定目标架构，同时生成对应 PTX 与 SASS | 架构不同、指令集不同；指定错误会走 JIT 或根本跑不起来 |
| `-gencode arch=...,code=...` | 精确控制「生成哪个 PTX 版本 + 哪个 SASS 目标」 | 没带目标架构的 SASS 时，运行期才 JIT，首次加载变慢 |
| `-G` | 生成设备调试信息 | **同时关闭大部分优化**，SASS 与发布版完全不同；**不能**用它编译后测性能 |
| `-lineinfo` | 生成行号信息，供 profiler 关联源码 | 基本不改变优化后的指令，是性能分析时该用的那个 |

**`--use_fast_math` 是契约变更，不是纯优化。** 它放松了除法和平方根的精度、打开 flush-to-zero、允许更激进的乘加融合。结果是：更快，但数值结果可能与不带该选项时不同，也可能偏离参考实现。一项优化如果改变了输出，就必须回到[精度契约](../02-foundations/contracts/dtype-and-precision.md)登记它，而不是当成透明加速。

**`-G` 与 `-lineinfo` 的区别是常见坑。** 想在 profiler 里看到源码行号时，用 `-lineinfo`；一旦用了 `-G`，优化被关掉，跑出来的时间既不代表发布版，也不能拿来比较。很多「开了 profiler 就慢好多」的观察，其实是编译选项用错了。

**`-arch` / `-gencode` 与运行环境要对上。** 只把 SASS 编译到 `sm_90`，拿到 `sm_80` 的卡上会报「no kernel image is available」；若同时嵌入了 PTX，驱动会用 JIT 现场编译，能跑但首次加载明显变慢，且 JIT 的结果可能与离线编译不同。报告里必须写清编译的目标架构。

## 4. 怎么判断寄存器 spill

寄存器不够时，编译器把放不下的变量放到 **local memory**（物理上位于全局内存，经 L1/L2 缓存）。这不会报错，只会**静默变慢**。

两条判据：

**1. 看 SASS 里有没有 `LDL` / `STL`。**

~~~text
LDL   R4, [R2+0x10]     ← local load：从 local memory 读
STL   [R2+0x08], R7     ← local store：写回 local memory
~~~

出现这类指令，就说明有变量被放到了 local memory。循环里出现尤其危险——每次迭代都多一趟访问。

**2. 看 `ptxas -v` 的 spill 统计。**

~~~text
ptxas info : Used 56 registers, 2048 bytes smem,
             48 bytes spill stores, 16 bytes spill loads
~~~

（格式示意，具体字段随版本变化。）只要 spill 计数非零，就应追查原因；计数越大越严重。

**注意方向性：** 不要为了满足 occupancy 目标而硬压寄存器，压到 spill 反而更慢。spill 与 occupancy 的权衡见 [occupancy](techniques/occupancy.md)。

## 5. 怎么确认用了 Tensor Core

矩阵乘加指令是 Tensor Core 的直接证据。SASS 里出现：

~~~text
HMMA.16816.F32 ...      ← Half（fp16）矩阵乘加，示意
IMMA ...                ← Integer（整型）矩阵乘加，示意
QMMA ...                ← Quantized 矩阵乘加，示意
~~~

（`HMMA` = Half、`IMMA` = Integer、`QMMA` = Quantized、还有 `BMMA` 等；**确切助记符与形态随架构和工具版本不同**，以本机 `nvdisasm` 输出为准。）PTX 层的对应物是 `wmma` / `mma.sync` / `wgmma` 之类——但 PTX 里出现 `mma` 不代表 SASS 里一定落到矩阵指令，所以最终仍要看 SASS。

两点辨析：

- **SASS 里有 `HMMA`，只说明指令被生成了**，不代表 Tensor Core 利用率高、也不代表它跑到了峰值。利用率要看 profiler（[analysis.md](analysis.md)）。
- **用不用 Tensor Core，还取决于 dtype 与编译/运行开关。** 例如 TF32 在某些框架里默认开启，会让 FP32 路径实际走到矩阵单元上，从而**改变数值结果**；这属于契约，见[精度契约](../02-foundations/contracts/dtype-and-precision.md)。

## 6. 常见误区

**把 PTX 当成 SASS。** PTX 是虚拟 ISA，寄存器近似无限、跨代可移植；SASS 才是真正执行的指令，寄存器受限、与架构绑定。分析性能要落到 SASS。

**用 `-G` 编译后测性能。** `-G` 会关闭大部分优化，结果与发布版无关。要行号就用 `-lineinfo`。

**把 `--use_fast_math` 当纯优化。** 它改变数值结果，是契约变更，必须回归正确性。

**只编译到一个架构就到处跑。** 架构不匹配要么直接失败，要么走 JIT 现场编译，首次加载慢且结果可能与离线编译不同。报告里要写明目标架构。

**看到 `--use_fast_math` / `-O3` 变快就下结论。** 先确认数值是否还在容差内（[精度契约](../02-foundations/contracts/dtype-and-precision.md)），再看速度。

**看到 `HMMA` 就以为到了 Tensor Core 峰值。** 指令在，不代表利用率高。

**看到 `ptxas -v` 寄存器数高就立刻压。** 压到 spill 会更慢。先看有没有 spill，再谈 occupancy。

**在本仓拿本页命令当已验证事实。** 见开头的限制声明：这些工具在本仓**未经验证**，必须在有设备的环境里重新确认。

## 相关页面

- [analysis.md](analysis.md)：从 profiler 数字到瓶颈类型
- [tooling.md](tooling.md)：nsys / ncu / compute-sanitizer 的采集
- [techniques/occupancy.md](techniques/occupancy.md)：寄存器、spill 与 occupancy 的权衡
- [techniques/tensor-core.md](techniques/tensor-core.md)：用上矩阵单元的成立条件
- [../02-foundations/contracts/dtype-and-precision.md](../02-foundations/contracts/dtype-and-precision.md)：`--use_fast_math`、TF32 与精度契约
- [../00-start/environment.md](../00-start/environment.md)：无 `nvcc` / 无 GPU 时的能力边界
- [../99-reference/glossary.md](../99-reference/glossary.md)：SASS、PTX、MMA 等术语
