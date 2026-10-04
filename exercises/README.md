# 调试训练场：先看症状，再修改实现

这里是可判定的练习，不是阅读完后自行打勾的问答。只需标准 Python，不需要 GPU。12 个故意写错的版本共有 39 组输入，覆盖 Vector Add、Reduce、Softmax、统计合并、GEMM 协作装载和 Online Attention。

## 先做第一题

从仓库根目录运行：

~~~powershell
python exercises/check.py list
python exercises/check.py show VA01
python exercises/check.py grade VA01
~~~

你会看到整除长度可能通过、末项却为零的失败。打开 [broken.py](fault_injection/broken.py) 的 va01，预测哪些线程被漏掉；在 [student.py](fault_injection/student.py) 中添加对应分支，实现修复，再运行 grade。返回码 0 表示该题全部回归输入通过，1 表示尚未修好。student 初始失败是教学设计，不是课程测试坏了。

需要提示时执行 show VA01 --hint。完成后才阅读 [solutions.py](fault_injection/solutions.py)；参考解重在恢复语义，GPU 的高效实现仍要另做。

修完后读 [故障复盘手册](fault_injection/debrief.md)：逐项检查为什么会错、为什么某些输入掩盖错误、怎样补充回归，以及 Python 通过后还没有证明什么。

## 选题路线

| 案例 | 给学生的症状 | 建议关联 |
|---|---|---|
| VA01 | 非整除末尾保持零 | 网格覆盖 |
| VA02 | 连续输入正常，切片失败 | shape 与 stride |
| VA03 | bias 广播后不同列异常 | 模型投影 epilogue |
| VA04 | 4 元素路径升级后丢尾部 | 向量化合法范围 |
| RD01 | 输入增大，结果像只算一截 | 每线程多元素 |
| RD02 | 全负数的 max 变为 0 | 归约中性值 |
| RD03 | 5 个相同值的平均值不等于它们 | padding 与数学计数 |
| RD04 | 换成非 2 的幂后树形归约漏项 | 算法适用条件 |
| SM01 | 普通负值正常，极负值 NaN | 平移等价性与下溢 |
| LN01 | 线程数变化导致均值变化 | 带 count 的统计合并 |
| GM01 | 输出越界线程退出，合法输出也错 | 协作装载和屏障 |
| AT01 | 注意力输出超出可见 V 范围 | 同步重缩放 m/l/o |

每题先记录“输入、症状、候选原因”，再动代码。修复只对第一组输入奏效不算完成；例如 VA01 还要覆盖空输入和 N<block 的情况。

## 一个修复的形状

student.py 提供统一入口，允许按题号单独替换：

~~~python
def solve(case_id, payload):
    if case_id == "VA01":
        # 在这里实现你验证过的覆盖逻辑，返回输出列表。
        ...
    return broken.solve(case_id, payload)
~~~

不要修改 broken.py 来抹掉原始故障；它用于保留证据。不要把 solutions 的调用作为自己的修复。自动判题能检查输出，但无法判断学习者是否理解，所以提交时还要附上一段解释和一个自己补充的反例。

## 安全范围与覆盖范围

这里的“线程”“共享 tile”多数是 Python 数据流模拟。GM01 用缺失装载重现数值症状，不在真实 GPU 上制造死锁。它能训练索引责任，不能替代 Compute Sanitizer 的竞争/同步检测。

判题比较嵌套形状、有限值和数值容差，NaN 不会被误判通过。自检模式还用手算常数核对每题第一个参考答案：

~~~powershell
python exercises/check.py selftest
python exercises/check.py grade all --impl solution
~~~

selftest 要求每个故意错误版本至少被一个输入击中，并检查参考实现的手算锚点。它不把学生尚未完成的 TODO 当作仓库回归失败。

## 换一种学习节奏

接着做 [陌生代码阅读题](reading/README.md)，再回到 [Vector Add 五件套](../01-hello-world-ops/01-vector-add/case-study.md) 或 [Reduce 五件套](../01-hello-world-ops/03-reduction-sum-max/case-study.md)。最后在 [模型切片](../06-model-slice/README.md) 中观察这些错误会破坏哪个阶段。

## 参考资料

- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)：理解真实设备上应追加哪些检查。
- [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)：核对协作范围与线程索引。
