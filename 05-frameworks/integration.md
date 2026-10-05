# 接口层、设备层和 kernel 层

## 建议分层

~~~text
Public API / schema
    ↓ 参数检查、dispatch、错误信息
Backend adapter
    ↓ 设备、stream、workspace、runtime
Kernel implementation
    ↓ CUDA / Triton / vendor language
Tests and reference
    ↓ backend-independent semantics
~~~

公共 API 定义行为；backend adapter 处理设备特定资源；kernel 层只实现计算。不要让 CUDA 类型泄漏到所有公共接口，也不要为了抽象而隐藏必要设备约束。

## 接口检查项

- shape、stride、dtype、device。
- layout 和对齐要求。
- 输出分配或 out 参数。
- workspace 大小与生命周期。
- stream/queue 和同步规则。
- 原地修改、alias 和错误处理。
- 支持的精度和近似公式。

## 多 backend 的组织

公共层放数学语义、reference、误差阈值、输入生成和性能接口；backend 目录隔离编译、runtime、同步、profiling 和专用指令。warp 宽度、向量宽度、bank 数量不得写成全局假设。

## 案例：把 bias+ReLU 变成可被框架调用的算子

先运行 [框架实验](impl/custom_bias_relu.py)：

~~~powershell
python 05-frameworks/impl/custom_bias_relu.py
~~~

本例使用 Python 自定义算子注册 API，内部计算仍由 PyTorch 完成。这样可以先验证接口与框架边界，再把设备计算替换成专门的 kernel；两个学习问题有各自的测试。

输入契约为 x[B,D]、bias[D]，同 device、同 dtype，支持 float32/float64；输出新分配，输入不修改，允许非连续输入。形状检查放在 validate 中，真实实现与 fake 实现都调用它，防止普通运行接受/拒绝的输入与图模式不一致。

## 注册代码每一部分承担什么责任

~~~python
@torch.library.custom_op("op_course::bias_relu", mutates_args=())
def bias_relu(x: Tensor, bias: Tensor) -> Tensor:
    validate(x, bias)
    return torch.relu(x + bias)
~~~

op_course 是本课程示例命名空间，bias_relu 是算子名。mutates_args=() 声明不修改输入，这是一项框架可以检查和依赖的承诺。内部 torch.relu(x+bias) 会产生新输出；若改成对输入原地修改，就必须同时改变契约与注册信息，不能只改函数体。

fake 实现只通过元数据产生输出描述：本例返回 empty_like(x)。它并不需要真算 ReLU，也不应读取 tensor 值。真实输出和 fake 输出必须在 shape/dtype/device 等约定上兼容，否则图捕获可能失败。

## backward 的参数与返回值怎样对应

setup_context 收到 forward 的输入和输出，本例保存 x/bias。backward 收到输出梯度，返回两个结果，顺序与两个输入一致。若某个参数不可微，一般需要按接口要求返回对应的空梯度，而不是错位返回。

本例先算 active=(x+bias)>0；dX=grad_output*active；dBias=dX.sum(dim=0)。后者的归约来自 bias 的广播。把 dim=0 错写成 dim=1 会在 B=D 时出现极具迷惑性的同形错误，所以测试不应永远只用方阵。

## 真正替换成 CUDA 后还要增加哪些工作

Python 例子对非连续输入的支持来自内部 PyTorch 运算。替换为本仓库的连续行 CUDA kernel 时，必须保留这种能力、显式转换，或收紧公共契约并提供清晰错误，不能让原本合法的输入静默算错。

后端还要从框架取得正确 device 与当前 stream，保持输出/临时存储生命周期。不要在库算子内固定使用某个设备或默认 stream，也不要用全设备同步掩盖异步依赖错误。具体接入 API 取决于所用框架/扩展版本，应按官方 C++/CUDA 自定义算子教程实现。

若为了性能加入形状特化路径，让公共层根据 D、dtype、layout 等条件分派，测试需覆盖每个条件两侧。调度逻辑本身也是算子实现的一部分。

## 练习与验收

将 x 改为 [3,2]，bias 仍为 [2]，对输出求和，手算 dBias 每一列有多少正激活贡献。再传一个 [2,3] 的非连续视图，确认输出语义，而不是仅检查程序没有异常。

交付记录应分别列出 forward 数值测试、gradcheck、opcheck 与图捕获结果。图捕获用 eager backend 只证明能进入图，不代表已进行高性能编译；CPU 注册成功也不证明 CUDA kernel 已实现。

## 参考资料

- [PyTorch dispatcher and custom operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [PyTorch PrivateUse1 backend integration](https://docs.pytorch.org/tutorials/advanced/privateuseone.html)
- [CMake CUDA language](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)
