"""CPU-runnable framework integration lab (tested with PyTorch 2.8.0+cpu).

This is a Python custom operator composed of torch operations, not a fused CUDA
kernel. It demonstrates schema, fake execution, autograd, opcheck and graph capture.
"""

import torch
from torch import Tensor


def validate(x: Tensor, bias: Tensor) -> None:
    # 用 torch._check 而不是裸 assert：这些检查要在 torch.compile / fake 执行路径下也生效，
    # _check 携带一段延迟求值的报错信息，且不会被优化编译当成可删的断言去掉。
    torch._check(x.dim() == 2, lambda: "x must be a matrix [B,D]")
    torch._check(bias.dim() == 1, lambda: "bias must be a vector [D]")
    # bias 必须与 x 的最后一维对齐，这是广播成立的前提；
    # 形状不匹配时若不在这里拒绝，下面 x + bias 会静默广播成错误的形状而不是报错。
    torch._check(x.shape[1] == bias.shape[0], lambda: "bias width must match x")
    # dtype/device 一并在入口校验：算子自身不负责类型提升，也不做跨设备拷贝。
    if x.dtype not in (torch.float32, torch.float64) or bias.dtype != x.dtype:
        raise TypeError("x and bias must have the same float32/float64 dtype")
    if x.device != bias.device:
        raise ValueError("x and bias must be on the same device")


# torch.library.custom_op 同时做两件事：注册 schema（算子名 op_course::bias_relu 及参数类型）
# 和默认实现。mutates_args=() 声明本算子不改动任何输入——声明错了会让 autograd 与编译器
# 对"输入能否被复用"做出错误假设。
@torch.library.custom_op("op_course::bias_relu", mutates_args=())
def bias_relu(x: Tensor, bias: Tensor) -> Tensor:
    # 默认实现是纯 torch 组合，CPU 即可运行；它是组合算子，不是融合 CUDA kernel。
    validate(x, bias)
    return torch.relu(x + bias)


# fake/meta 版本只推导输出的形状与元信息，不真的算数。
# torch.compile 在追踪阶段需要它来分配中间张量；缺了它图捕获会失败。
@bias_relu.register_fake
def bias_relu_fake(x: Tensor, bias: Tensor) -> Tensor:
    # 即使"只算形状"的路径也要做同样的校验：否则 fake 阶段漏过的坏形状
    # 会拖到真实执行或编译产物里才炸，报错位置更难定位。
    validate(x, bias)
    return torch.empty_like(x)


def setup_context(ctx, inputs, output):
    # 前向是 torch 组合、反向要手写，所以这里把反向需要的输入存下来。
    # 保存 x、bias 是因为 dX 与 dBias 都由 x+bias 的符号决定。
    x, bias = inputs
    ctx.save_for_backward(x, bias)


def backward(ctx, grad_output):
    x, bias = ctx.saved_tensors
    # ReLU 在 x+bias=0 处不可导，这里取右导数：>0 处梯度为 1，其余为 0。
    active = (x + bias) > 0
    # 链式法则：上游梯度乘本层对 x 的局部导数。
    # active 转成 grad_output 的 dtype 再乘，避免 bool 参与乘法时的隐式类型提升出意外。
    grad_x = grad_output * active.to(grad_output.dtype)
    # dBias 是把 grad_x 沿 batch 维求和：bias 被广播到每个 batch，
    # 反向上它的梯度必须把各 batch 的贡献加回去——这一步最容易漏。
    grad_bias = grad_x.sum(dim=0)
    # 返回顺序必须与 forward 的输入顺序严格一致（x, bias）。
    return grad_x, grad_bias


# 把反向挂到算子名上，setup_context 负责保存前向输入。
# 注册后这个算子才能在 requires_grad 的张量上参与训练。
torch.library.register_autograd(
    "op_course::bias_relu", backward, setup_context=setup_context
)


def main():
    # 用 float64：gradcheck 靠有限差分，float32 的舍入噪声可能大于差分步长，
    # 会让本来正确的反向公式也被判失败。
    x = torch.tensor([[-2.0, 1.0], [3.0, -4.0]], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([0.25, 0.5], dtype=torch.float64, requires_grad=True)
    actual = bias_relu(x, bias)
    # 与 torch 组合实现的 reference 对拍，验证 forward 的数学正确性。
    expected = torch.relu(x + bias)
    torch.testing.assert_close(actual, expected)
    print("forward:", actual.detach().tolist())

    # Use values away from the nondifferentiable ReLU kink at x+bias=0.
    # gradcheck 用数值微分对比解析梯度，检查的是"反向公式对不对"，不是前向。
    # 输入刻意避开 x+bias=0，因为 ReLU 在该点不可导，数值差分在那里没有意义。
    assert torch.autograd.gradcheck(bias_relu, (x, bias))
    print("gradcheck: PASS")

    # opcheck 查的是另一组契约：schema 是否注册、fake 是否可用、autograd 是否接好、
    # 能否进图编译——它不检查数值梯度。两类检查互补，缺一不可。
    result = torch.library.opcheck(bias_relu, (x, bias))
    print("opcheck:", result)

    # fullgraph=True 要求整个函数被捕获成一张图，这正是 fake 实现存在的意义。
    # backend="eager" 只验证"能捕获、能跑通"，不声称任何加速。
    compiled = torch.compile(bias_relu, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x, bias), expected)
    print("fullgraph capture (eager backend, no speed claim): PASS")

    # 非连续输入：.t() 得到的转置视图 strides 不是常规布局，
    # 用来确认算子的索引/广播逻辑没有默认内存连续这一隐含假设。
    view = torch.arange(12.0, dtype=torch.float64).reshape(3, 4).t()
    view_bias = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)
    # 先确认它确实不连续，否则这个用例并没有测到想测的东西。
    assert not view.is_contiguous()
    torch.testing.assert_close(bias_relu(view, view_bias), torch.relu(view + view_bias))
    print("noncontiguous input: PASS")

    # 负向用例：喂一个形状不匹配的 bias，确认 validate 会拒绝。
    # 期望走到 except；若没报错而进了 else，说明入口校验形同虚设。
    try:
        bias_relu(x, torch.zeros(3, dtype=x.dtype))
    except RuntimeError as error:
        print("invalid shape rejected:", error)
    else:
        raise AssertionError("expected mismatched bias shape to fail")


if __name__ == "__main__":
    main()
