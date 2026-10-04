"""CPU-runnable framework integration lab (tested with PyTorch 2.8.0+cpu).

This is a Python custom operator composed of torch operations, not a fused CUDA
kernel. It demonstrates schema, fake execution, autograd, opcheck and graph capture.
"""

import torch
from torch import Tensor


def validate(x: Tensor, bias: Tensor) -> None:
    torch._check(x.dim() == 2, lambda: "x must be a matrix [B,D]")
    torch._check(bias.dim() == 1, lambda: "bias must be a vector [D]")
    torch._check(x.shape[1] == bias.shape[0], lambda: "bias width must match x")
    if x.dtype not in (torch.float32, torch.float64) or bias.dtype != x.dtype:
        raise TypeError("x and bias must have the same float32/float64 dtype")
    if x.device != bias.device:
        raise ValueError("x and bias must be on the same device")


@torch.library.custom_op("op_course::bias_relu", mutates_args=())
def bias_relu(x: Tensor, bias: Tensor) -> Tensor:
    validate(x, bias)
    return torch.relu(x + bias)


@bias_relu.register_fake
def bias_relu_fake(x: Tensor, bias: Tensor) -> Tensor:
    validate(x, bias)
    return torch.empty_like(x)


def setup_context(ctx, inputs, output):
    x, bias = inputs
    ctx.save_for_backward(x, bias)


def backward(ctx, grad_output):
    x, bias = ctx.saved_tensors
    active = (x + bias) > 0
    grad_x = grad_output * active.to(grad_output.dtype)
    grad_bias = grad_x.sum(dim=0)
    return grad_x, grad_bias


torch.library.register_autograd(
    "op_course::bias_relu", backward, setup_context=setup_context
)


def main():
    x = torch.tensor([[-2.0, 1.0], [3.0, -4.0]], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([0.25, 0.5], dtype=torch.float64, requires_grad=True)
    actual = bias_relu(x, bias)
    expected = torch.relu(x + bias)
    torch.testing.assert_close(actual, expected)
    print("forward:", actual.detach().tolist())

    # Use values away from the nondifferentiable ReLU kink at x+bias=0.
    assert torch.autograd.gradcheck(bias_relu, (x, bias))
    print("gradcheck: PASS")

    result = torch.library.opcheck(bias_relu, (x, bias))
    print("opcheck:", result)

    compiled = torch.compile(bias_relu, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x, bias), expected)
    print("fullgraph capture (eager backend, no speed claim): PASS")

    view = torch.arange(12.0, dtype=torch.float64).reshape(3, 4).t()
    view_bias = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)
    assert not view.is_contiguous()
    torch.testing.assert_close(bias_relu(view, view_bias), torch.relu(view + view_bias))
    print("noncontiguous input: PASS")

    try:
        bias_relu(x, torch.zeros(3, dtype=x.dtype))
    except RuntimeError as error:
        print("invalid shape rejected:", error)
    else:
        raise AssertionError("expected mismatched bias shape to fail")


if __name__ == "__main__":
    main()
