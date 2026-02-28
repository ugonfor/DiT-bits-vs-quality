"""
1.58-bit ternary quantization for FLUX transformers.
Reproduces: https://chenglin-yang.github.io/1.58bit.flux.github.io/
Method: absmean quantization (BitNet b1.58 style), weights -> {-1, 0, +1}
Activations remain in full precision (as per the paper).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1} * per-channel scale.
    Per-channel absmean quantization gives best accuracy.
    """

    def __init__(self, linear: nn.Linear, per_channel: bool = True):
        super().__init__()
        W = linear.weight.data.float()  # (out_features, in_features)

        if per_channel:
            scale = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)  # (out, 1)
        else:
            scale = W.abs().mean().clamp(min=1e-8)                      # scalar

        W_q = (W / scale).round().clamp_(-1, 1).to(torch.int8)

        self.register_buffer("weight_q", W_q)                           # int8, memory-efficient
        self.register_buffer("scale", scale.to(linear.weight.dtype))    # original dtype

        self.bias = linear.bias  # full-precision bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.weight_q.to(x.dtype) * self.scale
        return F.linear(x, W_eff, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits=1.58"


def quantize_to_ternary(
    model: nn.Module,
    per_channel: bool = True,
    verbose: bool = False,
) -> nn.Module:
    """
    Replace all nn.Linear layers in `model` with TernaryLinear in-place.
    Returns the modified model.
    """
    replaced = 0

    def _replace(module: nn.Module, prefix: str = ""):
        nonlocal replaced
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, TernaryLinear):
                pass  # already quantized
            elif isinstance(child, nn.Linear):
                tl = TernaryLinear(child, per_channel=per_channel)
                setattr(module, name, tl)
                replaced += 1
                if verbose:
                    print(f"  [ternary] {full_name}: ({child.in_features} -> {child.out_features})")
            else:
                _replace(child, full_name)

    _replace(model)
    print(f"[ternary] Quantized {replaced} Linear layers to 1.58-bit")
    return model


def memory_stats(model: nn.Module) -> dict:
    """Estimate model memory in MB."""
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    total_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "params_mb": total_params / 1024**2,
        "buffers_mb": total_buffers / 1024**2,
        "total_mb": (total_params + total_buffers) / 1024**2,
    }
