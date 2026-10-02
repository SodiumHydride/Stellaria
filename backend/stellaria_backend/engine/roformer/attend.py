# Vendored from Music-Source-Separation-Training by Roman Solovyev (ZFTurbo), MIT licence:
# https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/models/bs_roformer/attend.py
# which builds on BS-RoFormer by Phil Wang (lucidrains), MIT licence. Licence texts: NOTICE.txt.
# Changes: the fused path always uses torch's scaled_dot_product_attention (no `packaging`
# dependency, no deprecated `torch.backends.cuda.sdp_kernel`); torch picks the kernel per device.
# fmt: off
# ruff: noqa

import torch
from torch import nn, einsum
import torch.nn.functional as F

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    INFERENCE_SDPA_BACKENDS = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    _HAS_SDPA_KERNEL = True
except ImportError:
    _HAS_SDPA_KERNEL = False

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash

    def flash_attn(self, q, k, v):
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # inference on cuda: prefer cuDNN attention, fall back to the others if unsupported
        if q.is_cuda and not self.training and _HAS_SDPA_KERNEL:
            try:
                with sdpa_kernel(INFERENCE_SDPA_BACKENDS, set_priority=True):
                    return F.scaled_dot_product_attention(q, k, v)
            except TypeError:  # torch without `set_priority`
                pass

        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.
        )

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
