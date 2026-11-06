"""Exact full-softmax cross-entropy over a large item table without materializing all the logits.

For states h [N, d] and an output table W [V, d], the loss is sum_i (logsumexp_j h_i . W_j - h_i . W_{y_i}),
the same quantity as F.cross_entropy(h @ W.T, y, reduction="sum"). Over 329,445 Electronics items, a batch of
a few thousand positions would need several GB of logits, and autograd would keep them for the backward pass.

This function walks over the rows in chunks. For each chunk it computes the logits in float32, their
logsumexp, and, because the loss is the last node of the graph, its gradient right away:
softmax(logits) - onehot(y) gives dL/dh = G W and dL/dW += G^T h. Only one chunk of logits exists at a time,
and the backward pass only scales the stored gradients by the incoming gradient. The arithmetic is the plain
softmax cross-entropy; nothing is sampled or approximated. Liger-Kernel's fused linear cross-entropy uses the
same idea.
"""

from __future__ import annotations

import torch

CHUNK_ELEMENTS = 1 << 28  # logits per chunk: 2^28 float32 values are 1 GiB


class _FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h: torch.Tensor, w: torch.Tensor, target: torch.Tensor, chunk: int) -> torch.Tensor:
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_h = torch.empty_like(h) if need_h else None
        grad_w = torch.zeros(w.shape, dtype=torch.float32, device=w.device) if need_w else None
        loss = torch.zeros((), dtype=torch.float32, device=h.device)
        for s in range(0, h.shape[0], chunk):
            hc, tc = h[s : s + chunk], target[s : s + chunk]
            logits = (hc @ w.T).float()
            lse = torch.logsumexp(logits, dim=1)
            loss += (lse - logits.gather(1, tc[:, None]).squeeze(1)).sum()
            if need_h or need_w:
                probs = logits.sub_(lse[:, None]).exp_()  # softmax, in place of the logits
                probs[torch.arange(len(tc), device=tc.device), tc] -= 1.0
                g = probs.to(h.dtype)
                if need_h:
                    grad_h[s : s + chunk] = g @ w
                if need_w:
                    grad_w.addmm_(g.T.float(), hc.float())
        ctx.save_for_backward(grad_h, grad_w)
        ctx.w_dtype = w.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        grad_h, grad_w = ctx.saved_tensors
        gh = grad_h * grad_loss.to(grad_h.dtype) if grad_h is not None else None
        gw = (grad_w * grad_loss).to(ctx.w_dtype) if grad_w is not None else None
        return gh, gw, None, None


def fused_linear_cross_entropy(
    h: torch.Tensor, w: torch.Tensor, target: torch.Tensor, chunk: int | None = None
) -> torch.Tensor:
    """Summed cross-entropy of the rows of ``h @ w.T`` against ``target`` (indices into the rows of w)."""
    if chunk is None:
        chunk = max(1, CHUNK_ELEMENTS // w.shape[0])
    return _FusedLinearCrossEntropy.apply(h, w, target, chunk)
