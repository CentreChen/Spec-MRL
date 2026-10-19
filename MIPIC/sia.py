"""MIPIC last-layer SIA: paper Eqs. 4-14 and Appendix A.5.

No PIC. Qwen adaptation: a detached masked mean replaces the CLS anchor.
KL teacher remains detached; CKA retains both branches as in the notebook.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class LastLayerSIA(nn.Module):
    def __init__(self, full_dim=1024, temperature=0.05):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('SIA temperature must be finite and positive')
        self.full_dim = full_dim
        self.temperature = temperature
        # Preserve the paper's dimension-to-ratio mapping, not a new four-point schedule.
        self.ratios = {64: .4, 128: .5, 256: .6, 512: .7}
        self.projections = nn.ModuleDict({
            str(d): nn.Linear(d, full_dim, bias=False) for d in self.ratios})

    @staticmethod
    def cka_loss(student, teacher):
        # Center over tokens; equivalent token-Gram formulation avoids D x D work.
        # Match the notebook's FP64 arithmetic and additive epsilon.
        student = student.to(torch.float64)
        teacher = teacher.to(torch.float64)
        student = student - student.mean(0, keepdim=True)
        teacher = teacher - teacher.mean(0, keepdim=True)
        ks, kt = student @ student.T, teacher @ teacher.T
        denom = torch.linalg.vector_norm(ks) * torch.linalg.vector_norm(kt)
        similarity = (ks * kt).sum() / (denom + 1e-8)
        return 1. - similarity

    def forward(self, hidden, mask):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            h = hidden.float()
            valid = mask.bool()
            if not valid.any(dim=1).all():
                raise ValueError('SIA requires at least one valid token per sentence')
            teacher = h.detach()
            anchor = (teacher * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            teacher_scores = (teacher * anchor[:, None]).sum(-1) / math.sqrt(self.full_dim)
            teacher_scores = (teacher_scores / self.temperature).masked_fill(~valid, -torch.inf)
            # User-requested KL-only stabilization; CKA selection retains raw scores.
            normalized_anchor = F.normalize(anchor, dim=-1)
            kl_teacher_scores = (F.normalize(teacher, dim=-1) * normalized_anchor[:, None]).sum(-1)
            kl_teacher_scores = (kl_teacher_scores / (math.sqrt(self.full_dim) * self.temperature)).masked_fill(~valid, -torch.inf)
            teacher_logp = F.log_softmax(kl_teacher_scores, -1)
            # Avoid undefined 0 * (-inf - -inf) on padding in KL(student || teacher).
            teacher_logp = teacher_logp.masked_fill(~valid, 0.)
            teacher_order = torch.argsort(teacher_scores, dim=-1, descending=True, stable=True)
            lengths = valid.sum(1).tolist()
            attention = h.sum() * 0.
            geometry = h.sum() * 0.
            for dim, ratio in self.ratios.items():
                # Normalize AFTER projecting the token to the full-dimensional space.
                projected = F.linear(h[..., :dim], self.projections[str(dim)].weight.float())
                scores = (F.normalize(projected, dim=-1) * normalized_anchor[:, None]).sum(-1) / math.sqrt(self.full_dim)
                scores = (scores / self.temperature).masked_fill(~valid, -torch.inf)
                logp = F.log_softmax(scores, -1)
                prob = logp.exp()
                logp = logp.masked_fill(~valid, 0.)
                attention = attention + (prob * (logp - teacher_logp)).sum(-1).mean()
                losses = []
                for b in range(h.size(0)):
                    n = lengths[b]
                    k = min(n, max(8, math.ceil(ratio * n)))
                    # Same teacher ranking for all widths => nested token sets.
                    indices = teacher_order[b, :k]
                    # Unlike KL targets, the full-width CKA branch is trainable.
                    losses.append(self.cka_loss(h[b, indices, :dim], h[b, indices]))
                geometry = geometry + torch.stack(losses).mean().float()
            return attention, geometry
