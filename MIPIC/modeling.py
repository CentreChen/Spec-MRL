import copy
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import math
import json
from .sia import LastLayerSIA

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor, nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


class BiEncoderModel(nn.Module):
    def __init__(self,
                 model_name: str = None,
                 sentence_pooling_method: str = 'mean',
                 temperature: float = 1.0,
                 query_instruction_for_retrieval: Optional[str] = None,
                 passage_instruction_for_retrieval: Optional[str] = None,
                 mipic_alpha: float = 0.5,
                 sia_temperature: float = 0.05,
                 ):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", trust_remote_code=True, use_cache=False, torch_dtype=torch.bfloat16)
        unembed = model.lm_head.weight.data.clone()
        self.model = model.model
        del model

        self.dims = [1024, 512, 256, 128, 64]
        self.weights = [1, 0.3, 0.2, 0.1, 0.1]

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.passage_instruction_for_retrieval = passage_instruction_for_retrieval

        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.config = self.model.config
        self.dim = self.config.hidden_size
        if self.dim != 1024 or sentence_pooling_method != 'mean':
            raise ValueError('This MIPIC adaptation requires D=1024 and mean pooling')
        if not math.isfinite(mipic_alpha) or not 0 < mipic_alpha < 1:
            raise ValueError('mipic_alpha must be between 0 and 1 (exclusive)')
        self.mipic_alpha = mipic_alpha
        self.sia = LastLayerSIA(self.dim, sia_temperature)
        sidecar = os.path.join(model_name, 'sia.pt')
        if os.path.isfile(sidecar):
            self.sia.load_state_dict(torch.load(sidecar, map_location='cpu', weights_only=True))


        if dist.is_available() and dist.is_initialized():
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.process_rank = 0
            self.world_size = 1

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1), dim=1)
            d = mask.sum(axis=1, keepdim=True)
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'last':
            left_padding = (mask[:, -1].sum() == mask.shape[0])
            if left_padding:
                return hidden_state[:, -1]
            else:
                sequence_lengths = mask.sum(dim=1) - 1
                batch_size = hidden_state.shape[0]
                return hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths]
        else:
            raise NotImplementedError(f"Pooling method {self.sentence_pooling_method} not implemented")

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps_ori = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        # p_reps = torch.nn.functional.normalize(p_reps_ori, dim=-1)
        return p_reps_ori.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def encode_with_sia(self, features):
        hidden = self.model(**features, return_dict=True).last_hidden_state
        reps = self.sentence_embedding(hidden, features['attention_mask']).contiguous()
        att, cka = self.sia(hidden, features['attention_mask'])
        return reps, att, cka
    
    def get_Vh(self, A: torch.Tensor, num_iters: int = 1):
        # U, S, Vh = torch.linalg.svd((A.T@A).float())
        try:
            with torch.amp.autocast('cuda', enabled=False):
                A_float32 = A.float()
                ATA = A_float32.T @ A_float32
                Q = self.last_Vh.T.to(A.device)
                for _ in range(num_iters):
                    Z = ATA @ Q
                    Q, _ = torch.linalg.qr(Z)
                Vh = Q.T
        except Exception as e:
            print(f"SVD failed with error: {e}, using last Vh")
            print(A)
            Vh = self.last_Vh

        return Vh.to(A.device)


    def compute_loss_from_reps(self, q_reps, p_reps):
        """Compute alpha-weighted original MRL loss from pooled reps.

        `q_reps` are LOCAL query reps; `p_reps` are LOCAL passage reps and are
        gathered across ranks here. Split out from `forward` so the GradCache
        path can call it on cached leaf tensors. Token-level SIA is computed
        separately during encoder replay, never from these pooled leaf tensors.
        """
        # q_reps = self._dist_gather_tensor(q_reps)
        p_reps = self._dist_gather_tensor(p_reps)

        local_bs = q_reps.size(0)
        group_size = p_reps.size(0) // (local_bs * self.world_size)

        target = torch.arange(local_bs, device=q_reps.device, dtype=torch.long) * group_size
        target = target + self.process_rank * local_bs * group_size


        loss_NCE = 0.0
        for i in range(len(self.dims)):
            sub_dim = self.dims[i]
            q_reps_part = q_reps[:, :sub_dim]
            p_reps_part = p_reps[:, :sub_dim]

            scores_part = self.compute_similarity(F.normalize(q_reps_part, p=2, dim=-1), F.normalize(p_reps_part, p=2, dim=-1)) / self.temperature
            scores_part = scores_part.view(q_reps.size(0), -1)

            loss_NCE += self.weights[i] * self.compute_loss(scores_part, target)

        self._last_loss_dict = {'loss_NCE': loss_NCE.detach().item()}

        return self.mipic_alpha * loss_NCE

    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None, step: int = None):
        q_reps, q_att, q_cka = self.encode_with_sia(query)
        p_reps, p_att, p_cka = self.encode_with_sia(passage)
        loss = self.compute_loss_from_reps(q_reps, p_reps)
        att, cka = (q_att + p_att) / 2, (q_cka + p_cka) / 2
        loss = loss + (1 - self.mipic_alpha) * (att + cka)
        self._last_loss_dict.update(loss_SIA_att=att.detach().item(),
                                    loss_SIA_cka=cka.detach().item(), loss=loss.detach().item())
        return loss

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

        
    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        """
        Save model weights.

        Move the state_dict tensor-by-tensor to CPU instead of deep-copying the
        whole model to GPU first: a full `copy.deepcopy(self.model)` briefly
        doubles GPU memory (a second 4B copy) and can OOM at save time, whereas
        streaming the state_dict to CPU keeps the peak flat. proj_head is a
        Parameter on self.model, so it is included in the state_dict just as the
        old deepcopy path saved it; last_Vh / proj_head are additionally dumped
        as standalone files for downstream projection use.
        """
        cpu_state = {k: v.detach().to('cpu') for k, v in self.model.state_dict().items()}
        self.model.save_pretrained(output_dir, state_dict=cpu_state)
        torch.save({k: v.detach().cpu() for k, v in self.sia.state_dict().items()},
                   os.path.join(output_dir, 'sia.pt'))
        with open(os.path.join(output_dir, 'mipic_config.json'), 'w') as f:
            json.dump(dict(method='MIPIC-last-layer-SIA-adapted-v2', alpha=self.mipic_alpha,
                           sia_temperature=self.sia.temperature, dims=self.dims, weights=self.weights,
                           ratios=self.sia.ratios, anchor='detached_masked_mean',
                           kl_teacher_detached=True, cka_teacher_detached=False,
                           cka_dtype='float64', cka_eps=1e-8, PIC=False), f, indent=2)
