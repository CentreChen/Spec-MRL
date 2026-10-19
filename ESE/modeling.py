import copy
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import math

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
                 ese_compression_size: int = 128,
                 ese_kl_temperature: float = 1.0,
                 query_instruction_for_retrieval: Optional[str] = None,
                 passage_instruction_for_retrieval: Optional[str] = None,
                 ):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", trust_remote_code=True, use_cache=False, torch_dtype=torch.bfloat16)
        unembed = model.lm_head.weight.data.clone()
        self.model = model.model
        del model

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.passage_instruction_for_retrieval = passage_instruction_for_retrieval

        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.ese_compression_size = ese_compression_size
        self.ese_kl_temperature = ese_kl_temperature
        self.config = self.model.config
        self.dim = self.config.hidden_size
        if not 0 < self.ese_compression_size < self.dim:
            raise ValueError(
                f"ese_compression_size must be in (0, {self.dim}), "
                f"got {self.ese_compression_size}"
            )
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
    
    def compute_loss_from_reps(self, q_reps, p_reps):
        """Compute last-layer ESE from already-pooled representations.

        `q_reps` are LOCAL query reps; `p_reps` are LOCAL passage reps and are
        gathered across ranks for the task losses. The PCA teacher is estimated
        from the local query+passage batch, matching the per-rank batch behavior
        of the original ESE implementation. This method is split out so the
        GradCache path can call it on cached leaf tensors.

        Only two task dimensions are optimized: the full representation and one
        compression anchor k. Dimensions such as 512/256/64 remain inference-time
        prefix truncations rather than additional MRL objectives.
        """
        p_reps_local = p_reps
        p_reps_all = self._dist_gather_tensor(p_reps_local)

        local_bs = q_reps.size(0)
        group_size = p_reps_all.size(0) // (local_bs * self.world_size)

        target = torch.arange(local_bs, device=q_reps.device, dtype=torch.long) * group_size
        target = target + self.process_rank * local_bs * group_size

        full_loss = self._infonce_loss(q_reps, p_reps_all, target)

        k = self.ese_compression_size
        slim_loss = self._infonce_loss(q_reps[:, :k], p_reps_all[:, :k], target)

        # The official ESE code builds one batch-level d x d dependency matrix.
        # Query and local passage embeddings are concatenated because both are
        # sentence representations produced by this training batch.
        local_reps = torch.cat((q_reps, p_reps_local), dim=0)
        pca_targets = self.pca_compress(local_reps, k)
        alignment_loss = self.compute_ese_alignment_loss(
            local_reps[:, :k],
            pca_targets,
        )

        loss = full_loss + slim_loss + alignment_loss

        self._last_loss_dict = {
            'loss': loss.detach().item(),
            'loss_NCE_full': full_loss.detach().item(),
            f'loss_NCE_dim_{k}': slim_loss.detach().item(),
            'loss_ESE_alignment': alignment_loss.detach().item(),
        }

        return loss

    def _infonce_loss(self, q_reps, p_reps, target):
        q_reps = F.normalize(q_reps, p=2, dim=-1)
        p_reps = F.normalize(p_reps, p=2, dim=-1)
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        return self.compute_loss(scores.view(q_reps.size(0), -1), target)

    @torch.no_grad()
    def pca_compress(self, reps: torch.Tensor, k: int) -> torch.Tensor:
        """Build the detached k-dimensional ESE PCA teacher in FP32.

        This follows the released ESE implementation rather than conventional
        centered PCA: softmax(M^T M / sqrt(d)) is decomposed and M U_k Sigma_k
        is used as the low-dimensional teacher.
        """
        device_type = reps.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            reps_fp32 = reps.float()
            num_reps = reps_fp32.shape[0]
            dependencies = F.softmax(
                (reps_fp32.T @ reps_fp32)
                / (num_reps * math.sqrt(reps_fp32.shape[-1])),
                dim=-1,
            )
            u, singular_values, _ = torch.svd_lowrank(dependencies, q=k)
            topk_dependencies = u * singular_values.unsqueeze(0)
            return reps_fp32 @ topk_dependencies

    def compute_ese_alignment_loss(
        self,
        slimmed_reps: torch.Tensor,
        pca_targets: torch.Tensor,
    ) -> torch.Tensor:
        """MSE + KL alignment used by the released ESE implementation."""
        slimmed_fp32 = slimmed_reps.float()
        pca_targets = pca_targets.float()
        loss = F.mse_loss(slimmed_fp32, pca_targets)

        temperature = self.ese_kl_temperature
        if temperature > 0:
            loss = loss + F.kl_div(
                F.log_softmax(slimmed_fp32 / temperature, dim=-1),
                F.softmax(pca_targets / temperature, dim=-1),
                reduction='batchmean',
            ) * temperature
        return loss

    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None, step: int = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)
        return self.compute_loss_from_reps(q_reps, p_reps)

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
        whole model to GPU first, which would briefly double GPU memory.
        """
        cpu_state = {k: v.detach().to('cpu') for k, v in self.model.state_dict().items()}
        self.model.save_pretrained(output_dir, state_dict=cpu_state)
