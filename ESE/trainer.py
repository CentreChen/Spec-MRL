from transformers.trainer import *

import contextlib
import torch.distributed as dist

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
from torch.utils.data.distributed import DistributedSampler

class BiTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        now_step = self.state.global_step
        total_steps = self.state.max_steps
        loss = model(**inputs, step = now_step)

        # Log sub-losses to W&B via Trainer if available on the (possibly wrapped) model
        if self.is_world_process_zero():
            try:
                inner_model = model.module if hasattr(model, "module") else model
                self.log(inner_model._last_loss_dict)
            except Exception:
                pass

        return loss
    

    def training_step(self, model, inputs, num_items_in_batch=None):
        """GradCache training step: cache reps, then backprop rep-grads chunk-by-chunk.

        Falls back to the default Trainer.training_step when gc_chunk_size <= 0.
        Multiple backward() calls are incompatible with the DDP/DeepSpeed backward
        hooks, so we run everything on the unwrapped module under no_sync and
        average the resulting gradients across ranks manually (== DDP averaging).
        Requires launching WITHOUT --deepspeed.
        """
        if getattr(self.args, "gc_chunk_size", 0) <= 0:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        core = model.module if hasattr(model, "module") else model
        query, passage = inputs["query"], inputs["passage"]
        chunk = self.args.gc_chunk_size
        accum = self.args.gradient_accumulation_steps
        no_sync = model.no_sync if hasattr(model, "no_sync") else contextlib.nullcontext

        q_chunks = self._split_batch(query, chunk)
        p_chunks = self._split_batch(passage, chunk)

        # ---- Phase 1: collect pooled reps with NO autograd graph (peak = 1 chunk) ----
        with torch.no_grad(), self.accelerator.autocast():
            q_reps = torch.cat([core.encode(c) for c in q_chunks], dim=0)
            p_reps = torch.cat([core.encode(c) for c in p_chunks], dim=0)
        q_reps = q_reps.detach().requires_grad_(True)
        p_reps = p_reps.detach().requires_grad_(True)

        # ---- Loss on the FULL batch -> grad w.r.t. reps (and proj_head, if any) ----
        with no_sync():
            with self.accelerator.autocast():
                loss = core.compute_loss_from_reps(q_reps, p_reps)
            (loss / accum).backward()
        q_grad, p_grad = q_reps.grad, p_reps.grad  # already scaled by 1/accum

        if self.is_world_process_zero():
            try:
                self.log(core._last_loss_dict)
            except Exception:
                pass

        # ---- Phase 2: re-encode each chunk WITH graph, inject cached rep-grads ----
        with no_sync():
            with self.accelerator.autocast():
                offset = 0
                for c in q_chunks:
                    n = c["input_ids"].size(0)
                    reps = core.encode(c)
                    (reps * q_grad[offset:offset + n]).sum().backward()
                    offset += n
                offset = 0
                for c in p_chunks:
                    n = c["input_ids"].size(0)
                    reps = core.encode(c)
                    (reps * p_grad[offset:offset + n]).sum().backward()
                    offset += n

        # ---- Manually average grads across ranks (DDP reducer was bypassed) ----
        if core.world_size > 1:
            grads = [p.grad for p in core.parameters() if p.grad is not None]
            if grads:
                flat = torch._utils._flatten_dense_tensors(grads)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(core.world_size)
                for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                    g.copy_(synced)

        return loss.detach() / accum

    @staticmethod
    def _split_batch(features, chunk_size):
        """Slice a tokenized dict {input_ids, attention_mask, ...} along dim 0,
        then trim each chunk along dim 1 to its own longest real sequence.

        The whole batch is right-padded to the global max length by the collator,
        but within a chunk most rows are shorter. Trimming the trailing all-pad
        columns cuts wasted encoder compute (~2x on this data).

        NOTE: this is NOT bit-exact. Ideally right padding + masked mean pooling
        makes trailing pad columns irrelevant, but Qwen3 under bf16+FA2 shows a
        tiny dependence of a token's hidden state on padded length (most visible
        on near-all-pad rows). Measured effect on pooled reps vs the unpadded
        ground truth is tiny (mean abs diff ~7.0e-3 -> ~7.5e-3, cos >0.9997), well
        below bf16 training noise, so it is accepted as a speed/accuracy trade-off.
        Revert to a plain slice if strict bit-exactness is required.
        """
        n = features["input_ids"].size(0)
        chunks = []
        for i in range(0, n, chunk_size):
            chunk = {k: v[i:i + chunk_size] for k, v in features.items()}
            mask = chunk["attention_mask"]
            # right padding: real tokens occupy the leftmost columns; keep up to the
            # longest real sequence in this chunk (at least 1 to avoid a 0-width tensor)
            keep = int(mask.sum(dim=1).max().item())
            keep = max(keep, 1)
            if keep < mask.size(1):
                chunk = {k: (v[:, :keep] if v.dim() >= 2 and v.size(1) == mask.size(1) else v)
                         for k, v in chunk.items()}
            chunks.append(chunk)
        return chunks

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        g = torch.Generator()
        g.manual_seed(self.args.seed)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "generator": g,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        