import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from datasets import Dataset
from mteb import MTEB
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoConfig, AutoModel

from utils.tasks import ALL_TASKS
from utils.utils import move_to_cuda


def _file_signature(path):
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_aligned_projection_basis(proj_head_path, last_vh_path):
    """Run one float32 power/subspace iteration on GPU and cache its aligned basis."""
    cache_path = os.path.join(
        os.path.dirname(proj_head_path), "proj_power1_gpu_vh.pt"
    )
    source_signature = {
        "proj_head": _file_signature(proj_head_path),
        "last_Vh": _file_signature(last_vh_path),
        "algorithm": "gpu_subspace_iteration_qr_float32_v1",
        "num_iters": 1,
    }

    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu")
        basis = cached.get("Vh") if isinstance(cached, dict) else None
        if (
            isinstance(cached, dict)
            and cached.get("source") == source_signature
            and isinstance(basis, torch.Tensor)
            and basis.ndim == 2
            and basis.shape[0] == basis.shape[1]
        ):
            print(f"Loaded cached GPU-aligned projection basis: {cache_path}")
            return basis

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the aligned one-step power iteration.")

    # mmap avoids eagerly copying the projection head into anonymous RAM.
    proj_head = torch.load(proj_head_path, map_location="cpu", mmap=True)
    last_vh = torch.load(last_vh_path, map_location="cpu", mmap=True)
    if proj_head.ndim != 2:
        raise ValueError(f"proj_head must be 2-D; got shape {tuple(proj_head.shape)}")
    hidden_dim = proj_head.shape[1]
    if tuple(last_vh.shape) != (hidden_dim, hidden_dim):
        raise ValueError(
            f"last_Vh must have shape ({hidden_dim}, {hidden_dim}); "
            f"got {tuple(last_vh.shape)}"
        )

    device = torch.device("cuda")
    print(
        f"Computing one-step aligned projection basis on {torch.cuda.get_device_name(device)}..."
    )
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        a_gpu = proj_head.to(device=device, dtype=torch.float32)
        ata = a_gpu.T @ a_gpu
        del a_gpu, proj_head

        q = last_vh.T.contiguous().to(device=device, dtype=torch.float32)
        del last_vh
        z = ata @ q
        del ata, q
        q, _ = torch.linalg.qr(z)
        del z
        basis = q.T.contiguous().cpu()
        del q

    # Release the temporary A/ATA/QR allocations before loading the encoder on GPU.
    torch.cuda.empty_cache()

    payload = {"source": source_signature, "Vh": basis}
    temp_path = f"{cache_path}.{os.getpid()}.tmp"
    torch.save(payload, temp_path)
    os.replace(temp_path, cache_path)
    print(f"Computed and cached GPU-aligned projection basis: {cache_path}")
    return basis

class EncoderWrapper:
    """SpecMRL multi-dimension encoder.

    The transformer backbone produces the same pooled hidden state regardless of
    the target embedding dimension. All dimensions use prefixes of one GPU-aligned projection, including the
    full dimension. Pooled and projected representations are computed
    once per unique sentence and reused across all requested dimensions.

    MTEB datasets such as MindSmall contain millions of repeated strings. An
    inverse index restores their exact input order without encoding duplicates.
    """

    def __init__(self, model, tokenizer, template="{text}", pool="last", batch_size_per_gpu=32,
                 dims=(1024, 512, 256, 128, 64), projection_basis=None, proj_chunk=8192):
        self.encoder = model
        self.tokenizer = tokenizer
        self.template = template
        self.pool = pool
        self.batch_size_per_gpu = batch_size_per_gpu
        self.max_length = 512
        self.proj_chunk = proj_chunk

        self.encoder = self.encoder.cuda()
        self.encoder.eval()

        # pooled embeds are [N, H], and the filter for a projected dimension is
        # Vh[:dim].T. The one-step GPU-aligned basis is cached by checkpoint.
        if projection_basis is None:
            raise ValueError("projection_basis is required")
        self.hidden_dim = projection_basis.shape[1]

        # The first requested dimension must be the full dimension.
        # Every requested dimension uses the one-step aligned projection.
        self.dims = tuple(dict.fromkeys(dims))
        invalid_dims = [dim for dim in self.dims if dim <= 0 or dim > self.hidden_dim]
        if not self.dims or invalid_dims or self.dims[0] != self.hidden_dim:
            raise ValueError(
                f"First dim must equal hidden dim {self.hidden_dim}, and all dims "
                f"must be in [1, {self.hidden_dim}]; got {dims}, invalid={invalid_dims}."
            )
        self.cur_dim = self.dims[0]

        # Project once at the largest requested dimension.
        # Every smaller dim is the first-d columns of this projection.
        trunc_dims = self.dims[:]
        self.proj_dim = max(trunc_dims) if trunc_dims else None
        self.projection_filter = (
            projection_basis[:self.proj_dim].T.contiguous().cuda()
            if self.proj_dim is not None
            else None
        )

        # key -> (unique pooled [U,H], unique projected [U,P], inverse [N]); CPU only.
        self._embedding_cache = {}
        self.l2_normalize_default = True # overridden per task-type in main
        # PairClassification overrides this to float64 because MTEB 1.4.0 leaks
        # NumPy threshold scalars into task_results and json.dump rejects float32.
        self.numpy_output_dtype = np.float32

        self.gpu_num = torch.cuda.device_count()
        if self.gpu_num > 1:
            self.encoder = nn.DataParallel(self.encoder)

    # ---- dim / cache management -------------------------------------------------

    def set_dim(self, dim):
        assert dim in self.dims, f"dim {dim} not prepared; available: {self.dims}"
        self.cur_dim = dim

    def clear_cache(self):
        self._embedding_cache.clear()

    # ---- backbone forward (dim-independent, cached) -----------------------------

    @torch.no_grad()
    def _pool_backbone(self, sentences, effective_template):
        # Tokenize once, then sort by token length to reduce padding. The inverse
        # permutation restores the pre-sort order after independent pooling.
        templated = [effective_template.format(text=t) for t in sentences]
        tokenized = self.tokenizer(
            templated,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
        )
        lengths = tokenized.pop("length")
        order = sorted(range(len(sentences)), key=lengths.__getitem__)
        dataset = Dataset.from_dict({
            name: [values[i] for i in order]
            for name, values in tokenized.items()
        })

        data_collator = DataCollatorWithPadding(self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size_per_gpu, shuffle=False,
                                 drop_last=False, num_workers=0, collate_fn=data_collator,
                                 pin_memory=True)

        pooled = []
        for batch in tqdm(data_loader):
            batch = move_to_cuda(batch)

            outputs = self.encoder(**batch, use_cache=False)
            if self.pool == "last":
                left_padding = (batch['attention_mask'][:, -1].sum() == batch['attention_mask'].shape[0])
                if left_padding:
                    embeds = outputs.last_hidden_state[:, -1, :]
                else:
                    sequence_lengths = batch['attention_mask'].sum(dim=1) - 1
                    batch_size = outputs.last_hidden_state.shape[0]
                    embeds = outputs.last_hidden_state[torch.arange(batch_size, device=outputs.last_hidden_state.device), sequence_lengths]
            elif self.pool == "mean":
                embeds = (outputs.last_hidden_state * batch['attention_mask'].unsqueeze(-1)).sum(dim=1) / batch['attention_mask'].sum(dim=1).unsqueeze(-1)
            elif self.pool == "cls":
                embeds = outputs.last_hidden_state[:, 0, :]
            else:
                raise ValueError(f"Invalid pool: {self.pool}")

            # Store full-dim pooled embeddings on CPU as float32 for all dims.
            pooled.append(embeds.to(dtype=torch.float32).detach().cpu())

        pooled = torch.cat(pooled, dim=0)
        # Restore the order before length sorting.
        inv = torch.empty(len(order), dtype=torch.long)
        inv[torch.tensor(order)] = torch.arange(len(order))
        return pooled[inv]

    @staticmethod
    def _deduplicate(sentences):
        """Return unique strings and an inverse index preserving input order."""
        unique_sentences = []
        unique_ids = {}
        inverse = np.empty(len(sentences), dtype=np.int64)
        for row, sentence in enumerate(sentences):
            unique_id = unique_ids.get(sentence)
            if unique_id is None:
                # Stored ids are one-based so dict.get() can use None safely.
                unique_id = len(unique_sentences) + 1
                unique_ids[sentence] = unique_id
                unique_sentences.append(sentence)
            inverse[row] = unique_id - 1
        # Most retrieval corpus chunks are already unique. Mark the identity case
        # so materialization can avoid copying the entire embedding matrix.
        if len(unique_sentences) == len(sentences):
            inverse = None
        return unique_sentences, inverse

    # ---- projection to proj_dim (computed once, cached) -------------------------

    @torch.no_grad()
    def _project(self, pooled):
        """Project unique pooled vectors once at the largest projected dimension.

        proj_dim is None when no dim needs truncation -> pooled is the projection.
        """
        if self.proj_dim is None:
            return pooled

        out = []
        for i in range(0, pooled.shape[0], self.proj_chunk):
            chunk = pooled[i:i + self.proj_chunk].cuda(non_blocking=True)
            chunk = chunk @ self.projection_filter
            out.append(chunk.to(dtype=torch.float32).cpu())
        return torch.cat(out, dim=0)

    def _restore_rows(self, unique_result, inverse):
        """Restore repeated rows with bounded temporary CPU memory."""
        if inverse is None:
            return unique_result.astype(self.numpy_output_dtype, copy=False)

        result = np.empty(
            (len(inverse), self.cur_dim), dtype=self.numpy_output_dtype
        )
        bytes_per_row = self.cur_dim * np.dtype(self.numpy_output_dtype).itemsize
        chunk_rows = max(1, (256 * 1024**2) // bytes_per_row)
        for start in range(0, len(inverse), chunk_rows):
            end = min(start + chunk_rows, len(inverse))
            result[start:end] = unique_result[inverse[start:end]]
        return result

    @torch.no_grad()
    def _materialize_dimension(self, unique_source, inverse, l2_normalize):
        """Normalize unique vectors in bounded GPU chunks, then restore repeats."""
        # Classification does not normalize. Avoid a pointless GPU round trip.
        if not l2_normalize:
            return self._restore_rows(unique_source.numpy(), inverse)

        bytes_per_row = self.cur_dim * torch.tensor([], dtype=torch.float32).element_size()
        chunk_rows = max(1, (256 * 1024**2) // bytes_per_row)
        unique_result = np.empty(
            (unique_source.shape[0], self.cur_dim), dtype=np.float32
        )
        for start in range(0, unique_source.shape[0], chunk_rows):
            end = min(start + chunk_rows, unique_source.shape[0])
            chunk = unique_source[start:end].cuda(non_blocking=True)
            if l2_normalize:
                chunk = F.normalize(chunk, p=2, dim=-1)
            unique_result[start:end] = chunk.cpu().numpy()

        return self._restore_rows(unique_result, inverse)

    # ---- MTEB interface ---------------------------------------------------------

    @torch.no_grad()
    def encode(self, sentences, l2_normalize=None, is_query=True, **kwargs) -> np.ndarray:
        if l2_normalize is None:
            l2_normalize = self.l2_normalize_default
        sentences = list(sentences)
        if not sentences:
            return np.empty((0, self.cur_dim), dtype=self.numpy_output_dtype)

        effective_template = self.template if is_query else "{text}"

        # In-process cache key shared across dimensions; hash collisions are possible.
        key = (is_query, effective_template, len(sentences), hash(tuple(sentences)))
        cached = self._embedding_cache.get(key)
        if cached is None:
            unique_sentences, inverse = self._deduplicate(sentences)
            unique_pooled = self._pool_backbone(unique_sentences, effective_template)
            unique_projected = self._project(unique_pooled)
            cached = (unique_pooled, unique_projected, inverse)
            self._embedding_cache[key] = cached
            if len(unique_sentences) != len(sentences):
                print(
                    f"Deduplicated {len(sentences):,} -> "
                    f"{len(unique_sentences):,} sentences "
                    f"({len(unique_sentences) / len(sentences):.2%} unique)."
                )

        _, unique_projected, inverse = cached
        unique_source = unique_projected[:, :self.cur_dim]
        return self._materialize_dimension(unique_source, inverse, l2_normalize)

    @torch.no_grad()
    def encode_queries(self, sentences, **kwargs) -> np.ndarray:
        return self.encode(sentences, is_query=True, **kwargs)

    @torch.no_grad()
    def encode_corpus(self, sentences, **kwargs) -> np.ndarray:
        if sentences and isinstance(sentences[0], dict):
            sentences = [
                (doc["title"] + " " + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in sentences
            ]
        return self.encode(sentences, is_query=False, **kwargs)


def build_template(task_type, task):
    from utils.utils import KaLM_TASK_INSTRUCT
    if task_type in ["Classification", "Clustering"]:
        return f"Instruct: {KaLM_TASK_INSTRUCT[task]} \n Query: {{text}}"
    elif task_type in ["Reranking", "Retrieval"]:
        return f"Instruct: Given a query, retrieve documents that answer the query. \n Query: {{text}}"
    elif task_type in ["STS"]:
        return f"Instruct: Retrieve semantically similar text. \n Query: {{text}}"
    else:
        return f"{{text}}"


def output_folder_for(args, model_name, dim, task_type):
    return os.path.join(args.output_dir, f"dim{dim}", task_type)


def run_task_groups(task_groups, model, args, model_name):
    for task_type, tasks in task_groups:
        # Classification uses unnormalized projected embeddings; other tasks normalize.
        model.l2_normalize_default = False if task_type in ["Classification"] else True
        # PairClassification is small and needs float64 for JSON-compatible NumPy
        # thresholds. Large tasks keep float32 for faster scoring and half the RAM.
        model.numpy_output_dtype = (
            np.float64 if task_type == "PairClassification" else np.float32
        )

        for task in tasks:
            model.template = build_template(task_type, task)

            # Encode and project once for this task; later dimensions reuse prefixes.
            for dim in model.dims:
                model.set_dim(dim)
                output_folder = output_folder_for(args, model_name, dim, task_type)

                evaluation = MTEB(tasks=[task], task_langs=["en", "eng-Latn"])
                evaluation.run(
                    model,
                    eval_splits=["test"],
                    output_folder=output_folder,
                    verbosity=0,
                    overwrite_results=False,
                )

            # free the pooled cache before moving to the next task (bounds memory)
            model.clear_cache()


if __name__ == "__main__":
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--trained_model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dims", type=int, nargs="+", default=[1024, 512, 256, 128, 64],
                        help="Embedding dims to evaluate in one backbone pass.")
    parser.add_argument("--pool", type=str, default="mean")
    parser.add_argument("--output_dir", type=str, default="./output/SpecMRL")
    args = parser.parse_args()

    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    apply_liger_kernel_to_qwen3(rope=True, rms_norm=True, swiglu=True,
                               cross_entropy=False, fused_linear_cross_entropy=False)

    pretrained_model = args.model_name
    model_name = args.model_name
    template = "{text}"

    pool = args.pool

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    config = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
    if args.trained_model_path:
        if args.trained_model_path[-1] == "/":
            args.trained_model_path = args.trained_model_path[:-1]
        model = AutoModel.from_pretrained(args.trained_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        proj_head_path = os.path.join(args.trained_model_path, "proj_head.pt")
        last_vh_path = os.path.join(args.trained_model_path, "last_Vh.pt")
        projection_basis = load_aligned_projection_basis(
            proj_head_path, last_vh_path
        )
    else:
        model = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        raise ValueError(
            "proj_head.pt and last_Vh.pt are required; pass --trained_model_path "
            "pointing to their checkpoint directory."
        )

    model = EncoderWrapper(model=model, tokenizer=tokenizer, template=template, pool=pool,
                           batch_size_per_gpu=args.batch_size, dims=args.dims,
                           projection_basis=projection_basis)
    print(f"Evaluating dims: {model.dims} (hidden dim {model.hidden_dim}) in a single backbone pass per task.")

    run_task_groups(ALL_TASKS, model, args, model_name)

    end_time = time.time()
    print(f"ALL_TASKS done! Total time: {(end_time - start_time) / 60} minutes")
