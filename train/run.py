import logging
import os
from pathlib import Path
import torch
import random
import numpy as np

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments
from .data import TrainDatasetForEmbedding, EmbedCollator
from .trainer import BiTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, RetrieverTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: RetrieverTrainingArguments
    if model_args.method == "rawMRL":
        from .modeling_rawMRL import BiEncoderModel
    else:
        from .modeling import BiEncoderModel
    # Keep user-selected output paths.
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    seed = 42
    training_args.seed = seed
    set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path,)
    logger.info('Config: %s', config)

    # Import after argument parsing so --help does not require a GPU driver.
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3

    # Patch the backbone kernels before loading the model; no LM loss is used.
    apply_liger_kernel_to_qwen3(
        rope=True,
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )

    model = BiEncoderModel(model_name=model_args.model_name_or_path,
                           sentence_pooling_method=training_args.sentence_pooling_method,
                           temperature=training_args.temperature,
                           query_instruction_for_retrieval=data_args.query_instruction_for_retrieval,
                           passage_instruction_for_retrieval=data_args.passage_instruction_for_retrieval,
                           )

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        processing_class=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
