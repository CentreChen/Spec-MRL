import logging
import os
from pathlib import Path
import shutil
import torch
import random
import numpy as np

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from liger_kernel.transformers import apply_liger_kernel_to_qwen3

from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments
from .data import TrainDatasetForEmbedding, EmbedCollator
from .modeling import BiEncoderModel
from .trainer import BiTrainer

logger = logging.getLogger(__name__)


def backup_code(src_dir, dst_dir):
    """
    将src_dir目录下的所有.py文件拷贝到dst_dir目录
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    # 确保目标目录存在
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # 遍历源目录下的所有.py文件
    for file in src_path.rglob("*.py"):
        # 计算相对路径，以便在目标目录中保持相同的目录结构
        relative_path = file.relative_to(src_path)
        target_file = dst_path / relative_path
        # 确保目标文件的目录存在
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target_file)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, RetrieverTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: RetrieverTrainingArguments
    # 获取目录名称
    current_dir = os.path.dirname(os.path.abspath(__file__))
    method_id = current_dir.split('_')[-1]

    training_args.output_dir = f"./trained_models/{method_id}"
    training_args.logging_dir = os.path.join(training_args.output_dir, "tensorboard_logs")

    # 获取当前执行脚本的绝对路径
    current_script_path = Path(__file__).resolve()
    # 获取项目根目录（假设训练脚本在项目根目录）
    project_root = current_script_path.parent

    backup_code(project_root, Path(training_args.output_dir) / f"{method_id}_code_backup")

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

    # Set seed (仅固定随机源以保证数据顺序/采样可复现；不再强制逐位确定性 kernel)
    seed = 42
    training_args.seed = seed
    set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 允许 cuDNN autotune、放开非确定性快速 kernel（提速；不影响最终效果，仅牺牲逐位复现）
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # 放开 TF32：让残留的 fp32 matmul / cudnn 路径走 TensorCore（bf16 主路径不受影响，近零风险提速）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path,)
    logger.info('Config: %s', config)

    # Liger 逐层融合 kernel（RMSNorm/SwiGLU/RoPE），与 FlashAttention-2 可叠加、数值等价不影响效果。
    # 必须在构造模型（from_pretrained）之前给 Qwen3 类打补丁才生效。
    # 关键：本模型删除了 lm_head、不走 CausalLM 的 CE，故显式关闭 (fused_linear_)cross_entropy；
    # 且 proj_head 用的是 from_pretrained 后、patch 前 clone 的 lm_head 权重，Liger 不影响其初始化。
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
                           mipic_alpha=training_args.mipic_alpha,
                           sia_temperature=training_args.sia_temperature,
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

# torchrun --nproc-per-node 4 --master_port=6667 -m train.run
