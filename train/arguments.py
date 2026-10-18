import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    method: str = field(default="SpecMRL", metadata={"choices": ["SpecMRL", "rawMRL"]})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_lora : bool = field(
        default=False, metadata={"help": "Whether to use LoRA for finetuning"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "target_modules in LoraConfig"}
    )
    lora_rank: Optional[int] = field(
        default=8, metadata={"help": "r in LoraConfig"}
    )
    lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "lora_alpha in LoraConfig"}
    )
    lora_dropout: Optional[float] = field(
        default=0, metadata={"help": "lora_dropout in LoraConfig"}
    )



@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )

    task_column: str = field(
        default="task_name", 
        metadata={"help": "the column name of each task data. you can also split the tasks by inputing train_data with different task files or directories"}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    report_to: Optional[List[str]] = field(default_factory=lambda: ["none"])
    run_name: Optional[str] = field(default="anonymous")
    temperature: Optional[float] = field(default=0.01)
    sentence_pooling_method: str = field(default='mean', metadata={"help": "the pooling method, should be cls or mean or lasttoken"})
    gc_chunk_size: int = field(
        default=0,
        metadata={"help": "GradCache sub-batch size (sequences per chunk). <=0 disables GradCache (normal full-batch step). Must run WITHOUT --deepspeed when >0."},
    )
