import math
import os
import os.path
import random
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Sized, Optional
import concurrent.futures

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import DataArguments


logger = logging.getLogger(__name__)

class TrainDatasetForEmbedding(Dataset):
    def _load_task(self, task: str, task_path: str, args: DataArguments) -> Optional[datasets.Dataset]:
        """在单独的线程中加载单个任务。"""
        try:
            # 这里的 num_proc 会被 datasets 库自动调整（如日志所示）
            num_proc = 1

            if os.path.isdir(task_path):
                temp_dataset = datasets.load_dataset('parquet', data_dir=task_path, split='train', num_proc=num_proc)
            elif task_path.endswith('.parquet'):
                temp_dataset = datasets.load_dataset('parquet', data_files=task_path, split='train', num_proc=num_proc)
            else:
                return None

            if len(temp_dataset) > args.max_example_num_per_dataset:
                # (这里也包含了之前的优化：高效采样)
                temp_dataset = temp_dataset.shuffle(seed=42).select(range(args.max_example_num_per_dataset))
            
            task_column = [task] * len(temp_dataset)
            temp_dataset = temp_dataset.add_column(args.task_column, task_column)

            logger.info(f'Loading Task: {task} === {len(temp_dataset)}')
            return temp_dataset
        except Exception as e:
            logger.warning(f'Failed to load task {task} from {task_path} due to {e}')
            return None

    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        args.task_column = "task_name" if args.task_column is None else args.task_column

        if os.path.isdir(args.train_data):
            train_datasets = []
            tasks_to_load = []
            for task in sorted(os.listdir(args.train_data)[:]):
                if task.startswith('.'):
                    continue
                task_path = os.path.join(args.train_data, task)
                if os.path.isdir(task_path) or task_path.endswith('.parquet'):
                    tasks_to_load.append((task, task_path))

            # 设置合理的线程数
            max_workers = 4
            logger.info(f"Loading {len(tasks_to_load)} tasks in parallel with {max_workers} workers...")

            train_datasets = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务，保持顺序
                futures = [executor.submit(self._load_task, task, path, args) for task, path in tasks_to_load]
                
                # 按提交顺序获取结果，保证确定性
                for future in futures:
                    result = future.result()
                    if result is not None:
                        train_datasets.append(result)

            logger.info("All tasks loaded. Concatenating datasets...")
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            num_proc = 1
            self.dataset = datasets.load_dataset('parquet', data_files=args.train_data, split='train', num_proc=num_proc)

        self.dataset = self.dataset.shuffle(seed=42).select(range(96*4*3000))
        # self.dataset = self.dataset.map(data_process, num_proc=4)

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset) #5770443

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        # if self.args.query_instruction_for_retrieval is not None:
        #     query = self.args.query_instruction_for_retrieval + query

        passages = []

        assert isinstance(self.dataset[item]['pos'], list)
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        return query, passages

@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 512
    passage_max_len: int = 512

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}