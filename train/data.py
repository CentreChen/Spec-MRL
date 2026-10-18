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
        try:
            num_proc = 1

            if os.path.isdir(task_path):
                temp_dataset = datasets.load_dataset('parquet', data_dir=task_path, split='train', num_proc=num_proc)
            elif task_path.endswith('.parquet'):
                temp_dataset = datasets.load_dataset('parquet', data_files=task_path, split='train', num_proc=num_proc)
            else:
                return None

            if len(temp_dataset) > args.max_example_num_per_dataset:
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

            max_workers = 4
            logger.info(f"Loading {len(tasks_to_load)} tasks in parallel with {max_workers} workers...")

            train_datasets = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._load_task, task, path, args) for task, path in tasks_to_load]
                
                for future in futures:
                    result = future.result()
                    if result is not None:
                        train_datasets.append(result)

            logger.info("All tasks loaded. Concatenating datasets...")
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            num_proc = 1
            self.dataset = datasets.load_dataset('parquet', data_files=args.train_data, split='train', num_proc=num_proc)

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']

        passages = []

        assert isinstance(self.dataset[item]['pos'], list)
        pos_list = [p for p in self.dataset[item]['pos'] if isinstance(p, str) and p.strip()]
        neg_list = [n for n in self.dataset[item]['neg'] if isinstance(n, str) and n.strip()]

        if not (isinstance(query, str) and query.strip()) or not pos_list:
            logger.warning(f'Invalid query/pos at index {item}, skipping to next example.')
            return self.__getitem__((item + 1) % self.total_len)

        pos = random.choice(pos_list)
        passages.append(pos)

        num_neg = self.args.train_group_size - 1
        if not neg_list:
            logger.warning(f'Empty neg at index {item}, padding negatives with pos.')
            negs = [pos] * num_neg
        elif len(neg_list) < num_neg:
            num = math.ceil(num_neg / len(neg_list))
            negs = random.sample(neg_list * num, num_neg)
        else:
            negs = random.sample(neg_list, num_neg)
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
