import json
import os
import random

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


class LongBERTConfig(object):
    def __init__(self, tokenizer=None):
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.max_position_embeddings = 70_000
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.pad_token_id = 0
        self.vocab_size = len(tokenizer) if tokenizer is not None else None
        self.segment_size = [16, 128, 512, 1024, 2048]
        self.dilated_rate = [1, 16, 64, 256, 512]

    def __call__(self):
        return self

    def __str__(self):
        return str(self.__dict__)

    def save_pretrained(self, ckpt="."):
        ckpt = os.path.join(ckpt, "config.json")
        with open(ckpt, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def from_pretrained(cls, ckpt):
        if os.path.isdir(ckpt):
            path = os.path.join(ckpt, "config.json")
        elif os.path.isfile(ckpt):
            path = ckpt
        else:
            path = hf_hub_download(repo_id=ckpt, filename="config.json")
        with open(path, "r") as f:
            config_json = json.load(f)
        return cls.from_dict(config_json)

    @classmethod
    def from_dict(cls, _dict):
        config = cls()
        config.attention_probs_dropout_prob = _dict["attention_probs_dropout_prob"]
        config.hidden_dropout_prob = _dict["hidden_dropout_prob"]
        config.hidden_size = _dict["hidden_size"]
        config.max_position_embeddings = _dict["max_position_embeddings"]
        config.num_attention_heads = _dict["num_attention_heads"]
        config.num_hidden_layers = _dict["num_hidden_layers"]
        config.pad_token_id = _dict["pad_token_id"]
        config.vocab_size = _dict["vocab_size"]
        config.segment_size = _dict["segment_size"]
        config.dilated_rate = _dict["dilated_rate"]
        return config


class Config(object):
    def __init__(self, args):
        # General settings
        self.seed = args.seed
        self.ver = args.ver
        self.use_log = bool(args.use_log)
        self.use_tqdm = bool(args.use_tqdm)
        self.debug = bool(args.debug)
        # Model
        backbone = args.backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.config = LongBERTConfig(self.tokenizer)
        # Data
        self.max_len = args.max_len
        # Training
        self.train_one_part = bool(args.train_one_part)
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.apex = bool(args.apex)
        self.device = torch.device(args.device)
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size
        self.num_workers = os.cpu_count()
        # Optimizer
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.encoder_lr = self.lr
        self.decoder_lr = 1e-3
        self.min_lr = args.min_lr
        self.eps = 1e-6
        self.betas = (0.9, 0.999)
        # Scheduler
        self.scheduler_type = args.scheduler_type
        if self.scheduler_type == "cosine":
            self.num_cycles = 0.5
        self.num_warmup_steps = args.num_warmup_steps
        # Paths
        self.train_data_dir = args.train_data_dir
        self.valid_data_dir = args.valid_data_dir
        self.test_data_dir = args.test_data_dir
        self.output_dir = f"model/{self.ver[:-1]}/{self.ver[-1]}"
        os.makedirs(self.output_dir, exist_ok=True)

    def __str__(self):
        return str(self.__dict__)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
