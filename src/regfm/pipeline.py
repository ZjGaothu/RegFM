"""User-facing RegFM API: download, load, and predict in a few lines."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from transformers import BertConfig, DNATokenizer

from utils import build_dna_tokenizer, load_finetuned_checkpoint


DEFAULT_REPO = "Deku21/RegFM"
DEFAULT_VARIANT = "PBMC"
KNOWN_VARIANTS = ("PBMC", "HVG", "AllProtein")
TRANS_MAX_LEN = 2112
CIS_MAX_LEN = 71680
N_TFCR = 2103
WEIGHT_FILES = ("model.pth", "pytorch_model.bin", "modelwhole.pth")

PathLike = Union[str, os.PathLike]


def _as_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _is_local_dir(path: PathLike) -> bool:
    return os.path.isdir(os.path.expanduser(str(path)))


def _first_existing(*candidates: Path) -> Optional[Path]:
    for path in candidates:
        if path is not None and path.exists():
            return path
    return None


def _download_repo(repo_id: str, variant: str, cache_dir: Optional[str] = None) -> Path:
    from huggingface_hub import snapshot_download

    tokenizers = ["tokenizers/**"]
    config = [f"pretrained_models/{variant}/config.json"]
    local_dir = None
    for weight in ("model.pth", "pytorch_model.bin"):
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            allow_patterns=tokenizers + config + [f"pretrained_models/{variant}/{weight}"],
            ignore_patterns=["**/.DS_Store", "**/optimizer.pt", "**/scheduler.pt", "**/modelwhole.pth"],
        )
        weight_path = Path(local_dir) / "pretrained_models" / variant / weight
        if weight_path.is_file():
            return Path(local_dir)

    return Path(local_dir)


def _resolve_checkpoint_dir(root: Path, variant: str) -> Path:
    found = _first_existing(
        root / "pretrained_models" / variant,
        root / variant,
        root,
    )
    if found is None:
        raise FileNotFoundError(
            "Could not find a checkpoint directory under {}. "
            "Expected pretrained_models/{} or a folder containing model.pth.".format(root, variant)
        )
    if not any((found / name).is_file() for name in WEIGHT_FILES):
        raise FileNotFoundError(
            "No model weights in {} (looked for model.pth / pytorch_model.bin / modelwhole.pth).".format(found)
        )
    return found


def _resolve_tokenizer_root(root: Path, checkpoint_dir: Path) -> Path:
    found = _first_existing(
        root / "tokenizers",
        checkpoint_dir.parent.parent / "tokenizers",
        checkpoint_dir / "tokenizers",
    )
    if found is None:
        raise FileNotFoundError(
            "Could not find tokenizers/ under {}. "
            "The Hub repo should contain tokenizers/{{TFCR,Exp,cisDNA}}.".format(root)
        )
    return found


def _pad(ids, max_len, pad_id):
    ids = list(ids)[:max_len]
    mask = [1] * len(ids) + [0] * (max_len - len(ids))
    return ids + [pad_id] * (max_len - len(ids)), mask


def _tfcr_order_from_vocab(vocab_path: Path):
    with open(vocab_path) as handle:
        tokens = [line.strip().split()[0] for line in handle if line.strip()]
    names = tokens[5:]
    if len(names) != N_TFCR:
        raise ValueError("Expected {} TF/CRs in {}, got {}.".format(N_TFCR, vocab_path, len(names)))
    return names


class RegFM:
    """High-level wrapper: load a pretrained checkpoint and predict expression."""

    def __init__(
        self,
        model,
        dna_tokenizer,
        tf_tokenizer,
        exp_tokenizer,
        tfcr_order,
        device: torch.device,
        trans_max_len: int = TRANS_MAX_LEN,
        cis_max_len: int = CIS_MAX_LEN,
        variant: str = DEFAULT_VARIANT,
    ):
        self.model = model
        self.dna_tokenizer = dna_tokenizer
        self.tf_tokenizer = tf_tokenizer
        self.exp_tokenizer = exp_tokenizer
        self.tfcr_order = tfcr_order
        self.device = device
        self.trans_max_len = trans_max_len
        self.cis_max_len = cis_max_len
        self.variant = variant

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: PathLike = DEFAULT_REPO,
        variant: str = DEFAULT_VARIANT,
        device: Union[str, torch.device] = "auto",
        cache_dir: Optional[str] = None,
        dna_tokenizer: Optional[PathLike] = None,
        trans_tokenizer: Optional[PathLike] = None,
        exp_tokenizer: Optional[PathLike] = None,
        config: Optional[PathLike] = None,
    ):
        """Load weights and tokenizers from Hugging Face or a local folder.

        Args:
            pretrained_model_name_or_path: Hub repo id (e.g. ``Deku21/RegFM``)
                or a local snapshot / checkpoint directory.
            variant: Checkpoint name under ``pretrained_models/``
                (``PBMC``, ``HVG``, or ``AllProtein``).
            device: ``"cuda"``, ``"cpu"``, ``"auto"``, or a ``torch.device``.
                Falls back to CPU if CUDA is requested but unavailable.
        """
        if variant not in KNOWN_VARIANTS:
            raise ValueError(
                "Unknown variant {!r}. Choose one of: {}.".format(variant, ", ".join(KNOWN_VARIANTS))
            )

        device_obj = _as_device(device)
        root = (
            Path(os.path.expanduser(str(pretrained_model_name_or_path))).resolve()
            if _is_local_dir(pretrained_model_name_or_path)
            else _download_repo(str(pretrained_model_name_or_path), variant, cache_dir=cache_dir)
        )

        checkpoint_dir = _resolve_checkpoint_dir(root, variant)
        tokenizer_root = _resolve_tokenizer_root(root, checkpoint_dir)

        dna_tok_path = Path(dna_tokenizer) if dna_tokenizer else tokenizer_root / "cisDNA" / "DNA_tokenizer.json"
        trans_tok_path = Path(trans_tokenizer) if trans_tokenizer else tokenizer_root / "TFCR"
        exp_tok_path = Path(exp_tokenizer) if exp_tokenizer else tokenizer_root / "Exp"
        trans_vocab = trans_tok_path if trans_tok_path.is_file() else trans_tok_path / "vocab.txt"
        config_path = Path(config) if config else _first_existing(
            checkpoint_dir / "config.json",
            tokenizer_root / "Exp" / "config.json",
        )
        if config_path is None:
            raise FileNotFoundError(
                "Could not find a Bert config (checkpoint config.json or tokenizers/Exp/config.json)."
            )

        missing = [
            str(path)
            for path in (dna_tok_path, trans_vocab, exp_tok_path if exp_tok_path.is_file() else exp_tok_path / "vocab.txt")
            if not Path(path).exists()
        ]
        if missing:
            raise FileNotFoundError("Missing tokenizer files: {}".format(", ".join(missing)))

        dna_tok = build_dna_tokenizer(Namespace(dna_tokenizer_name=str(dna_tok_path)))
        tf_tok = DNATokenizer.from_pretrained(str(trans_tok_path))
        exp_tok = DNATokenizer.from_pretrained(str(exp_tok_path))
        tfcr_order = _tfcr_order_from_vocab(trans_vocab)

        bert_config = BertConfig.from_pretrained(str(config_path), num_labels=1, finetuning_task="genepred")
        bert_config.hidden_dropout_prob = 0.1
        bert_config.attention_probs_dropout_prob = 0.1

        model = load_finetuned_checkpoint(str(checkpoint_dir), device_obj, config=bert_config)
        model.to(device_obj)
        model.eval()
        return cls(model, dna_tok, tf_tok, exp_tok, tfcr_order, device_obj, variant=variant)

    def _encode_one(self, dna_seq: str, trans_values: np.ndarray):
        dna_seq = dna_seq.strip().upper()
        trans_values = np.asarray(trans_values).reshape(-1)
        if trans_values.shape[0] != len(self.tfcr_order):
            raise ValueError(
                "trans_expression length must be {}, got {}.".format(len(self.tfcr_order), trans_values.shape[0])
            )

        tf_ids, tf_mask = _pad(
            self.tf_tokenizer.encode(" ".join(self.tfcr_order), add_special_tokens=True),
            self.trans_max_len,
            self.tf_tokenizer.pad_token_id or 0,
        )
        trans_ids, _ = _pad(
            self.exp_tokenizer.encode(" ".join(map(str, map(int, trans_values))), add_special_tokens=True),
            self.trans_max_len,
            self.exp_tokenizer.pad_token_id or 0,
        )
        dna_ids, dna_mask = _pad(
            self.dna_tokenizer.encode(dna_seq, add_special_tokens=True),
            self.cis_max_len,
            self.dna_tokenizer.pad_token_id or 0,
        )
        return {
            "input_ids": torch.tensor([tf_ids], device=self.device),
            "attention_mask": torch.tensor([tf_mask], device=self.device),
            "trans_ids": torch.tensor([trans_ids], device=self.device),
            "dna_ids": torch.tensor([dna_ids], device=self.device),
            "dna_attention_mask": torch.tensor([dna_mask], device=self.device),
        }

    @torch.no_grad()
    def predict_expression(
        self,
        dna_sequence: Union[str, Sequence[str]],
        trans_expression,
    ):
        """Predict gene expression from cis-DNA and TF/CR context.

        Args:
            dna_sequence: DNA string, or a list of strings.
            trans_expression: Length-2103 TF/CR vector, or an array of shape
                ``(N, 2103)``. A single DNA sequence is broadcast across a
                batch of contexts, and vice versa.

        Returns:
            A float for a single query, otherwise a 1-D numpy array.
        """
        single_dna = isinstance(dna_sequence, str)
        dna_list = [dna_sequence] if single_dna else [str(seq) for seq in dna_sequence]

        trans = np.asarray(trans_expression)
        single_trans = trans.ndim == 1
        if trans.ndim == 1:
            trans = trans[None, :]
        elif trans.ndim != 2:
            raise ValueError("trans_expression must have shape (2103,) or (N, 2103).")

        n_dna, n_trans = len(dna_list), trans.shape[0]
        n = max(n_dna, n_trans)
        if n_dna == 1 and n > 1:
            dna_list = dna_list * n
        if n_trans == 1 and n > 1:
            trans = np.repeat(trans, n, axis=0)
        if len(dna_list) != trans.shape[0]:
            raise ValueError(
                "Got {} DNA sequences and {} trans vectors; they must match or one of them must be length 1.".format(
                    n_dna, n_trans
                )
            )

        preds = []
        self.model.eval()
        for seq, vec in zip(dna_list, trans):
            outputs = self.model(**self._encode_one(seq, vec))
            preds.append(float(outputs[0].view(-1)[0].cpu()))

        if single_dna and single_trans:
            return preds[0]
        return np.asarray(preds, dtype=np.float32)

    def __repr__(self):
        return "RegFM(variant={!r}, device={}, n_tfcr={})".format(
            self.variant, self.device, len(self.tfcr_order)
        )
