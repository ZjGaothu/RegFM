# coding=utf-8
"""Training, evaluation, and model helpers for RegFM."""

import glob
import logging
import os
import random
import re
import shutil
from typing import List

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import PreTrainedTokenizerFast
from transformers import glue_compute_metrics as compute_metrics
from module import TransContextForMaskedLM

from dataset import load_and_cache_examples
from model import CisDNATrans, RegFM

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def set_seed(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit or args.save_total_limit <= 0:
        return

    checkpoints_sorted = sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def build_dna_tokenizer(args):
    dna_tokenizer = Tokenizer.from_file(args.dna_tokenizer_name)
    dna_tokenizer = PreTrainedTokenizerFast(dna_tokenizer)
    dna_tokenizer.kmer = "6"
    dna_tokenizer.add_special_tokens(
        {
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
        }
    )
    return dna_tokenizer


def build_regfm(args, config, dna_config):
    dna_model = CisDNATrans.from_pretrained(
        args.cis_model_name_or_path,
        from_tf=bool(".ckpt" in args.cis_model_name_or_path),
        config=dna_config,
    )
    tf_model = TransContextForMaskedLM.from_pretrained(
        args.trans_model_name_or_path,
        from_tf=bool(".ckpt" in args.cis_model_name_or_path),
        config=config,
    )
    model = RegFM(config)
    model.dna_bert = dna_model.bert
    model.tf_bert = tf_model.bert
    return model


def _register_legacy_pickle_aliases():
    """Register old pickle names only at checkpoint-load time.

    Historical ``modelwhole.pth`` files reference ``longnetmodels`` and class
    names such as ``CrossAttention3``; map them to the current modules/classes
    without exporting those aliases from ``model`` / ``module``.
    """
    import sys
    import model as model_module
    import module as module_module

    sys.modules["longnetmodels"] = model_module
    model_module.LongBertForGenePrediction7168015wNew = model_module.RegFM
    model_module.LongBertForMaskedLM71680 = model_module.CisDNATrans
    model_module.CrossAttention3 = module_module.CrossAttention
    module_module.CrossAttention3 = module_module.CrossAttention
    module_module.GenomicLLMForMaskedLM2103New = module_module.TransContextForMaskedLM
    return model_module


def _unwrap_parallel(model):
    """Return the underlying nn.Module from DataParallel / DDP wrappers."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module

    raw = getattr(model, "__dict__", {})
    if "module" in raw and isinstance(raw["module"], torch.nn.Module):
        return raw["module"]

    modules = getattr(model, "_modules", None)
    if isinstance(modules, dict) and "module" in modules:
        return modules["module"]

    return model


def _load_state_dict_into_regfm(config, state_path, device):
    """Build a fresh RegFM and load ``model.pth`` (handles DDP ``module.`` prefixes)."""
    model = RegFM(config)
    state_dict = torch.load(state_path, map_location="cpu", weights_only=False)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading %s: %s", state_path, missing[:20])
    if unexpected:
        logger.warning("Unexpected keys when loading %s: %s", state_path, unexpected[:20])
    logger.info("Loaded finetuned weights from %s into RegFM", state_path)
    return model


def load_finetuned_checkpoint(checkpoint_dir, device, config=None):
    """Load a finetuned RegFM checkpoint for inference.

    Prefer ``model.pth`` + a freshly constructed ``RegFM`` when ``config`` is
    given. This avoids unpickling historical ``DistributedDataParallel`` objects
    in ``modelwhole.pth``, which often fail across torch / CUDA upgrades.

    Falls back to unwrapping ``modelwhole.pth`` only when ``model.pth`` is absent.
    """
    whole_path = os.path.join(checkpoint_dir, "modelwhole.pth")
    state_path = os.path.join(checkpoint_dir, "model.pth")

    if config is not None and os.path.isfile(state_path):
        return _load_state_dict_into_regfm(config, state_path, device)

    if os.path.isfile(whole_path):
        _register_legacy_pickle_aliases()
        import torch.distributed as dist

        # Unpickling a DDP object may require a process group.
        if dist.is_available() and not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29591")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

        loaded = torch.load(whole_path, map_location="cpu", weights_only=False)
        model = _unwrap_parallel(loaded)
        if type(model).__name__ == "DistributedDataParallel":
            raise RuntimeError(
                "Could not unwrap DistributedDataParallel from modelwhole.pth. "
                "Provide model.pth and pass config to load_finetuned_checkpoint()."
            )
        logger.info("Loaded finetuned model from %s", whole_path)
        return model

    if os.path.isfile(state_path):
        raise FileNotFoundError(
            "Found model.pth but config was not provided. "
            "Call load_finetuned_checkpoint(..., config=config)."
        )

    raise FileNotFoundError(
        "No finetuned checkpoint found under {} (expected model.pth or modelwhole.pth)".format(
            checkpoint_dir
        )
    )


def train(args, train_dataset, model, tokenizer, epi_tokenizer, dna_tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    warmup_steps = int(args.warmup_percent * t_total)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        0,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)

    best_auc = 0
    stop_count = 0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
                "epi_ids": batch[4],
                "dna_ids": batch[5],
                "dna_attention_mask": batch[6],
            }
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, epi_tokenizer, dna_tokenizer)

                        if results["corr"] > best_auc:
                            best_auc = results["corr"]

                        if args.early_stop != 0:
                            if results["corr"] < best_auc:
                                stop_count += 1
                            else:
                                stop_count = 0

                            if stop_count == args.early_stop:
                                logger.info("Early stop")
                                return global_step, tr_loss / global_step

                        for key, value in results.items():
                            logs["eval_{}".format(key)] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    logs["learning_rate"] = scheduler.get_lr()[0]
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    model.eval()
                    torch.save(model.state_dict(), output_dir + "/model.pth")
                    torch.save(model, output_dir + "/modelwhole.pth")
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

                    rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, epi_tokenizer, dna_tokenizer, prefix="", evaluate=True):
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, epi_tokenizer, dna_tokenizer, 3000, evaluate=evaluate
    )

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
                "epi_ids": batch[4],
                "dna_ids": batch[5],
                "dna_attention_mask": batch[6],
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    preds = np.squeeze(preds)
    result = compute_metrics(args.task_name, preds, out_label_ids, None)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        eval_result = prefix + " "
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            eval_result = eval_result + str(result[key])[:5] + " "
        writer.write(eval_result + "\n")

    return result


def predict(args, model, tokenizer, epi_tokenizer, dna_tokenizer, prefix=""):
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)

    pred_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, epi_tokenizer, dna_tokenizer, 30000, evaluate=True
    )

    args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running prediction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", args.pred_batch_size)

    preds = None
    out_label_ids = None

    for batch in tqdm(pred_dataloader, desc="Predicting"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
                "epi_ids": batch[4],
                "dna_ids": batch[5],
                "dna_attention_mask": batch[6],
            }
            outputs = model(**inputs)
            _, logits = outputs[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    preds = np.squeeze(preds)
    result = compute_metrics(args.task_name, preds, out_label_ids)

    output_pred_file = os.path.join(args.predict_dir, "pred_results_%s.npy" % args.save_name)
    logger.info("***** Pred results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    np.save(output_pred_file, preds)


def visual_cross(args, model, tokenizer, epi_tokenizer, dna_tokenizer, prefix=""):
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)

    pred_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, epi_tokenizer, dna_tokenizer, 15000, evaluate=True
    )

    if not os.path.exists(args.predict_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.predict_dir)

    args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(
        pred_dataset,
        sampler=pred_sampler,
        batch_size=args.pred_batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running prediction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", args.pred_batch_size)

    model.eval()
    model.dna_bert.eval()

    reduced_attns = []
    count = 0
    file_index = 1
    pred_output_dir = args.predict_dir

    with torch.inference_mode():
        for batch in tqdm(pred_dataloader, desc="Predicting"):
            count += 1
            batch = tuple(t.to(args.device, non_blocking=True) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
                "epi_ids": batch[4],
                "dna_ids": batch[5],
                "dna_attention_mask": batch[6],
            }
            outputs = model(**inputs)
            _, logits, attn, embed, _ = outputs[:5]
            vec = attn[:, 0, :].cpu().numpy()
            reduced_attns.append(vec)

            if count % 20000 == 0:
                np.save(
                    os.path.join(
                        pred_output_dir,
                        f"pred_attn_part{file_index}_alltok_sumlayer_test_layer4_new_cls.npy",
                    ),
                    np.stack(reduced_attns, axis=0),
                )
                print(f"Saved part {file_index} (count={count})")
                reduced_attns.clear()
                file_index += 1

    if len(reduced_attns) > 0:
        np.save(
            os.path.join(
                pred_output_dir,
                f"pred_attn_part{file_index}_alltok_sumlayer_test_layer4_new_cls.npy",
            ),
            np.stack(reduced_attns, axis=0),
        )
