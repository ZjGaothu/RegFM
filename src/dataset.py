# coding=utf-8
"""Dataset and feature loading for RegFM."""

import logging
import os
from multiprocessing import Pool

import torch
from torch.utils.data import Dataset
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


logger = logging.getLogger(__name__)


class RegFMDataset(Dataset):
    def __init__(self, features, trans_features, dna_features, output_mode, gene_num):
        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        self.all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        self.all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        self.all_trans_ids = torch.tensor([f.input_ids for f in trans_features], dtype=torch.long)
        self.all_dna_ids = torch.tensor([f.input_ids for f in dna_features], dtype=torch.long)
        self.dna_attention_mask = torch.tensor([f.attention_mask for f in dna_features], dtype=torch.long)
        self.gene_num = gene_num

        if output_mode == "classification":
            self.all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            self.all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        else:
            raise ValueError("Invalid output_mode. Must be 'classification' or 'regression'.")

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        dna_idx = idx % self.gene_num
        return (
            self.all_input_ids[idx],
            self.all_attention_mask[idx],
            self.all_token_type_ids[idx],
            self.all_labels[idx],
            self.all_trans_ids[idx],
            self.all_dna_ids[dna_idx],
            self.dna_attention_mask[dna_idx],
        )


def convert_features(args, examples, tokenizer, label_list, output_mode, max_length, evaluate):
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_on_left = False
    pad_token_segment_id = 0

    if args.n_process == 1:
        return convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_length,
            output_mode=output_mode,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,
        )

    n_proc = int(args.n_process)
    if evaluate:
        n_proc = max(int(n_proc / 4), 1)
    print("number of processes for converting feature: " + str(n_proc))

    p = Pool(n_proc)
    indexes = [0]
    len_slice = int(len(examples) / n_proc)
    for i in range(1, n_proc + 1):
        if i != n_proc:
            indexes.append(len_slice * i)
        else:
            indexes.append(len(examples))

    results = []
    for i in range(n_proc):
        results.append(
            p.apply_async(
                convert_examples_to_features,
                args=(
                    examples[indexes[i] : indexes[i + 1]],
                    tokenizer,
                    max_length,
                    None,
                    label_list,
                    output_mode,
                    pad_on_left,
                    pad_token,
                    pad_token_segment_id,
                    True,
                ),
            )
        )
        print(str(i + 1) + " processor started !")

    p.close()
    p.join()

    features = []
    for result in results:
        features.extend(result.get())
    return features


def load_and_cache_examples(args, task, tokenizer, epi_tokenizer, dna_tokenizer, gene_num, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = processors["genepred"]()
    output_mode = output_modes["genepred"]

    def _cache_path(data_dir, max_length, include_model_name=True):
        if args.do_predict or args.do_visualcross or not include_model_name:
            return os.path.join(
                data_dir,
                "cached_{}_{}_{}".format("dev" if evaluate else "train", str(max_length), str(task)),
            )
        return os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate else "train",
                list(filter(None, args.cis_model_name_or_path.split("/"))).pop(),
                str(max_length),
                str(task),
            ),
        )

    use_short_cache = args.do_predict or args.do_visualcross
    cached_features_file = _cache_path(args.tfcr_dir, args.max_seq_length, include_model_name=not use_short_cache)
    epi_cached_features_file = _cache_path(args.exp_dir, args.max_seq_length, include_model_name=not use_short_cache)
    dna_cached_features_file = _cache_path(args.dna_dir, args.max_dna_seq_length, include_model_name=not use_short_cache)

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file, weights_only=False)
    else:
        logger.info("Creating features from dataset file at %s", args.tfcr_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.tfcr_dir) if evaluate else processor.get_train_examples(args.tfcr_dir)
        print("finish loading examples")
        features = convert_features(args, examples, tokenizer, label_list, output_mode, args.max_seq_length, evaluate)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if os.path.exists(dna_cached_features_file):
        logger.info("Loading features from cached file %s", dna_cached_features_file)
        dna_features = torch.load(dna_cached_features_file, weights_only=False)
    else:
        logger.info("Creating features from dataset file at %s", args.dna_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.dna_dir) if evaluate else processor.get_train_examples(args.dna_dir)
        print("finish loading examples")
        dna_features = convert_features(
            args, examples, dna_tokenizer, label_list, output_mode, args.max_dna_seq_length, evaluate
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", dna_cached_features_file)
            torch.save(dna_features, dna_cached_features_file)

    if os.path.exists(epi_cached_features_file):
        logger.info("Loading features from cached file %s", epi_cached_features_file)
        trans_features = torch.load(epi_cached_features_file, weights_only=False)
    else:
        logger.info("Creating features from dataset file at %s", args.exp_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.exp_dir) if evaluate else processor.get_train_examples(args.exp_dir)
        print("finish loading examples")
        trans_features = convert_features(
            args, examples, epi_tokenizer, label_list, output_mode, args.max_seq_length, evaluate
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", epi_cached_features_file)
            torch.save(trans_features, epi_cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    return RegFMDataset(features, trans_features, dna_features, "regression", gene_num)
