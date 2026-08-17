# coding=utf-8
"""Fine-tuning RegFM for gene expression prediction."""

import argparse
import logging
import os
from datetime import timedelta

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_TIMEOUT"] = "1800"

import torch

torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from transformers import BertConfig, DNATokenizer
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from dataset import load_and_cache_examples
from utils import (
    build_dna_tokenizer,
    build_regfm,
    evaluate,
    load_finetuned_checkpoint,
    predict,
    set_seed,
    train,
    visual_cross,
)


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # data / model paths
    parser.add_argument("--tfcr_dir", default=None, type=str, required=True, help="TF chromatin region input data dir.")
    parser.add_argument("--dna_dir", default=None, type=str, required=True, help="DNA input data dir.")
    parser.add_argument("--exp_dir", default=None, type=str, required=True, help="Expression input data dir.")
    parser.add_argument("--cis_model_name_or_path", default=None, type=str, required=True, help="Path to cis-DNA pretrained model.")
    parser.add_argument("--trans_model_name_or_path", default=None, type=str, required=True, help="Path to TF/trans pretrained model.")
    parser.add_argument("--exp_config_name", default="", type=str, required=True, help="Expression config name or path.")
    parser.add_argument("--dna_config_name", default="", type=str, required=True, help="DNA config name or path.")
    parser.add_argument("--trans_tokenizer_name", default="", type=str, required=True, help="TF/trans tokenizer name or path.")
    parser.add_argument("--exp_tokenizer_name", default="", type=str, required=True, help="Expression tokenizer name or path.")
    parser.add_argument("--dna_tokenizer_name", default="", type=str, required=True, help="DNA tokenizer name or path.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="Task name selected in the list: " + ", ".join(processors.keys()),
    )

    # modes
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction.")
    parser.add_argument("--do_visualcross", action="store_true", help="Whether to extract cross-attention.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Evaluate during training.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory.")

    # sequence / data processing
    parser.add_argument("--max_seq_length", default=128, type=int, help="Max TF/expression sequence length.")
    parser.add_argument("--max_dna_seq_length", default=128, type=int, help="Max DNA sequence length.")
    parser.add_argument("--n_process", default=2, type=int, help="Number of processes used for data processing.")

    # training hyperparameters
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Train batch size per GPU.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Eval batch size per GPU.")
    parser.add_argument("--per_gpu_pred_batch_size", default=8, type=int, help="Predict batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Hidden dropout.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total training epochs.")
    parser.add_argument("--max_steps", default=-1, type=int, help="Override num_train_epochs if > 0.")
    parser.add_argument("--warmup_percent", default=0, type=float, help="Linear warmup percent of total steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X update steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X update steps.")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Max number of checkpoints to keep.")
    parser.add_argument("--early_stop", default=0, type=int, help="Early stop patience (0 disables).")

    # predict
    parser.add_argument("--predict_dir", default=None, type=str, help="Output directory for prediction.")
    parser.add_argument("--save_name", type=str, default="", help="Name for saving prediction result.")

    # distributed
    parser.add_argument("--local-rank", type=int, default=-1, help="Distributed training local rank.")

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=480))
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    config = BertConfig.from_pretrained(
        args.exp_config_name,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.attention_probs_dropout_prob = 0.1

    dna_config = BertConfig.from_pretrained(
        args.dna_config_name,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    dna_config.vocab_size = 261
    dna_config.max_position_embeddings = 512
    dna_config.hidden_dropout_prob = args.hidden_dropout_prob
    dna_config.attention_probs_dropout_prob = 0.1

    tokenizer = DNATokenizer.from_pretrained(args.trans_tokenizer_name)
    epi_tokenizer = DNATokenizer.from_pretrained(args.exp_tokenizer_name)
    dna_tokenizer = build_dna_tokenizer(args)

    model = None
    if args.do_train:
        model = build_regfm(args, config, dna_config)
        print(model)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        logger.info("finish loading model")

    if args.local_rank == 0:
        torch.distributed.barrier()

    if model is not None:
        model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, epi_tokenizer, dna_tokenizer, 3000, evaluate=False
        )
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, epi_tokenizer, dna_tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(model.state_dict(), args.output_dir + "/model.pth")
        torch.save(model, args.output_dir + "/modelwhole.pth")
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        model.to(args.device)

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        logger.info("Evaluate the following checkpoint: %s", args.output_dir)
        model = load_finetuned_checkpoint(args.output_dir, args.device, config=config)
        model.to(args.device)
        results = evaluate(args, model, tokenizer, epi_tokenizer, dna_tokenizer)

    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("Predict using the following checkpoint: %s", args.output_dir)
        model = load_finetuned_checkpoint(args.output_dir, args.device, config=config)
        model.to(args.device)
        predict(args, model, tokenizer, epi_tokenizer, dna_tokenizer)

    if args.do_visualcross and args.local_rank in [-1, 0]:
        logger.info("Visualcross using the following checkpoint: %s", args.output_dir)
        model = load_finetuned_checkpoint(args.output_dir, args.device, config=config)
        model.to(args.device)
        visual_cross(args, model, tokenizer, epi_tokenizer, dna_tokenizer)

    return results


if __name__ == "__main__":
    main()
