#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH}"

# Edit these paths before running
TRANS_MODEL_PATH=/path/to/trans_pretrain
CIS_MODEL_PATH=/path/to/cis_pretrain
TFCR_PATH=/path/to/tf
EXP_PATH=/path/to/exp
DNA_PATH=/path/to/dna
OUTPUT_PATH=/path/to/output

DNA_TOKENIZER=/path/to/dna_tokenizer.json
DNA_CONFIG=/path/to/dna_config.json
TRANS_TOKENIZER=/path/to/trans_tokenizer/vocab.txt
EXP_TOKENIZER=/path/to/exp_tokenizer/vocab.txt
EXP_CONFIG=/path/to/exp_config.json

python -m torch.distributed.launch --nproc_per_node=2 --master_port=16148 "${ROOT}/src/main.py" \
    --dna_tokenizer_name=$DNA_TOKENIZER \
    --dna_config_name=$DNA_CONFIG \
    --trans_tokenizer_name=$TRANS_TOKENIZER \
    --exp_tokenizer_name=$EXP_TOKENIZER \
    --exp_config_name=$EXP_CONFIG \
    --trans_model_name_or_path $TRANS_MODEL_PATH \
    --cis_model_name_or_path $CIS_MODEL_PATH \
    --task_name genepred \
    --do_train \
    --do_eval \
    --early_stop 4 \
    --save_total_limit 20 \
    --num_train_epochs 50 \
    --tfcr_dir $TFCR_PATH \
    --exp_dir $EXP_PATH \
    --dna_dir $DNA_PATH \
    --max_seq_length 2112 \
    --max_dna_seq_length 71680 \
    --gradient_accumulation_steps 2 \
    --per_gpu_eval_batch_size=1 \
    --per_gpu_train_batch_size=1 \
    --learning_rate 1e-5 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 500 \
    --save_steps 500 \
    --max_steps 1000000 \
    --warmup_percent 0.001 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output_dir \
    --weight_decay 0.01 \
    --n_process 8
