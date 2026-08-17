#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH}"

# Edit these paths before running
MODEL_PATH=/path/to/checkpoint
DNA_PATH=/path/to/dna
TFCR_PATH=/path/to/tf
EXP_PATH=/path/to/exp

DNA_TOKENIZER=/path/to/dna_tokenizer.json
DNA_CONFIG=/path/to/dna_config.json
TRANS_TOKENIZER=/path/to/trans_tokenizer/vocab.txt
EXP_TOKENIZER=/path/to/exp_tokenizer/vocab.txt
EXP_CONFIG=/path/to/exp_config.json

python -m torch.distributed.launch --nproc_per_node=1 --master_port=18567 "${ROOT}/src/main.py" \
    --dna_tokenizer_name=$DNA_TOKENIZER \
    --dna_config_name=$DNA_CONFIG \
    --trans_tokenizer_name=$TRANS_TOKENIZER \
    --exp_tokenizer_name=$EXP_TOKENIZER \
    --exp_config_name=$EXP_CONFIG \
    --trans_model_name_or_path $MODEL_PATH \
    --cis_model_name_or_path $MODEL_PATH \
    --task_name genepred \
    --do_predict \
    --tfcr_dir $TFCR_PATH \
    --exp_dir $EXP_PATH \
    --dna_dir $DNA_PATH \
    --max_seq_length 2112 \
    --max_dna_seq_length 71680 \
    --per_gpu_pred_batch_size=1 \
    --output_dir $MODEL_PATH \
    --predict_dir $EXP_PATH \
    --save_name predict \
    --n_process 8
