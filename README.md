# RegFM: a regulatory foundation model for gene expression prediction

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Deku21%2FRegFM-ffd21e?logo=huggingface&logoColor=white)](https://huggingface.co/Deku21/RegFM)

**RegFM** is a regulatory foundation model that predicts gene expression by jointly modeling long-range cis-DNA sequence and trans-acting cellular context. It encodes megabase-scale DNA with dilated attention and integrates transcription-factor and expression signals via cross-attention, enabling accurate, cell-type-aware expression prediction.

![](init/model.jpg)

## Capabilities

- **Gene expression prediction**: predict gene expression by coupling long-range cis-regulatory sequences (CREs) with trans-acting regulators (TFs / CRs) in a cell-context-aware manner.
- **Functional genome annotation**: annotate and characterize CREs from learned regulatory representations.
- **Perturbation-response prediction**: predict transcriptional responses under regulatory perturbations.
- **Interpretability**: provide an interpretable view of cis–trans regulatory interactions.

## Pretrained models

Pretrained RegFM checkpoints are available on Hugging Face:

> [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-ffd21e?logo=huggingface&logoColor=white)](https://huggingface.co/Deku21/RegFM) **[Deku21/RegFM](https://huggingface.co/Deku21/RegFM)**

Download the checkpoint from the Hub, then point `--cis_model_name_or_path`, `--trans_model_name_or_path`, or `--output_dir` (for prediction) to the local directory. The demo notebook `RegFM_geneexp_predict_demo.ipynb` also loads from a local checkpoint path.

## Installation

Requirements:
1. Python 3.11
2. CUDA 11.8 (compatible with the PyTorch build below)
3. Packages:
   - torch (2.2.2, CUDA 11.8 build)
   - tokenizers (0.15.2)
   - numpy, tqdm, sentencepiece, sacremoses, filelock, requests, boto3
   - tensorboard (optional, for training logs)

```bash
pip install -e .
```

This installs RegFM together with the customized `transformers` under `src/transformers`.

or, after pushing to GitHub:

```bash
pip install git+https://github.com/ZjGaothu/RegFM.git
```

## Usage

Edit path placeholders in the scripts (`MODEL_PATH`, data dirs, tokenizers/configs), then:

```bash
# Training
bash scripts/run_train.sh

# Prediction
bash scripts/run_predict.sh
```

The scripts set `PYTHONPATH` to `src/` automatically.

For a single-cell-line walkthrough, see `RegFM_geneexp_predict_demo.ipynb`: it uses a model trained with leave-one-out on PBMC data, then runs prediction on the held-out **CD8 TEM 1** cell type.

### Main arguments

| Argument | Description |
| --- | --- |
| `--tfcr_dir` | TF / CR input directory |
| `--exp_dir` | Expression input directory |
| `--dna_dir` | Cis-DNA input directory |
| `--trans_model_name_or_path` | Trans-context transformer pretrained checkpoint  |
| `--cis_model_name_or_path` | Cis-DNA pretrained checkpoint  |
| `--trans_tokenizer_name` | TF tokenizer (`vocab.txt`) |
| `--exp_tokenizer_name` | Expression tokenizer (`vocab.txt`) |
| `--dna_tokenizer_name` | DNA tokenizer (`.json`) |
| `--exp_config_name` | Trans-context transformer config (`.json`) |
| `--dna_config_name` | Cis-DNA transformer config (`.json`) |
| `--output_dir` | Checkpoint / output directory |
| `--task_name` | Task name; use `genepred` |
| `--do_train` / `--do_eval` / `--do_predict` | Run training, evaluation, or prediction |
| `--max_seq_length` | Max TF / expression sequence length (e.g. `2112`) |
| `--max_dna_seq_length` | Max DNA sequence length (e.g. `71680`) |
| `--per_gpu_train_batch_size` | Train batch size per GPU |
| `--per_gpu_eval_batch_size` | Eval batch size per GPU |
| `--per_gpu_pred_batch_size` | Predict batch size per GPU |
| `--learning_rate` | Adam learning rate |
| `--num_train_epochs` | Number of training epochs |
| `--predict_dir` | Directory to write prediction outputs |
| `--save_name` | Tag for prediction file (`pred_results_{save_name}.npy`) |
| `--n_process` | Processes for feature conversion |

Prediction writes `pred_results_{save_name}.npy` under `--predict_dir`.

## Citation
If you use RegFM in your research, please cite:

Zijing Gao, et al. RegFM: a regulatory foundation model for gene expression prediction. bioRxiv, (2026).  
*(DOI will be added upon public release.)*

## Contact

If you have any questions, please contact: `gzj21@mails.tsinghua.edu.cn`

## License

This project is licensed under the [MIT License](LICENSE).
