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

- Python 3.11
- CUDA 11.8 (for the PyTorch build below)
- torch 2.2.2 (CUDA 11.8), tokenizers 0.15.2, plus numpy, tqdm, sentencepiece, sacremoses, filelock, requests, boto3, huggingface_hub; tensorboard if you want training logs

```bash
pip install -e .
```

This also installs the local `transformers` under `src/transformers` — `import transformers` should resolve to that copy.

```bash
pip install git+https://github.com/ZjGaothu/RegFM.git
```

## Inference

First call downloads the checkpoint and tokenizers:

```python
from regfm import RegFM

model = RegFM.from_pretrained("Deku21/RegFM", device="cuda")  # PBMC by default

pred = model.predict_expression(
    dna_sequence=dna,          # DNA string
    trans_expression=trans,    # length-2103, order is model.tfcr_order
)
print(pred)
```

`variant="HVG"` or `"AllProtein"` if you need those. A local snapshot directory works instead of the repo id.

- one gene: `RegFM_single_query_demo.ipynb`
- held-out PBMC cell type (CD8 TEM 1): `RegFM_geneexp_predict_demo.ipynb` — model comes from the Hub; DNA / TF / expression matrices you still point to locally

## Training / batch prediction

Fill in the path placeholders in the scripts, then:

```bash
bash scripts/run_train.sh
bash scripts/run_predict.sh
```

`PYTHONPATH` is set to `src/` in the scripts. Predictions are written as `pred_results_{save_name}.npy` under `--predict_dir`.

### Main arguments

| Argument | Description |
| --- | --- |
| `--tfcr_dir` | TF / CR input directory |
| `--exp_dir` | Expression input directory |
| `--dna_dir` | Cis-DNA input directory |
| `--trans_model_name_or_path` | Trans-context transformer pretrained checkpoint |
| `--cis_model_name_or_path` | Cis-DNA pretrained checkpoint |
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

## Citation
If you use RegFM in your research, please cite:

Zijing Gao, et al. RegFM: an interpretable context-aware foundation model for human transcriptional regulation. bioRxiv (2026). https://doi.org/10.64898/2026.08.17.744355

## Contact

If you have any questions, please contact: `gzj21@mails.tsinghua.edu.cn`

## License

This project is licensed under the [MIT License](LICENSE).
