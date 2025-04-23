### Prepare environments

Navigate to `openr1/`

```bash
# Create virtual environment, activate it, and upgrade pip
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
```

```bash
# Install vllm and flash-attn
uv pip install vllm==0.7.2
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
```

```bash
# Install openr1 in editable mode with dev dependencies, skipping LFS smudge
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
```

```bash
# Log in to Hugging Face Hub and WandB
huggingface-cli login
wandb login
```

Navigate to `trl/`

```bash
# Install trl in editable mode
uv pip install -e .
```

---
### Data preprocessing

Make sure you have `OpenR1-Math-Raw` downloaded in `experiments_trl/hf_datasets`. You can download the dataset with

```
git lfs install

git clone https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw
```

Navigate to `experiments_trl/`

1.  To clean the LandMV correct data (77629 entries, subset of OpenR1 raw), run:

    ```bash
    python data_preprocess/clean_base_shuffled.py
    ```

2.  To curate the data for positive/negative pairs (53175 queries, used in our methods and DPO), run:

    ```bash
    python data_preprocess/clean_base_shuffled_dpo.py
    ```

3.  To filter out data beyond our defined length limits, run:

    ```bash
    uv pip install matplotlib seaborn
    python data_preprocess/study_length_distribution.py
    ```
---

### SFT start

To start running experiments on Qwen/Qwen2.5-Math-1.5B, make sure you **modify** the config:

```json
{
  "max_position_embeddings": 32768,
  "rope_theta": 300000.0
}
```

Navigate to `experiments_trl/`

1.  Train on the LandMV subset for 5 epochs:

    ```bash
    bash recipes/sft/sft_landmv_5ep/train.sh
    bash recipes/sft/sft_landmv_5ep/eval.sh
    ```

### Learn **negative** samples

1.  Learn **negative** samples with our proposed reinforcement distillation for 1 epoch:

    ```bash
    bash recipes/red/red_1_08_1e-6/train.sh
    bash recipes/red/red_1_08_1e-6/eval.sh
    ```
