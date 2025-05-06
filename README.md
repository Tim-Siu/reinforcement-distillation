<div align="center">

# Reinforcement Distillation

<div>
Learning from Off-policy Negative Data 🌟
</div>
</div>
<div>
<br>

<div align="center">

[![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
)](https://github.com/Tim-Siu/reinforcement-distillation)
[![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white
)](https://shuyaoxu.notion.site/redi)

</div>

</div>

## Overview
We propose **Reinforcement Distillation (REDI)**, an efficient approach for large language models (LLMs) post-training using offline RL and distilled data. Our **`REDI-1.5B-Preview`** model, fine-tuned from **`Qwen2.5-Math-1.5B`** using a curated 78k subset of the OpenR1 dataset (leveraging both positive and negative examples), achieves **83.1% on MATH-500 (pass@1)**. It performs comparably to or better than DeepSeek-R1-Distill-Qwen-1.5B on several math benchmarks, establishing a new state-of-the-art for 1.5B models fine-tuned offline using openly available distilled data.

A key finding is that **asymmetric weighting** of positive and negative sample gradients during optimization significantly enhances training stability and performance, allowing us to surpass DPO/SimPO without KL regularization.


<div align="center">
<img src="figures/redi_comparison.svg" width="80%" />

<sub>*For more details, see our [blog post](https://shuyaoxu.notion.site/redi).*</sub>
</div>


## News
- **[2025/04/30]** ⬆️ An In-Depth Blog Post on our [Training Recipe and Insights](https://shuyaoxu.notion.site/redi)
- **[2025/04/30]** REDI codebase is released. Try it out!

## Getting Started

### Installation
You can install REDI dependencies by running the following commands:

```bash
uv venv redi_env --python 3.11 && source redi_env/bin/activate && uv pip install --upgrade pip

# Install vllm and flash-attn
uv pip install vllm==0.7.2
uv pip install matplotlib seaborn
uv pip install setuptools && uv pip install flash-attn --no-build-isolation

cd openr1
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"

huggingface-cli login
wandb login
```

```bash
# Install trl from source
cd trl
uv pip install -e .
```

---
### Data preprocessing

Make sure you have `OpenR1-Math-Raw` downloaded in `experiments_trl/hf_datasets`. You can download the dataset with

```
git lfs install

git clone https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw
```

To prepare REDI Positives and REDI Pairs:

```bash
cd experiments_trl

# clean the REDI positive data (77629 entries, subset of OpenR1 raw)
python data_preprocess/clean_base_shuffled.py

# To curate the data for positive/negative pairs (53175 queries, used in our methods and DPO)
python data_preprocess/clean_base_shuffled_dpo.py

# To filter out data beyond our defined length limits
python data_preprocess/study_length_distribution.py
```

---

### Training Script


To start running experiments on Qwen/Qwen2.5-Math-1.5B, make sure you **modify** the config:

```json
{
  "max_position_embeddings": 32768,
  "rope_theta": 300000.0
}
```

Train on the REDI Postives for 5 epochs:

```bash
cd experiments_trl/
bash recipes/sft/sft_landmv_5ep/train.sh
bash recipes/sft/sft_landmv_5ep/eval.sh
```

Learn **negative** samples with our proposed reinforcement distillation and REDI Pairs for 1 epoch:

```bash
bash recipes/redi/redi_1_08_1e-6/train.sh
bash recipes/redi/redi_1_08_1e-6/eval.sh
```

## Evaluation

For comparison with other methods, we use [rllm](https://github.com/agentica-project/rllm) (formerly DeepScaleR) for evaluation.

<details>
<summary>Detailed evaluation instructions</summary>

You may first follow instructions on [rllm project](rllm/README.md) to install a seperate environment. Then:

1. The first step is that we need to force `thinking` for our models (also standard practice for official Deepseek models). To do so, navigate to `tokenizer_config.json` and modify the template. The new template should look like
    <details>
    <summary>Updated chat template</summary>

    ```
      "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'Please reason step by step, and put your final answer within \\\\boxed{}.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nPlease reason step by step, and put your final answer within \\\\boxed{}.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant<think>\\n' }}\n{%- endif %}\n"

    ```
    </details>
2. Use the command below
    ```
    cd rllm

    bash scripts/eval/eval_model.sh --model ${MODEL_PATH} \
    --datasets aime math amc minerva olympiad_bench \
    --data-split data2 \
    --output-dir ${OUTPUT_DIR} \
    --tp 1 \
    --n 16 \
    --max-length 30720
    ```
    Note that we slightly modify the evaluation prompt to align with [Open-R1](https://github.com/huggingface/open-r1), but in our experiments this does not introduce too large a diff.
  
</details>

| Model             | AIME24 | AMC23 | MATH500 | Minerva | Olympiad Bench | Avg. |
|-------------------|------|-------|--------|------------|----------------|------|
| Deepseek-R1-Distill-Qwen-1.5b | 28.3 | 62.1  | 83.2   | 26.0       | 43.1           | 48.5 |
| SimpleRL-Zero     | 4.2  | 35.0  | 59.0   | 20.2       | 21.0           | 27.9 |
| LUFFY             | 15.2 | 46.8  | 79.4   | 26.5       | 42.4           | 42.1 |
| REDI-SFT-1.5B      | 24.0 | 57.3  | 80.4   | 27.6       | 41.1           | 47.0 |
| REDI-1.5B-Preview  | 28.1 | 62.4  | 83.1   | 28.8       | 45.2           | 49.5 |

## Acknowledgements

We thank **Hugging Face** for the Open R1 dataset and libraries like `transformers` and `trl`. We thank the **Qwen** and **DeepSeek** teams for their open-source base models. We appreciate the **DeepScaleR** project for its evaluation framework. Finally, we thank the broader **open-source AI community** for their invaluable tools and collaborative spirit.

## Citation

If you find REDI useful in your research, please consider citing our work using the following BibTeX entry:

```bibtex
@misc{xu2025redi,
  author       = {Xu, Shuyao and Peng, Cheng and Long, Jiangxuan and Xu, Weidi},
  title        = {Reinforcement Distillation: Learning from Off-policy Negative Data},
  year         = {2025},
  month        = {April},
  howpublished = {Blog Post / Technical Report},
  url          = {https://shuyaoxu.notion.site/redi},
  note         = {Code available at \url{https://github.com/Tim-Siu/reinforcement-distillation}}
}
```