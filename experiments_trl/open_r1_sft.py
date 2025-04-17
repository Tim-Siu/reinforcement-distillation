# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage Examples:

# Filter strategy (custom preprocessing, custom collator):
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_length 4096 \
    --long_sequence_strategy filter \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill-Filter

# Padding-free strategy (SFTTrainer preprocessing, custom collator):
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_length 4096 \
    --long_sequence_strategy padding_free \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill-PaddingFree \
    --attn_implementation flash_attention_2 # Recommended for padding_free

# Truncate strategy (SFTTrainer preprocessing, custom collator):
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_length 4096 \
    --long_sequence_strategy truncate \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill-Truncate
"""

import logging
import os
import sys
import warnings # Added for warnings
from typing import Dict, List, Union # Added for type hints

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import SFTConfig
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM
)

print(f"--- ENV CHECK [PID {os.getpid()}] ---", file=sys.stderr)
print(f"RANK={os.environ.get('RANK', 'Not Set')}", file=sys.stderr)
print(f"WORLD_SIZE={os.environ.get('WORLD_SIZE', 'Not Set')}", file=sys.stderr)
print(f"LOCAL_RANK={os.environ.get('LOCAL_RANK', 'Not Set')}", file=sys.stderr)
print(f"MASTER_ADDR={os.environ.get('MASTER_ADDR', 'Not Set')}", file=sys.stderr)
print(f"MASTER_PORT={os.environ.get('MASTER_PORT', 'Not Set')}", file=sys.stderr)
print(f"--- END ENV CHECK ---", file=sys.stderr)
sys.stderr.flush()

INSTRUCTION_TEMPLATE = None
# RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
RESPONSE_TEMPLATE = "<｜Assistant｜>"

logger = logging.getLogger(__name__)

# --- Define Preprocessing Function (only needed for 'filter' strategy) ---
def preprocess_data_filter(
    examples: Dict[str, List],
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: int
) -> Dict[str, List[Union[int, List[int]]]]:
    """
    Preprocesses data for the 'filter' strategy:
    1. Applies chat template.
    2. Tokenizes *without* truncation.
    3. Filters out examples longer than max_seq_length.
    4. Appends EOS token if needed.
    """
    # 1. Apply chat template
    try:
        # Assumes 'messages' column or similar structure expected by apply_chat_template
        templated_texts = tokenizer.apply_chat_template(
            examples['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
    except KeyError:
        logger.error("Dataset does not contain a 'messages' column required for automatic chat templating.")
        raise
    except Exception as e:
        logger.error(f"Error applying chat template: {e}")
        # Log a sample for debugging
        if 'messages' in examples and examples['messages']:
             logger.error(f"Problematic sample structure: {examples['messages'][0]}")
        raise

    # 2. Tokenize without truncation first
    tokenized_outputs = tokenizer(
        templated_texts,
        truncation=False, # Important for filtering
        padding=False,
    )

    # 3. Filter based on tokenized length
    valid_indices = [
        i for i, ids in enumerate(tokenized_outputs['input_ids'])
        if len(ids) <= max_seq_length
    ]

    # 4. Keep only valid examples
    final_outputs = {}
    if valid_indices: # Check if any examples are left
        for key in tokenized_outputs:
            final_outputs[key] = [tokenized_outputs[key][i] for i in valid_indices]
    else: # Handle the case where a whole batch gets filtered out
         # Return an empty dict with the expected structure to avoid map errors
        for key in tokenized_outputs:
            final_outputs[key] = []

    # 5. Append EOS token if necessary (after filtering)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and "input_ids" in final_outputs and final_outputs["input_ids"]:
         for i in range(len(final_outputs["input_ids"])):
             # Check if EOS is already the last token
             if not final_outputs["input_ids"][i] or final_outputs["input_ids"][i][-1] != eos_token_id:
                  # Append only if sequence is not empty and EOS is missing
                  final_outputs["input_ids"][i].append(eos_token_id)
                  if "attention_mask" in final_outputs:
                      final_outputs["attention_mask"][i].append(1)

    return final_outputs


def main(script_args, training_args: SFTConfig, model_args):

    strategy = training_args.long_sequence_strategy
    if strategy not in ["truncate", "filter", "padding_free"]:
        raise ValueError(f"Invalid long_sequence_strategy: {strategy}. Must be 'truncate', 'filter', or 'padding_free'.")

    # --- Configure args based on strategy ---
    if strategy == "filter":
        # Custom preprocessing, custom collator
        training_args.dataset_kwargs = {"skip_prepare_dataset": True}
        training_args.padding_free = False # Cannot use custom collator with padding_free=True
        logger.info("Using 'filter' strategy: Custom preprocessing, custom DataCollatorForCompletionOnlyLM.")
    elif strategy == "padding_free":
        # SFTTrainer preprocessing, SFTTrainer padding_free collator
        # training_args.dataset_kwargs = {"skip_prepare_dataset": False}
        logger.info("Using 'padding_free' strategy: SFTTrainer preprocessing, SFTTrainer padding_free (DataCollatorWithFlattening).")
        # Warn if packing is also enabled (can be valid but needs attention)
        if training_args.packing:
             warnings.warn(
                 "Using packing=True with padding_free=True. SFTTrainer will handle this combination.",
                 UserWarning
             )
        # Warn if FA2 is not used
        if model_args.attn_implementation != "flash_attention_2":
             warnings.warn(
                 "Using padding_free=True without attn_implementation='flash_attention_2' is not recommended "
                 "and may lead to errors or unexpected behavior.", UserWarning
             )
    else: # strategy == "truncate"
        # SFTTrainer preprocessing, custom collator
        training_args.dataset_kwargs = {"skip_prepare_dataset": False}
        logger.info("Using 'truncate' strategy: SFTTrainer preprocessing, custom DataCollatorForCompletionOnlyLM.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # --- Logging Setup ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Effective SFTConfig settings for strategy '{strategy}':")
    logger.info(f"  output_dir: {training_args.output_dir}")
    logger.info(f"  padding_free: {training_args.padding_free}")
    logger.info(f"  dataset_kwargs: {training_args.dataset_kwargs}")
    logger.info(f"  max_length: {training_args.max_length}")
    logger.info(f"  packing: {training_args.packing}")
    # Log other relevant training args
    logger.info(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    logger.info(f"  gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  learning_rate: {training_args.learning_rate}")
    logger.info(f"  bf16: {training_args.bf16}")
    logger.info(f"  gradient_checkpointing: {training_args.gradient_checkpointing}")


    # --- Checkpoint Handling ---
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # --- W&B Init ---
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # --- Load Tokenizer ---
    tokenizer = get_tokenizer(model_args, training_args)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Setting pad token to EOS token: {tokenizer.eos_token}")


    # --- Load & Prepare Datasets ---
    max_seq_length = int(training_args.max_length) # Read from config

    if script_args.dataset_path:
        raw_dataset = load_from_disk(script_args.dataset_path)
    else:
        raw_dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    train_split = script_args.dataset_train_split
    eval_split = script_args.dataset_test_split

    train_dataset = None
    eval_dataset = None

    # Custom preprocessing only needed for 'filter' strategy
    if strategy == "filter":
        logger.info(f"Applying custom preprocessing for 'filter' strategy, max_length={max_seq_length}...")
        remove_columns = list(raw_dataset[train_split].column_names)

        train_dataset = raw_dataset[train_split].map(
            lambda examples: preprocess_data_filter(examples, tokenizer, max_seq_length),
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=remove_columns,
            desc=f"Preprocessing {train_split} (filter)",
        )
        logger.info(f"Train dataset size after filtering: {len(train_dataset)}")
        if len(train_dataset) == 0:
             logger.error("Training dataset is empty after filtering! Check data or filtering criteria.")
             sys.exit(1)

        if training_args.do_eval and eval_split in raw_dataset:
            eval_dataset = raw_dataset[eval_split].map(
                lambda examples: preprocess_data_filter(examples, tokenizer, max_seq_length),
                batched=True,
                num_proc=os.cpu_count(),
                remove_columns=remove_columns, # Use same columns as train
                desc=f"Preprocessing {eval_split} (filter)",
            )
            logger.info(f"Eval dataset size after filtering: {len(eval_dataset)}")
            if len(eval_dataset) == 0:
                 logger.warning(f"Evaluation dataset '{eval_split}' is empty after filtering. Disabling evaluation.")
                 training_args.do_eval = False
                 eval_dataset = None
        elif training_args.do_eval:
            logger.warning(f"Evaluation split '{eval_split}' not found in dataset. Disabling evaluation.")
            training_args.do_eval = False

    else:
        # For 'truncate' and 'padding_free', pass raw datasets to SFTTrainer
        logger.info(f"Using raw datasets for '{strategy}' strategy. SFTTrainer will preprocess.")
        train_dataset = raw_dataset[train_split]
        if training_args.do_eval and eval_split in raw_dataset:
            eval_dataset = raw_dataset[eval_split]
        elif training_args.do_eval:
            logger.warning(f"Evaluation split '{eval_split}' not found in dataset. Disabling evaluation.")
            training_args.do_eval = False

        # Log raw sizes for comparison later if needed
        logger.info(f"Raw train dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Raw eval dataset size: {len(eval_dataset)}")


    # --- Model init kwargs ---
    # Set it within the training_args object for SFTTrainer to use
    logger.info("*** Setting model_init_kwargs in TrainingArguments ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation, # Make sure this matches strategy if needed
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs # SFTTrainer reads this from args


    # --- Instantiate Collator (Conditionally) ---
    padding_free = False
    if strategy == "padding_free":
        padding_free = True

    logger.info("Instantiating DataCollatorForCompletionOnlyLM.")

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=RESPONSE_TEMPLATE,
        instruction_template=INSTRUCTION_TEMPLATE,
        mlm=False,
    )

    # --- Initialize the SFT Trainer (Corrected) ---
    logger.info(f"*** Initializing SFTTrainer (strategy: {strategy}) ***")
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args, # Pass the configured SFTConfig object
                             # It contains: model_init_kwargs, max_length, packing, padding_free, dataset_kwargs etc.
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        data_collator=data_collator,
    )

    logger.info(f"[Rank {trainer.args.process_index}] Trainer initialized. Args world_size: {trainer.args.world_size}")
    logger.info(f"[Rank {trainer.args.process_index}] Accelerator num_processes: {trainer.accelerator.num_processes}")

    # --- Training Loop ---
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics

    # --- Log & Save Train Metrics ---
    metrics["raw_train_samples"] = len(raw_dataset[train_split])
    try:
        # Get final effective train samples count (handles IterableDataset)
        metrics["train_samples"] = len(trainer.train_dataset)
    except TypeError: # Handle IterableDataset cases where len() doesn't work directly
        logger.info("Cannot determine exact train samples count for IterableDataset after processing.")
        metrics["train_samples"] = -1 # Placeholder
    metrics["long_sequence_strategy"] = strategy # Add strategy to metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # --- Save Model & Create Model Card ---
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1", f"sft-{strategy}"], # Add strategy tag
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        if hasattr(trainer.model.config, "use_cache"):
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)
        else:
            logger.warning("Model config does not have 'use_cache' attribute. Skipping restoration.")


    # --- Evaluate ---
    # Check trainer.eval_dataset as it might be None even if training_args.do_eval is True
    if training_args.do_eval and trainer.eval_dataset is not None:
        logger.info("*** Evaluate ***")
        eval_metrics = trainer.evaluate() # Use a different variable name

        # Store raw eval sample count
        if eval_split in raw_dataset:
             eval_metrics["raw_eval_samples"] = len(raw_dataset[eval_split])
        else:
             eval_metrics["raw_eval_samples"] = 0 # Split might not exist

        # Get final effective eval samples count
        try:
            eval_metrics["eval_samples"] = len(trainer.eval_dataset)
        except TypeError:
            logger.info("Cannot determine exact eval samples count for IterableDataset after processing.")
            eval_metrics["eval_samples"] = -1 # Placeholder
        eval_metrics["long_sequence_strategy"] = strategy # Add strategy to eval metrics

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    elif training_args.do_eval:
        logger.info("Evaluation requested but eval_dataset is None or empty after processing. Skipping evaluation.")


    # --- Push to Hub ---
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        # Ensure the repo name is clean if output_dir is local path like 'data/...'
        repo_name = training_args.hub_model_id or os.path.basename(training_args.output_dir)
        logger.info(f"Pushing to repo: {repo_name}")
        # Pass repo_id explicitly if needed, otherwise trainer uses output_dir or hub_model_id
        trainer.push_to_hub(repo_id=repo_name, **kwargs) # Pass repo_id for clarity

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    # Ensure SFTConfig has the `long_sequence_strategy` field with validation choices
    # Also ensure ModelConfig has `attn_implementation`
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Add validation for padding_free + attn_implementation early
    if training_args.long_sequence_strategy == "padding_free" and model_args.attn_implementation != "flash_attention_2":
        logger.warning(
             "Using long_sequence_strategy='padding_free' without attn_implementation='flash_attention_2' "
             "is discouraged. Consider setting --attn_implementation flash_attention_2 for compatibility and performance."
        )

    main(script_args, training_args, model_args)