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

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

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

INSTRUCTION_TEMPLATE = "<|im_start|>user\n" # Example
RESPONSE_TEMPLATE = "<|im_end|>\n<|im_start|>assistant\n"

logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):

    # Set dataset_kwargs to skip default SFTTrainer preparation
    # We will handle tokenization, filtering, and applying chat template manually.
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
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
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    #################
    # Load datasets
    #################
    max_seq_length = int(training_args.max_length)

    if script_args.dataset_path:
        raw_dataset = load_from_disk(script_args.dataset_path)
    else:
        raw_dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    train_split = script_args.dataset_train_split
    eval_split = script_args.dataset_test_split

    # --- Preprocessing Function ---
    def preprocess_data(examples):
        # 1. Apply chat template
        templated_texts = tokenizer.apply_chat_template(
            examples['messages'], # Assumes 'messages' column
            tokenize=False,
            add_generation_prompt=False
        )

        # 2. Tokenize based on strategy
        strategy = training_args.long_sequence_strategy
        if strategy == "truncate":
            # Tokenize and truncate
            tokenized_outputs = tokenizer(
                templated_texts,
                truncation=True,
                max_length=max_seq_length,
                padding=False, # Collator will handle padding
            )
            final_outputs = tokenized_outputs

        elif strategy == "filter":
            # Tokenize without truncation first
            tokenized_outputs = tokenizer(
                templated_texts,
                truncation=False, # Important for filtering
                padding=False,
            )

            # Filter based on tokenized length
            valid_indices = [
                i for i, ids in enumerate(tokenized_outputs['input_ids'])
                if len(ids) <= max_seq_length
            ]

            # Keep only valid examples
            final_outputs = {}
            if valid_indices: # Check if any examples are left
                for key in tokenized_outputs:
                    final_outputs[key] = [tokenized_outputs[key][i] for i in valid_indices]
            else: # Handle the case where a whole batch gets filtered out
                 # Return an empty dict with the expected structure to avoid map errors
                for key in tokenized_outputs:
                    final_outputs[key] = []

        else: # Should not happen due to validation in SFTConfig
            raise ValueError(f"Invalid long_sequence_strategy: {strategy}")


        # 3. Append EOS token if necessary (after potential truncation/filtering)
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

    # Determine columns to remove
    remove_columns = list(raw_dataset[train_split].column_names)

    # --- Apply Preprocessing ---
    logger.info(f"Preprocessing train dataset with strategy: '{training_args.long_sequence_strategy}', max_length={max_seq_length}...")
    train_dataset = raw_dataset[train_split].map(
        preprocess_data,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=remove_columns,
        desc=f"Preprocessing {train_split} ({training_args.long_sequence_strategy})",
    )
    logger.info(f"Train dataset size after preprocessing: {len(train_dataset)}")
    if len(train_dataset) == 0:
         logger.error("Training dataset is empty after preprocessing! Check data or filtering criteria.")
         # Depending on desired behavior, you might want to sys.exit(1) here


    eval_dataset = None
    if training_args.do_eval and eval_split in raw_dataset:
        logger.info(f"Preprocessing evaluation dataset with strategy: '{training_args.long_sequence_strategy}', max_length={max_seq_length}...")
        eval_dataset = raw_dataset[eval_split].map(
            preprocess_data,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=remove_columns,
            desc=f"Preprocessing {eval_split} ({training_args.long_sequence_strategy})",
        )
        logger.info(f"Eval dataset size after preprocessing: {len(eval_dataset)}")
        if len(eval_dataset) == 0:
             logger.warning(f"Evaluation dataset '{eval_split}' is empty after preprocessing. Disabling evaluation.")
             training_args.do_eval = False # Disable eval if dataset is empty
             eval_dataset = None

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    #######################
    # Instantiate Collator
    #######################
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=RESPONSE_TEMPLATE,
        instruction_template=INSTRUCTION_TEMPLATE,
        mlm=False, # Ensure causal LM objective
    )

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics

    metrics["raw_train_samples"] = len(raw_dataset[train_split])
    # Need to recalculate train samples if dataset changed size
    try:
      metrics["train_samples"] = len(train_dataset)
    except TypeError: # Handle IterableDataset
      logger.info("Cannot determine exact train samples for IterableDataset.")
      metrics["train_samples"] = -1 # Placeholder    trainer.log_metrics("train", metrics)

    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["raw_eval_samples"] = len(raw_dataset[eval_split])
        try:
           metrics["eval_samples"] = len(eval_dataset)
        except TypeError:
           logger.info("Cannot determine exact eval samples for IterableDataset.")
           metrics["eval_samples"] = -1 # Placeholder
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)