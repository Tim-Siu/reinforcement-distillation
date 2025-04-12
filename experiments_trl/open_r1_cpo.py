"""
CPO script for decoder language models.
"""

import logging
import os
import sys
from typing import Optional # Added for type hints

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import CPOConfig
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training

from trl import (
    CPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

# --- Environment Check (Optional but helpful) ---
print(f"--- ENV CHECK [PID {os.getpid()}] ---", file=sys.stderr)
print(f"RANK={os.environ.get('RANK', 'Not Set')}", file=sys.stderr)
print(f"WORLD_SIZE={os.environ.get('WORLD_SIZE', 'Not Set')}", file=sys.stderr)
print(f"LOCAL_RANK={os.environ.get('LOCAL_RANK', 'Not Set')}", file=sys.stderr)
print(f"MASTER_ADDR={os.environ.get('MASTER_ADDR', 'Not Set')}", file=sys.stderr)
print(f"MASTER_PORT={os.environ.get('MASTER_PORT', 'Not Set')}", file=sys.stderr)
print(f"--- END ENV CHECK ---", file=sys.stderr)
sys.stderr.flush()

logger = logging.getLogger(__name__)

def main(script_args: ScriptArguments, training_args: CPOConfig, model_args: ModelConfig):

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
    logger.info(f"CPOConfig parameters {training_args}")

    # --- Set seed for reproducibility ---
    set_seed(training_args.seed)

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
    # CPO uses the same tokenizer logic as SFT
    tokenizer = get_tokenizer(model_args, training_args)
    if not tokenizer.pad_token:
        # Important for CPO batching
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Setting pad token to EOS token: {tokenizer.eos_token}")

    # --- Load & Prepare Datasets ---
    # CPO requires 'prompt', 'chosen', 'rejected' columns
    if script_args.dataset_path:
        raw_dataset = load_from_disk(script_args.dataset_path)
    else:
        raw_dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    train_split = script_args.dataset_train_split
    eval_split = script_args.dataset_test_split

    # CPOTrainer handles preprocessing internally if the dataset has 'prompt', 'chosen', 'rejected'
    # It will apply chat templates if needed (controlled by args) and tokenize
    logger.info(f"Using raw datasets. CPOTrainer will preprocess.")
    train_dataset = raw_dataset[train_split]
    eval_dataset = None
    if training_args.do_eval and eval_split in raw_dataset:
        eval_dataset = raw_dataset[eval_split]
    elif training_args.do_eval:
        logger.warning(f"Evaluation split '{eval_split}' not found in dataset. Disabling evaluation.")
        training_args.do_eval = False

    logger.info(f"Raw train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Raw eval dataset size: {len(eval_dataset)}")

    # --- Model init kwargs ---
    logger.info("*** Setting model_init_kwargs in TrainingArguments ***")
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
    # CPOTrainer expects model_init_kwargs directly in CPOConfig, not TrainingArguments
    training_args.model_init_kwargs = model_kwargs

    # --- Initialize the CPOTrainer ---
    logger.info("*** Initializing CPOTrainer ***")
    trainer = CPOTrainer(
        model=model_args.model_name_or_path, # Can be path or model instance
        args=training_args,                 # Pass the CPOConfig object
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,         # Pass tokenizer/processor
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        # data_collator=None # Use CPOTrainer's default DataCollatorForPreference
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
    # CPOTrainer logs rewards/margins/etc. automatically. We can add raw sample counts.
    metrics["raw_train_samples"] = len(raw_dataset[train_split])
    try:
        metrics["train_samples"] = len(trainer.train_dataset) # After potential filtering/processing
    except TypeError:
        logger.info("Cannot determine exact train samples count for IterableDataset after processing.")
        metrics["train_samples"] = -1 # Placeholder

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
        "tags": ["open-r1", "dpo"], # Add CPO tag
    }
    if trainer.accelerator.is_main_process:
        # CPOTrainer has its own create_model_card method
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference (if applicable)
        # Access model directly through trainer.model
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        if hasattr(unwrapped_model.config, "use_cache"):
            unwrapped_model.config.use_cache = True
            unwrapped_model.config.save_pretrained(training_args.output_dir)
        else:
             logger.warning("Model config does not have 'use_cache' attribute. Skipping restoration.")

        # Also save tokenizer
        tokenizer.save_pretrained(training_args.output_dir)

    # --- Evaluate ---
    if training_args.do_eval and trainer.eval_dataset is not None:
        logger.info("*** Evaluate ***")
        eval_metrics = trainer.evaluate() # CPOTrainer's evaluate method

        # Add raw sample counts
        if eval_split in raw_dataset:
             eval_metrics["raw_eval_samples"] = len(raw_dataset[eval_split])
        else:
             eval_metrics["raw_eval_samples"] = 0

        try:
            eval_metrics["eval_samples"] = len(trainer.eval_dataset) # After potential processing
        except TypeError:
            logger.info("Cannot determine exact eval samples count for IterableDataset after processing.")
            eval_metrics["eval_samples"] = -1 # Placeholder

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
        # push_to_hub should be merged upstream first
        # trainer.push_to_hub(repo_id=repo_name, **kwargs)
        try:
            if trainer.is_world_process_zero(): # Ensure only main process pushes
                 # Push model and tokenizer
                 trainer.model.push_to_hub(repo_id=repo_name, commit_message="Upload CPO model")
                 tokenizer.push_to_hub(repo_id=repo_name, commit_message="Upload tokenizer")
                 logger.info(f"Model and tokenizer pushed to hub repo: {repo_name}")
            trainer.accelerator.wait_for_everyone() # Ensure all processes finish before exiting
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")


    logger.info("Script finished successfully.")


if __name__ == "__main__":
    # --- Parse Arguments ---
    parser = TrlParser((ScriptArguments, CPOConfig, ModelConfig)) # Use updated ModelConfig
    script_args, training_args, model_args = parser.parse_args_and_config()

    # --- Run Main Function ---
    main(script_args, training_args, model_args)