"""
Direct Preference Optimization (DPO) script for decoder language models.

Usage Examples:

# Standard DPO with PEFT (LoRA)
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/dpo.py \
    --model_name_or_path=mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset_name=trl-internal-testing/hh-rlhf-trl-style \
    --output_dir=data/Mistral-7B-Instruct-v0.2-DPO-HH \
    --learning_rate=5.0e-7 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.1 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --num_train_epochs=1 \
    --eval_strategy=steps \
    --eval_steps=100 \
    --logging_steps=10 \
    --log_level=info \
    --max_prompt_length=512 \
    --max_length=1024 \
    --max_completion_length=None \
    --beta=0.1 \
    --loss_type=sigmoid \
    --bf16 \
    --lora_r=16 \
    --lora_alpha=32 \
    --lora_dropout=0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --report_to=wandb

# Full Fine-tuning DPO (requires more memory)
# Note: No --use_peft, ref_model=None will cause an error unless precompute_ref_log_probs=True
# or a separate ref_model path is provided. For full FT, typically pass the same path to ref_model
# or use precompute_ref_log_probs=True.
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/dpo.py \
    --model_name_or_path=mistralai/Mistral-7B-v0.1 \
    --ref_model_name_or_path=mistralai/Mistral-7B-v0.1 \
    --dataset_name=trl-internal-testing/hh-rlhf-trl-style \
    --output_dir=data/Mistral-7B-v0.1-DPO-HH-Full \
    --learning_rate=1.0e-7 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing \
    --num_train_epochs=1 \
    --eval_strategy=steps \
    --eval_steps=100 \
    --logging_steps=10 \
    --log_level=info \
    --max_prompt_length=512 \
    --max_length=1024 \
    --max_completion_length=None \
    --beta=0.1 \
    --loss_type=sigmoid \
    --bf16 \
    --report_to=wandb
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

from open_r1.configs import DPOConfig
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training

from trl import (
    DPOTrainer,
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

def main(script_args: ScriptArguments, training_args: DPOConfig, model_args: ModelConfig):

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
    logger.info(f"DPOConfig parameters {training_args}")

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
    # DPO uses the same tokenizer logic as SFT
    tokenizer = get_tokenizer(model_args, training_args)
    if not tokenizer.pad_token:
        # Important for DPO batching
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Setting pad token to EOS token: {tokenizer.eos_token}")

    # --- Load & Prepare Datasets ---
    # DPO requires 'prompt', 'chosen', 'rejected' columns
    if script_args.dataset_path:
        raw_dataset = load_from_disk(script_args.dataset_path)
    else:
        raw_dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    train_split = script_args.dataset_train_split
    eval_split = script_args.dataset_test_split

    # DPOTrainer handles preprocessing internally if the dataset has 'prompt', 'chosen', 'rejected'
    # It will apply chat templates if needed (controlled by args) and tokenize
    logger.info(f"Using raw datasets. DPOTrainer will preprocess.")
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
    # DPOTrainer expects model_init_kwargs directly in DPOConfig, not TrainingArguments
    training_args.model_init_kwargs = model_kwargs

    # --- Reference Model Setup ---
    ref_model = None
    ref_model_kwargs = {} # Needs to be initialized
    if model_args.ref_model_name_or_path:
        logger.info(f"Loading explicit reference model: {model_args.ref_model_name_or_path}")
        ref_model = model_args.ref_model_name_or_path # Pass path to DPOTrainer
        # Set ref model kwargs (can customize independently if needed)
        ref_torch_dtype = ( # Allow separate dtype for ref model if desired
            model_args.ref_torch_dtype if model_args.ref_torch_dtype in ["auto", None] else getattr(torch, model_args.ref_torch_dtype)
        )
        ref_quantization_config = get_quantization_config(model_args) # Allow separate quantization
        ref_model_kwargs = dict(
            revision=model_args.ref_model_revision,
            trust_remote_code=model_args.trust_remote_code, # Usually same
            attn_implementation=model_args.ref_attn_implementation, # Usually same, or maybe 'eager' if not fine-tuned
            torch_dtype=ref_torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True, # Match active model's setting usually
            device_map=get_kbit_device_map() if ref_quantization_config is not None else None,
            quantization_config=ref_quantization_config,
        )
        training_args.ref_model_init_kwargs = ref_model_kwargs
    # elif training_args.use_peft and not training_args.precompute_ref_log_probs:
    #     # Use PEFT model's base model as reference (handled internally by DPOTrainer if ref_model=None)
    #     logger.info("Using PEFT. The base model will be used as the reference model implicitly.")
    #     ref_model = None
    #     training_args.ref_model_init_kwargs = None # DPOTrainer handles this
    elif training_args.precompute_ref_log_probs:
        logger.info("precompute_ref_log_probs=True. Reference model will only be loaded for precomputation.")
        ref_model = None # DPOTrainer handles loading for precomputation if ref_model=None
        training_args.ref_model_init_kwargs = model_kwargs # Use same kwargs as active model for precomputation loading
    else:
        # Needs either PEFT, precomputation, or an explicit ref model path
        logger.warning("No PEFT, no precomputation, and no explicit ref_model provided. "
                     "DPOTrainer might create a reference model copy, which consumes memory. "
                     "Consider using --use_peft, --precompute_ref_log_probs=True, or provide --ref_model_name_or_path.")
        ref_model = None # Let DPOTrainer create a copy if it needs to
        training_args.ref_model_init_kwargs = model_kwargs # Use same kwargs for the potential copy

    # --- Initialize the DPOTrainer ---
    logger.info("*** Initializing DPOTrainer ***")
    trainer = DPOTrainer(
        model=model_args.model_name_or_path, # Can be path or model instance
        ref_model=ref_model,                # Can be path, instance, or None (for PEFT/precompute)
        args=training_args,                 # Pass the DPOConfig object
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,         # Pass tokenizer/processor
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        # data_collator=None # Use DPOTrainer's default DataCollatorForPreference
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
    # DPOTrainer logs rewards/margins/etc. automatically. We can add raw sample counts.
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
        "tags": ["open-r1", "dpo"], # Add DPO tag
    }
    if trainer.accelerator.is_main_process:
        # DPOTrainer has its own create_model_card method
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
        eval_metrics = trainer.evaluate() # DPOTrainer's evaluate method

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
                 trainer.model.push_to_hub(repo_id=repo_name, commit_message="Upload DPO model")
                 tokenizer.push_to_hub(repo_id=repo_name, commit_message="Upload tokenizer")
                 logger.info(f"Model and tokenizer pushed to hub repo: {repo_name}")
            trainer.accelerator.wait_for_everyone() # Ensure all processes finish before exiting
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")


    logger.info("Script finished successfully.")


if __name__ == "__main__":
    # Note: We need ref_model args in ModelConfig if we want to control them separately
    # For now, add them manually or rely on DPOTrainer kwargs if needed later.
    # Let's adjust ModelConfig slightly to accommodate ref_model paths for clarity.

    # --- Update ModelConfig temporarily (Ideally move this to configs.py) ---
    from dataclasses import dataclass, field
    @dataclass
    class DPModelConfig(ModelConfig):
        ref_model_name_or_path: Optional[str] = field(
            default=None, metadata={"help": "Path to pretrained reference model or model identifier from huggingface.co/models."}
        )
        ref_model_revision: str = field(
            default="main", metadata={"help": "The specific model version to use (branch name, tag name or commit id)."}
        )
        ref_attn_implementation: Optional[str] = field(
            default=None,
            metadata={
                "help": "Attention implementation to use for the reference model (e.g., 'eager', 'sdpa', 'flash_attention_2')."
                 " Defaults to the same as the active model if not set."
            },
        )
        ref_torch_dtype: Optional[str] = field(
            default=None,
            metadata={
                "help": "Override the reference model default dtype and load the model under this dtype. If 'auto' is passed, the "
                "dtype will be automatically derived from the model's weights.",
                "choices": ["auto", "bfloat16", "float16", "float32"],
            },
        )
        # Add flags for ref model quantization if needed (mirroring active model flags)
        ref_load_in_8bit: bool = field(default=False, metadata={"help": "Load the reference model in 8 bits."})
        ref_load_in_4bit: bool = field(default=False, metadata={"help": "Load the reference model in 4 bits."})
        ref_bnb_4bit_quant_type: Optional[str] = field(
            default="nf4", metadata={"help": "Quantization type for reference model (fp4 or nf4)."}
        )
        ref_use_bnb_nested_quant: bool = field(
            default=False, metadata={"help": "Whether to use nested quantization for reference model."}
        )

    # --- Parse Arguments ---
    parser = TrlParser((ScriptArguments, DPOConfig, DPModelConfig)) # Use updated ModelConfig
    script_args, training_args, model_args = parser.parse_args_and_config()

    # --- Argument Validation (Optional) ---
    # if training_args.use_peft and model_args.ref_model_name_or_path and not training_args.force_use_ref_model:
    #      logger.warning("You are using PEFT and provided an explicit `ref_model_name_or_path`. "
    #                   "By default, DPOTrainer uses the PEFT base model as reference. "
    #                   "If you want to use the specified `ref_model_name_or_path`, set `--force_use_ref_model=True` in DPOConfig. "
    #                   "Otherwise, the explicit ref_model may be ignored or cause issues.")

    # if not training_args.use_peft and not model_args.ref_model_name_or_path and not training_args.precompute_ref_log_probs:
    #      logger.warning("Running DPO without PEFT, without an explicit reference model, and without precomputing reference log probs. "
    #                   "DPOTrainer may create a full copy of the model for reference, significantly increasing memory usage. "
    #                   "Consider using PEFT (--use_peft), providing a reference model (--ref_model_name_or_path), or enabling precomputation (--precompute_ref_log_probs=True).")

    # Set default ref model attention implementation if not provided
    if model_args.ref_model_name_or_path and not model_args.ref_attn_implementation:
        model_args.ref_attn_implementation = model_args.attn_implementation
        logger.info(f"Setting reference model attention implementation to: {model_args.ref_attn_implementation}")

    # --- Run Main Function ---
    main(script_args, training_args, model_args)