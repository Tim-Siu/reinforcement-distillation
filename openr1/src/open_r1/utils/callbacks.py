#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import subprocess
import os
import sys
import time
import torch # Added for device count
from typing import List, Optional

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR # Import prefix

# Import the modified/new evaluation functions
from .evaluation import run_benchmark_jobs_local, get_lighteval_tasks # Use the new local function

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, model_config) -> None:
        self.model_config = model_config

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken, so I do this workaround
            # Also if you instantiate a new SFTConfig, the accelerator dist state will be broken
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )  # don't push the optimizer states

            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                future.add_done_callback(run_benchmark_callback)

# --- Modified Callback for Local Async Benchmarking ---
class LocalBenchmarkCallback(TrainerCallback):
    """
    Callback to run evaluation benchmarks asynchronously on saved checkpoints locally.
    """
    def __init__(self, model_config, train_config) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.benchmarks_to_run = self._resolve_benchmarks(getattr(train_config, 'benchmarks', []))
        self.benchmark_process_pids = [] # Optional: Track launched PIDs

        if not self.benchmarks_to_run:
             print("LocalBenchmarkCallback initialized, but no valid benchmarks specified.", file=sys.stderr)
        else:
             print(f"LocalBenchmarkCallback initialized. Will run benchmarks: {self.benchmarks_to_run}", file=sys.stderr)


    def _resolve_benchmarks(self, benchmark_args: list[str]) -> list[str]:
        if not benchmark_args:
            return []
        if len(benchmark_args) == 1 and benchmark_args[0] == "all":
            return get_lighteval_tasks()
        else:
            valid_benchmarks = []
            available = get_lighteval_tasks()
            for b in benchmark_args:
                if b in available:
                    valid_benchmarks.append(b)
                else:
                    print(f"Warning: Benchmark '{b}' requested but not found in supported list: {available}. Skipping.", file=sys.stderr)
            return valid_benchmarks

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Triggered after a checkpoint save. Launches local benchmarks asynchronously.
        """
        # Ensure this runs only on the main process and if benchmarks are specified
        if state.is_world_process_zero and self.benchmarks_to_run:
            global_step = state.global_step
            checkpoint_path = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")

            # Verify the checkpoint directory exists
            if not os.path.isdir(checkpoint_path):
                print(f"Error: Checkpoint directory {checkpoint_path} not found. Skipping benchmarking for step {global_step}.", file=sys.stderr)
                return

            print(f"\n--- [Step {global_step}] LocalBenchmarkCallback: Launching local benchmarks asynchronously ---", file=sys.stderr)
            start_time = time.time()

            # Determine number of GPUs to use for benchmarking (e.g., all available)
            num_gpus_benchmarking = torch.cuda.device_count()
            if num_gpus_benchmarking == 0:
                 print("Error: No GPUs detected for local benchmarking. Skipping.", file=sys.stderr)
                 return

            # Define output and log directories for benchmarks
            # Example: data/my_dpo_run/evals/checkpoint-100/
            benchmark_output_root = os.path.join(args.output_dir, "evals") # Root for all benchmark results
            benchmark_step_output_dir = os.path.join(benchmark_output_root, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
            benchmark_log_dir = os.path.join(args.output_dir, "benchmark_logs", f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
            os.makedirs(benchmark_step_output_dir, exist_ok=True)
            os.makedirs(benchmark_log_dir, exist_ok=True)


            # Extract necessary info
            trust_remote_code = getattr(self.model_config, 'trust_remote_code', False)

            try:
                # Call the local async benchmark launching function
                all_benchmarks_succeeded = run_benchmark_jobs_local(
                    checkpoint_path=checkpoint_path,
                    benchmarks_list=self.benchmarks_to_run,
                    model_args_config=self.model_config,
                    train_args_config=self.train_config, # Pass train_config
                    num_gpus=num_gpus_benchmarking,
                    benchmark_output_dir=benchmark_step_output_dir, # Pass specific output dir
                    benchmark_log_dir=benchmark_log_dir, # Pass log dir
                    trust_remote_code=trust_remote_code
                )
                # Optional: Store PIDs if you want to track them later (e.g., in on_train_end)
                if not all_benchmarks_succeeded:
                     print(f"--- [Step {global_step}] Warning: One or more benchmarks failed. Check logs in {benchmark_log_dir} ---", file=sys.stderr)
            except Exception as e:
                 print(f"Error during local benchmark job submission in LocalBenchmarkCallback: {e}", file=sys.stderr)

            end_time = time.time()
            print(f"--- [Step {global_step}] LocalBenchmarkCallback: Finished synchronous benchmarks ({end_time - start_time:.2f}s). Resuming training... --- \n", file=sys.stderr)
            sys.stderr.flush()

    # Optional: Add on_train_end to check process status or cleanup
    # def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     if state.is_world_process_zero and self.benchmark_process_pids:
    #         print("Checking status of launched benchmark processes...", file=sys.stderr)
    #         # This requires more robust process management (checking if PIDs still exist/running)
    #         # For simplicity, just log that they were launched.
    #         print(f"Launched benchmark processes with PIDs: {self.benchmark_process_pids}", file=sys.stderr)

CALLBACKS = {
    # "push_to_hub_revision": PushToHubRevisionCallback,
    "local_benchmark": LocalBenchmarkCallback,
}


# def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
#     callbacks = []
#     for callback_name in train_config.callbacks:
#         if callback_name not in CALLBACKS:
#             raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
#         callbacks.append(CALLBACKS[callback_name](model_config))

#     return callbacks

def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    callbacks = []
    requested_callbacks = getattr(train_config, 'callbacks', [])
    for callback_name in requested_callbacks:
        if callback_name not in CALLBACKS:
            print(f"Warning: Callback {callback_name} not found in CALLBACKS. Skipping.", file=sys.stderr)
        else:
            # Pass both configs
            callbacks.append(CALLBACKS[callback_name](model_config=model_config, train_config=train_config))
    return callbacks