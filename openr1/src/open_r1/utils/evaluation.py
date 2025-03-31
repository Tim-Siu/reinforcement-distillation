import subprocess
import subprocess
import os
import sys
import shlex
import time
from typing import TYPE_CHECKING, Dict, Union, Optional, List

from .hub import get_gpu_count_for_vllm, get_param_count_from_repo_id


if TYPE_CHECKING:
    from trl import GRPOConfig, SFTConfig, ModelConfig
    from open_r1.configs import ModelConfig, DPOConfig

import os


# We need a special environment setup to launch vLLM from within Slurm training jobs.
# - Reference code: https://github.com/huggingface/brrr/blob/c55ba3505686d690de24c7ace6487a5c1426c0fd/brrr/lighteval/one_job_runner.py#L105
# - Slack thread: https://huggingface.slack.com/archives/C043JTYE1MJ/p1726566494958269
user_home_directory = os.path.expanduser("~")
VLLM_SLURM_PREFIX = [
    "env",
    "-i",
    "bash",
    "-c",
    f"for f in /etc/profile.d/*.sh; do source $f; done; export HOME={user_home_directory}; sbatch ",
]


def register_lighteval_task(
    configs: Dict[str, str], eval_suite: str, task_name: str, task_list: str, num_fewshot: int = 0
):
    """Registers a LightEval task configuration.

    - Core tasks can be added from this table: https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks_table.jsonl
    - Custom tasks that require their own metrics / scripts, should be stored in scripts/evaluation/extended_lighteval_tasks

    Args:
        configs (Dict[str, str]): The dictionary to store the task configuration.
        eval_suite (str, optional): The evaluation suite.
        task_name (str): The name of the task.
        task_list (str): The comma-separated list of tasks in the format "extended|{task_name}|{num_fewshot}|0" or "lighteval|{task_name}|{num_fewshot}|0".
        num_fewshot (int, optional): The number of few-shot examples. Defaults to 0.
        is_custom_task (bool, optional): Whether the task is a custom task. Defaults to False.
    """
    # Format task list in lighteval format
    task_list = ",".join(f"{eval_suite}|{task}|{num_fewshot}|0" for task in task_list.split(","))
    configs[task_name] = task_list


LIGHTEVAL_TASKS = {}

register_lighteval_task(LIGHTEVAL_TASKS, "custom", "math_500", "math_500", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "aime24", "aime24", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "aime25", "aime25", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "gpqa", "gpqa:diamond", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb", "lcb:codegeneration", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb_v4", "lcb:codegeneration_v4", 0)


def get_lighteval_tasks():
    return list(LIGHTEVAL_TASKS.keys())


SUPPORTED_BENCHMARKS = get_lighteval_tasks()


def run_lighteval_job(
    benchmark: str, training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig"
) -> None:
    task_list = LIGHTEVAL_TASKS[benchmark]
    model_name = training_args.hub_model_id
    model_revision = training_args.hub_model_revision
    # For large models >= 30b params or those running the MATH benchmark, we need to shard them across the GPUs to avoid OOM
    num_gpus = get_gpu_count_for_vllm(model_name, model_revision)
    if get_param_count_from_repo_id(model_name) >= 30_000_000_000:
        tensor_parallel = True
    else:
        num_gpus = 8
        tensor_parallel = False

    cmd = VLLM_SLURM_PREFIX.copy()
    cmd_args = [
        f"--gres=gpu:{num_gpus}",
        f"--job-name=or1_{benchmark}_{model_name.split('/')[-1]}_{model_revision}",
        "slurm/evaluate.slurm",
        benchmark,
        f'"{task_list}"',
        model_name,
        model_revision,
        f"{tensor_parallel}",
        f"{model_args.trust_remote_code}",
    ]
    if training_args.system_prompt is not None:
        cmd_args.append(f"--system_prompt={training_args.system_prompt}")
    cmd[-1] += " " + " ".join(cmd_args)
    subprocess.run(cmd, check=True)


def run_benchmark_jobs(training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig") -> None:
    benchmarks = training_args.benchmarks
    if len(benchmarks) == 1 and benchmarks[0] == "all":
        benchmarks = get_lighteval_tasks()
        # Evaluate on all supported benchmarks. Later we may want to include a `chat` option
        # that just evaluates on `ifeval` and `mt_bench` etc.

    for benchmark in benchmarks:
        print(f"Launching benchmark `{benchmark}`")
        if benchmark in get_lighteval_tasks():
            run_lighteval_job(benchmark, training_args, model_args)
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")

def run_lighteval_local(
    benchmark: str,
    task_list_template: str, # e.g., "custom|{task}|0|0"
    model_path: str,
    model_args_config: "ModelConfig",
    train_args_config: "DPOConfig", # Using DPOConfig as example type
    num_gpus: int,
    output_dir: str,
    log_dir: str,
    trust_remote_code: bool,
    use_chat_template_flag: bool, # Explicit flag
    custom_tasks_path: Optional[str], # Optional path
) -> bool: # Return True on success, False on failure
    """
    Launches and waits for a lighteval benchmark job locally.

    Returns:
        True if the benchmark completed successfully (exit code 0), False otherwise.
    """
    # --- Construct MODEL_ARGS exactly as specified ---
    dtype = getattr(model_args_config, 'torch_dtype', 'bfloat16')
    if dtype is None or dtype == "auto":
        dtype = 'bfloat16'

    # Use max_length from train_args_config or model_args_config or fallback
    max_model_len = 32768
    max_new_tokens = 32768
    temperature = 0.6
    top_p = 0.95
    gen_params_str_part = f"{{max_new_tokens:{max_new_tokens},temperature:{temperature},top_p:{top_p}}}"
    # Note: Based on your command, generation parameters are NOT part of MODEL_ARGS.
    # If lighteval vllm needs them, they might be inferred or set via a different flag.
    # We will NOT add generation_parameters here.
    model_args_list = [
        f"pretrained={model_path}",
        f"dtype={dtype}",
        # Following user command: use data_parallel_size if that's what their lighteval setup expects
        # If vLLM backend is used, tensor_parallel_size is more common. Adjust if needed.
        f"data_parallel_size={num_gpus}",
        f"max_model_length={max_model_len}", # Corrected from max_model_length if vLLM standard
        f"gpu_memory_utilization=0.50", # Keep high for standalone eval
        f"generation_parameters={gen_params_str_part}", # Hypothetical
    ]
    # Add trust_remote_code only if needed/present in original MODEL_ARGS setup
    # Assuming trust_remote_code is NOT part of MODEL_ARGS based on user command.
    # If it *is* needed within MODEL_ARGS, add: f"trust_remote_code={str(trust_remote_code).lower()}"

    model_args_str = ",".join(model_args_list)


    # --- Construct the Task String ---
    # Example: task_list_template = "custom|{task}|0|0"
    task_spec_str = task_list_template.format(task=benchmark)


    # --- Construct the lighteval command list ---
    cmd_list = [
        "lighteval", "vllm",
        model_args_str, # Pass the constructed MODEL_ARGS
        task_spec_str,  # Pass the task specification string
        "--output-dir", output_dir,
    ]

    # Add optional flags based on arguments and user's command structure
    if custom_tasks_path and os.path.exists(custom_tasks_path):
         cmd_list.extend(["--custom-tasks", custom_tasks_path])

    if use_chat_template_flag:
         cmd_list.append("--use-chat-template") # Corrected hyphen

    # Add system prompt if provided in train_args_config (lighteval supports this)
    system_prompt = getattr(train_args_config, 'system_prompt', None)
    if system_prompt:
        # Quote system prompt for shell safety
        cmd_list.extend(["--system_prompt", shlex.quote(system_prompt)])

    # If generation parameters are needed via a specific lighteval flag (e.g., --generation_config), add it here.
    # Example (IF lighteval supports it like this):
    # gen_params_dict = getattr(train_args_config, 'generation_params', {})
    # if gen_params_dict:
    #     gen_params_str = json.dumps(gen_params_dict) # Use JSON format
    #     cmd_list.extend(["--generation_config", gen_params_str]) # Hypothetical flag


    # --- Prepare log files and launch ---
    os.makedirs(log_dir, exist_ok=True)
    log_file_base = os.path.join(log_dir, f"benchmark_{benchmark}_{os.path.basename(model_path.rstrip('/'))}")
    stdout_log = f"{log_file_base}.stdout"
    stderr_log = f"{log_file_base}.stderr"

    # Convert command list to string only for printing
    cmd_str_for_log = " ".join(shlex.quote(str(arg)) for arg in cmd_list)

    print(f"--- Launching Benchmark Locally (Async): {benchmark} ---", file=sys.stderr)
    print(f"Model: {model_path}", file=sys.stderr)
    print(f"Command: {cmd_str_for_log}", file=sys.stderr) # Log the properly quoted command
    print(f"StdOut Log: {stdout_log}", file=sys.stderr)
    print(f"StdErr Log: {stderr_log}", file=sys.stderr)
    sys.stderr.flush()

    start_time = time.time()
    success = False
    try:
        process_env = os.environ.copy()
        # If necessary, limit GPUs for the benchmark process here:
        # Example: Use only the first 2 GPUs
        # visible_devices = ",".join(map(str, range(2)))
        # process_env['CUDA_VISIBLE_DEVICES'] = visible_devices
        # print(f"Setting CUDA_VISIBLE_DEVICES={visible_devices} for benchmark subprocess.", file=sys.stderr)

        with open(stdout_log, 'w') as f_stdout, open(stderr_log, 'w') as f_stderr:
            # Use subprocess.run to wait for completion
            result = subprocess.run(
                cmd_list,
                stdout=f_stdout,
                stderr=f_stderr,
                env=process_env,
                check=False, # Don't raise exception on non-zero exit
                # close_fds=True, # Good practice on Unix
            )
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"--- Benchmark {benchmark} completed successfully ({elapsed_time:.2f}s). ---", file=sys.stderr)
            success = True
        else:
            print(f"--- Benchmark {benchmark} failed with exit code {result.returncode} ({elapsed_time:.2f}s). Check logs: {stderr_log} ---", file=sys.stderr)
            success = False

    except FileNotFoundError:
        print(f"Error: 'lighteval' command not found. Is it installed and in the PATH?", file=sys.stderr)
        success = False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error running local benchmark {benchmark} ({elapsed_time:.2f}s): {e}", file=sys.stderr)
        success = False
    finally:
        sys.stderr.flush()

    return success

# --- Ensure run_benchmark_jobs_local calls the corrected function ---
def run_benchmark_jobs_local(
    checkpoint_path: str,
    benchmarks_list: List[str],
    model_args_config: "ModelConfig",
    train_args_config: "DPOConfig",
    num_gpus: int,
    benchmark_output_dir: str,
    benchmark_log_dir: str,
    trust_remote_code: bool,
    custom_tasks_path: Optional[str] = "../openr1/src/open_r1/evaluate.py", # Default or from config
    task_template: str = "custom|{task}|0|0" # Make template configurable if needed
) -> bool:
    """
    Runs benchmark jobs locally (synchronously) for a given checkpoint path.
    Executes benchmarks one after another.

    Returns:
        True if all launched benchmarks completed successfully, False otherwise.
    """
    # (Benchmark resolution logic remains the same...)
    benchmarks_to_run = []
    if not benchmarks_list: return []
    if len(benchmarks_list) == 1 and benchmarks_list[0] == "all":
        benchmarks_to_run = get_lighteval_tasks()
    else:
        valid_benchmarks = []
        available = get_lighteval_tasks()
        for b in benchmarks_list:
            if b in available:
                valid_benchmarks.append(b)
            else:
                print(f"Warning: Benchmark '{b}' requested but not found in supported list: {available}. Skipping.", file=sys.stderr)
        benchmarks_to_run = valid_benchmarks


    all_succeeded = True
    total_start_time = time.time()
    print(f"--- Starting Synchronous Local Benchmarks for Checkpoint: {checkpoint_path} ---", file=sys.stderr)
    sys.stderr.flush()

    # Get use_chat_template flag from train_args_config (safer default to False if not present)
    use_chat_template = True
    # Get custom tasks path (handle potential None)
    tasks_path = getattr(train_args_config, 'custom_tasks_path', custom_tasks_path)


    for benchmark in benchmarks_to_run:
        # Check if benchmark is valid (already done, but good practice)
        if benchmark in LIGHTEVAL_TASKS:
            # Create specific output subdir
            benchmark_step_output_dir = os.path.join(benchmark_output_dir, benchmark.replace(":", "_")) # Sanitize name
            os.makedirs(benchmark_step_output_dir, exist_ok=True)

            success = run_lighteval_local(
                benchmark=benchmark,
                task_list_template=task_template, # Pass the template
                model_path=checkpoint_path,
                model_args_config=model_args_config,
                train_args_config=train_args_config,
                num_gpus=num_gpus,
                output_dir=benchmark_step_output_dir,
                log_dir=benchmark_log_dir,
                trust_remote_code=trust_remote_code,
                use_chat_template_flag=use_chat_template, # Pass the boolean flag
                custom_tasks_path=tasks_path
            )
            if not success:
                all_succeeded = False
        else:
             # This case should ideally not be reached due to earlier check
             print(f"Internal Warning: Benchmark {benchmark} task list missing. Skipping.", file=sys.stderr)


    total_elapsed_time = time.time() - total_start_time
    print(f"--- Finished Local Benchmarks for {checkpoint_path} ({total_elapsed_time:.2f}s). Success: {all_succeeded} ---", file=sys.stderr)
    sys.stderr.flush()
    return all_succeeded