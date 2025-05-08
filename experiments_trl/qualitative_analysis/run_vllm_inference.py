# run_vllm_inference.py
import json
import torch # For cuda.empty_cache()
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser # As per your example

# !!! IMPORTANT: Define your model checkpoints here !!!
MODEL_CHECKPOINTS = {}

# !!! IMPORTANT: Define paths to your chat template files !!!
CHAT_TEMPLATE_FILES = {
    # "qwen_wo_ft_template": "qualitative_analysis/chat_template/qwen_wo_ft.jinja",
    "qwen_w_ft_template": "qualitative_analysis/chat_template/qwen_w_ft.jinja"
}

INPUT_SAMPLES_FILE = "qualitative_analysis/data/selected_samples_for_analysis.jsonl"
OUTPUT_GENERATIONS_FILE = "qualitative_analysis/data/vllm_generations_redi-1-1_ft.jsonl"

TENSOR_PARALLEL_SIZE = 1 # Set this to your desired tensor parallel size

def vllm_process_checkpoint(engine_args_dict: dict,
                            sampling_config: dict,
                            checkpoint_name: str,
                            checkpoint_model_path: str, # This will set engine_args_dict['model']
                            test_prompts_data: list,
                            all_generations: list):
    """
    Processes one model checkpoint using VLLM, adhering to the user's example structure.
    """
    current_engine_args = engine_args_dict.copy()
    current_engine_args['model'] = checkpoint_model_path
    # If tokenizer is different from model, it should also be in current_engine_args
    if 'tokenizer' not in current_engine_args or current_engine_args['tokenizer'] is None:
        current_engine_args['tokenizer'] = checkpoint_model_path # Default to model path

    print(f"\nLoading model for checkpoint: {checkpoint_name} from {current_engine_args['model']}")
    
    # These are specific to our script's needs, not for LLM directly.
    # The user example pops them from `args` before LLM init.
    # Here, `sampling_config` holds them. `chat_template_path_script_arg` is handled per template.
    max_tokens_script = sampling_config.get("max_tokens")
    temperature_script = sampling_config.get("temperature")
    top_p_script = sampling_config.get("top_p")
    top_k_script = sampling_config.get("top_k")

    try:
        # Create an LLM using only EngineArgs compatible keys from current_engine_args
        # Filter current_engine_args to only include valid EngineArgs or LLM constructor args
        # This is a bit tricky as EngineArgs are added to parser, not a dict to filter against easily.
        # We assume current_engine_args (derived from parsed FlexibleArgumentParser args) is mostly fine.
        llm = LLM(**current_engine_args)
    except Exception as e:
        print(f"Error loading model {current_engine_args['model']}: {e}")
        return

    # Create sampling params object as per the user's example
    sampling_params = llm.get_default_sampling_params()
    if max_tokens_script is not None:
        sampling_params.max_tokens = max_tokens_script
    if temperature_script is not None:
        sampling_params.temperature = temperature_script
    if top_p_script is not None:
        sampling_params.top_p = top_p_script
    if top_k_script is not None:
        sampling_params.top_k = top_k_script
    
    # It's good practice to also set stop tokens if not implicitly handled by chat template
    # or if you want to be explicit.
    # For Qwen models, im_end might be a stop token.
    # tokenizer = llm.get_tokenizer() # If you need to get specific token IDs
    # if tokenizer.eos_token_id is not None:
    #     sampling_params.stop_token_ids.append(tokenizer.eos_token_id)


    for template_name, template_path_from_config in CHAT_TEMPLATE_FILES.items():
        print(f"  Using chat template: {template_name} from {template_path_from_config}")
        chat_template_str = None
        try:
            with open(template_path_from_config, "r") as f_template:
                chat_template_str = f_template.read()
        except FileNotFoundError:
            print(f"    Chat template file not found: {template_path_from_config}. Trying model's default.")
            # If user explicitly provides chat_template_path via CLI, that would override this.
            # But here, we are iterating through CHAT_TEMPLATE_FILES.
            # If specific file not found, we can let llm.chat use its internal default by passing chat_template=None.
            chat_template_str = None # Fallback to model's default

        # Prepare batch of conversations
        conversations_batch = []
        map_idx_to_sample_id = [] # To map output back to original sample_id
        for sample_data in test_prompts_data:
            conversation = [{"role": "user", "content": sample_data["prompt_text"]}]
            # Example from user: system prompt, then user, then assistant, then user again.
            # For our test prompts, it's usually just a single user query.
            # Modify 'conversation' structure if your prompts need more context (e.g., system prompt)
            # conversation = [
            #     {"role": "system", "content": "You are a helpful math assistant."},
            #     {"role": "user", "content": sample_data["prompt_text"]}
            # ]
            conversations_batch.append(conversation)
            map_idx_to_sample_id.append(sample_data["sample_id"])
        
        if not conversations_batch:
            print("    No test prompts to process for this configuration.")
            continue

        print(f"  Generating responses for {len(conversations_batch)} prompts...")
        # Batch inference as per user example (passing list of conversations)
        # and optional chat_template string
        outputs = llm.chat(
            conversations_batch,
            sampling_params,
            use_tqdm=True,
            chat_template=chat_template_str
        )
        # outputs is List[RequestOutput]

        for i, output in enumerate(outputs):
            # As per user example: output.prompt, output.outputs[0].text
            original_prompt_text = output.prompt # This is the fully templated prompt
            generated_text = output.outputs[0].text.strip()
            original_sample_id = map_idx_to_sample_id[i]
            
            # Find the original raw prompt text (not the templated one)
            raw_prompt_text_for_logging = ""
            for pd in test_prompts_data:
                if pd["sample_id"] == original_sample_id:
                    raw_prompt_text_for_logging = pd["prompt_text"]
                    break

            all_generations.append({
                "sample_id": original_sample_id,
                "prompt_text_templated": original_prompt_text, # VLLM gives templated prompt
                "prompt_text_raw": raw_prompt_text_for_logging, # Our raw prompt
                "checkpoint_name": checkpoint_name,
                "model_path": checkpoint_model_path,
                "chat_template_name": template_name,
                "chat_template_file_used": template_path_from_config if chat_template_str else "model_default",
                "generated_assistant_text": generated_text
            })
            print(f"    Raw Prompt: {raw_prompt_text_for_logging[:50]}... -> Generated: {generated_text[:50]}...")
    
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main_cli_entrypoint(args_dict: dict):
    # Load selected test prompts
    test_prompts_data = []
    with open(INPUT_SAMPLES_FILE, "r") as f:
        for line in f:
            sample = json.loads(line)
            if sample["type"] == "test":
                test_prompts_data.append(sample)

    if not test_prompts_data:
        print(f"No test samples found in {INPUT_SAMPLES_FILE}. Exiting.")
        return

    all_generations = []
    
    # Extract sampling parameters from args_dict for our script's use
    # These will be passed to `vllm_process_checkpoint`
    # The keys must match what `FlexibleArgumentParser` (sampling_group) defined.
    sampling_config = {
        "max_tokens": args_dict.pop("script_max_tokens", 1024), # Use unique names to avoid EngineArgs clash
        "temperature": args_dict.pop("script_temperature", 0.0),
        "top_p": args_dict.pop("script_top_p", 1.0),
        "top_k": args_dict.pop("script_top_k", -1),
    }
    # The `chat_template_path` from user's example CLI args is for a single template.
    # Our script iterates CHAT_TEMPLATE_FILES. We don't need to pop it globally.
    # args_dict.pop("chat_template_path", None) # This was in user example

    # `args_dict` now primarily contains EngineArgs for LLM initialization
    # It will be further refined in `vllm_process_checkpoint` for `model` and `tokenizer`

    for ckpt_name, model_path in MODEL_CHECKPOINTS.items():
        if model_path.startswith("/path/to/your"): # Skip placeholder paths
            print(f"Skipping placeholder checkpoint {ckpt_name}: {model_path}")
            continue
        
        # Pass the remaining args_dict (mostly EngineArgs) and specific sampling_config
        vllm_process_checkpoint(
            engine_args_dict=args_dict, # These are CLI args for Engine
            sampling_config=sampling_config, # These are CLI args for Sampling
            checkpoint_name=ckpt_name,
            checkpoint_model_path=model_path,
            test_prompts_data=test_prompts_data,
            all_generations=all_generations
        )

    with open(OUTPUT_GENERATIONS_FILE, "w") as f_out:
        for gen in all_generations:
            f_out.write(json.dumps(gen) + "\n")
    print(f"\nSaved {len(all_generations)} generations to {OUTPUT_GENERATIONS_FILE}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Run VLLM inference for analysis, following user example.")
    
    # Add engine args (as per user's VLLM example)
    # These will be parsed into args_dict and passed to LLM(**args_dict)
    # The 'model' and 'tokenizer' from these args will be overridden by MODEL_CHECKPOINTS
    # but other engine settings (like tensor_parallel_size) will be used.
    engine_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(engine_group)
    parser.set_defaults(tensor_parallel_size=TENSOR_PARALLEL_SIZE)
    # Example: set a default model if user doesn't provide --model, though we override it.
    # engine_group.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct") 

    # Add sampling params (as per user's VLLM example, but with unique names)
    # These will be popped from args_dict before LLM init and used to configure SamplingParams.
    sampling_group = parser.add_argument_group("Sampling parameters (for script control)")
    sampling_group.add_argument("--script-max-tokens", type=int, default=32768) # Renamed
    sampling_group.add_argument("--script-temperature", type=float, default=0.6) # Renamed
    sampling_group.add_argument("--script-top-p", type=float, default=0.95) # Renamed
    sampling_group.add_argument("--script-top-k", type=int, default=-1) # Renamed
    
    # Add example param (as per user's VLLM example)
    # We don't use this directly as we iterate CHAT_TEMPLATE_FILES, but include for completeness
    # parser.add_argument("--chat-template-path", type=str, help="Path to a single chat template file (NOTE: this script iterates internal templates)")

    args = parser.parse_args()
    args_dict = vars(args)
    
    main_cli_entrypoint(args_dict)