# %% Import necessary libraries
import os
from datasets import load_dataset, Dataset, DatasetDict
import gc  # Garbage collector
import uuid  # For generating UUIDs
import random
import numpy as np
from transformers import AutoTokenizer

# %% Setup Environment and Paths
print("Current working directory:", os.getcwd())

# Specify the correct path to the dataset directory containing the raw data
dataset_path = "hf_datasets/OpenR1-Math-Raw"
split_name = "train"

output_base_dir = "processed_datasets/openr1_math_raw_processed"
os.makedirs(output_base_dir, exist_ok=True)

# Source to exclude
SOURCE_TO_EXCLUDE = "cn_k12"

# --- Configuration ---
SHUFFLE_SEED = 42  # Seed for reproducible shuffling of raw data
SPLIT_SEED = 123  # Seed for splitting Type A problems
MAX_RESPONSE_LENGTH = 19000  # Token limit for DPO responses

# Tokenizer for length calculation
TOKENIZER_PATH = "../../models/modified/Qwen2.5-Math-1.5B"

print(f"Loading dataset from path: {dataset_path}, split: {split_name}")
# Load the specified split of the dataset
try:
    ds_raw = load_dataset(dataset_path, name=None, split=split_name)
    print(f"Dataset '{split_name}' split loaded with {len(ds_raw)} entries.")
    print("\nDataset Features:")
    print(ds_raw.features)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print(f"Please ensure the dataset path ('{dataset_path}') and split name ('{split_name}') are correct.")
    raise e

# %% Add Original Index and Shuffle the Raw Dataset First
print(f"\n--- Preparing and Shuffling Raw Dataset (Seed: {SHUFFLE_SEED}) ---")
print("Adding original index column...")
try:
    ds_with_index = ds_raw.map(
        lambda example, idx: {'original_index': idx},
        with_indices=True,
        num_proc=os.cpu_count()
    )
    print("Original index added.")
    del ds_raw
    gc.collect()

    print("Shuffling the dataset...")
    ds_shuffled = ds_with_index.shuffle(seed=SHUFFLE_SEED)
    print(f"Dataset shuffled. New size: {len(ds_shuffled)}")
    del ds_with_index
    gc.collect()
    print("Intermediate datasets removed from memory.")

except Exception as e:
    print(f"Error during index adding or shuffling: {e}")
    raise e

# %% Load Tokenizer for Length Calculation
print(f"\n--- Loading Tokenizer ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise e

# %% Helper Functions
def save_hf_dataset_from_list(data_list, output_dir, dataset_name, force_multiple_shards=False):
    """Converts a list of dicts to a Dataset and saves it under the 'train' split."""
    if not data_list:
        print(f"Warning: Empty list provided for {dataset_name}. Skipping save.")
        return None

    full_output_path = os.path.join(output_dir, dataset_name)
    os.makedirs(full_output_path, exist_ok=True)

    print(f"Converting list of {len(data_list)} dicts to Dataset for {dataset_name}...")
    try:
        dataset_obj = Dataset.from_list(data_list)
        dataset_dict = DatasetDict({"train": dataset_obj})
        
        if force_multiple_shards:
            # Force multiple shards to work around HuggingFace bug
            # https://github.com/huggingface/datasets/issues/6823
            # Set max_shard_size to force at least 2 shards
            shard_size = "100MB"  # This should create multiple shards for most datasets
            dataset_dict.save_to_disk(full_output_path, max_shard_size=shard_size)
            print(f"Dataset {dataset_name} saved to: {full_output_path} with multiple shards")
        else:
            dataset_dict.save_to_disk(full_output_path)
            print(f"Dataset {dataset_name} saved to: {full_output_path}")
        
        return full_output_path
    except Exception as e:
        print(f"Error converting list to Dataset or saving for {dataset_name}: {e}")
        return None

def calculate_response_length(response_text):
    """Calculate token length of a response."""
    tokens = tokenizer(response_text, add_special_tokens=False)['input_ids']
    return len(tokens)

# %% Process Dataset and Categorize Problems
print(f"\n--- Processing Dataset and Categorizing Problems ---")

type_a_problems = []  # Problems with both LandMV correct and incorrect
type_b_problems = []  # Problems with only LandMV correct

processed_count = 0
filtered_invalid_count = 0
filtered_source_count = 0
mismatched_lengths_count = 0
no_generation_count = 0
no_problem_count = 0

correctness_key_reparsed = "math_verify_reparsed_answer"
print(f"Using correctness key: '{correctness_key_reparsed}'")

# Iterate through the SHUFFLED dataset
for item in ds_shuffled:
    processed_count += 1
    if processed_count % 20000 == 0:
        print(f"  Processed {processed_count} items...")

    original_index = item.get('original_index', -1)

    # Filter 1: problem_is_valid
    problem_valid_status = item.get('problem_is_valid')
    if not (isinstance(problem_valid_status, str) and problem_valid_status.strip().lower() == 'yes'):
        filtered_invalid_count += 1
        continue

    # Filter 2: source
    if item.get("source") == SOURCE_TO_EXCLUDE:
        filtered_source_count += 1
        continue

    # Extract necessary data
    problem = item.get("problem")
    generations = item.get("generations", [])
    correctness = item.get("correctness", {})
    llama_v = correctness.get("llama_verification", [])
    math_rep_v = correctness.get(correctness_key_reparsed, [])

    # Basic Checks
    if not problem:
        no_problem_count += 1
        continue
    if not generations:
        no_generation_count += 1
        continue
    num_gens = len(generations)
    if num_gens == 0:
        no_generation_count += 1
        continue

    # Length Consistency Check
    if not (isinstance(llama_v, (list, tuple)) and len(llama_v) == num_gens and
            isinstance(math_rep_v, (list, tuple)) and len(math_rep_v) == num_gens):
        mismatched_lengths_count += 1
        continue

    # Find first LandMV correct and incorrect indices
    first_landmv_correct_idx = -1
    first_landmv_incorrect_idx = -1
    
    try:
        for i, (l, mr) in enumerate(zip(llama_v, math_rep_v)):
            # LandMV correct: both verifiers say True
            if bool(l) and bool(mr) and first_landmv_correct_idx == -1:
                first_landmv_correct_idx = i
            # LandMV incorrect: both verifiers say False
            if not bool(l) and not bool(mr) and first_landmv_incorrect_idx == -1:
                first_landmv_incorrect_idx = i
            # Stop early if we found both
            if first_landmv_correct_idx != -1 and first_landmv_incorrect_idx != -1:
                break
    except TypeError:
        mismatched_lengths_count += 1
        continue

    # Common metadata
    item_uuid = str(uuid.uuid4())
    item_source = item.get("source")
    
    # Categorize problem
    if first_landmv_correct_idx != -1 and first_landmv_incorrect_idx != -1:
        # Type A: Has both correct and incorrect
        problem_data = {
            "problem": problem,
            "generations": generations,
            "landmv_correct_idx": first_landmv_correct_idx,
            "landmv_incorrect_idx": first_landmv_incorrect_idx,
            "source": item_source,
            "uuid": item_uuid,
            "original_index": original_index,
            "problem_type": "A"
        }
        type_a_problems.append(problem_data)
    elif first_landmv_correct_idx != -1:
        # Type B: Only has correct
        problem_data = {
            "problem": problem,
            "generations": generations,
            "landmv_correct_idx": first_landmv_correct_idx,
            "source": item_source,
            "uuid": item_uuid,
            "original_index": original_index,
            "problem_type": "B"
        }
        type_b_problems.append(problem_data)

# Clear shuffled dataset from memory
del ds_shuffled
gc.collect()
print("Shuffled dataset removed from memory.")

# Report categorization results
print("\n--- Categorization Summary ---")
print(f"Total items processed: {processed_count}")
print(f"Items filtered out (Invalid Problem): {filtered_invalid_count}")
print(f"Items filtered out (Source '{SOURCE_TO_EXCLUDE}'): {filtered_source_count}")
print(f"Items skipped (No Problem Text): {no_problem_count}")
print(f"Items skipped (No Generations): {no_generation_count}")
print(f"Items skipped (Mismatched Correctness Lengths): {mismatched_lengths_count}")
print("-" * 30)
print(f"Type A problems (has both correct and incorrect): {len(type_a_problems)}")
print(f"Type B problems (only has correct): {len(type_b_problems)}")
print(f"Problems with neither or only incorrect: {processed_count - filtered_invalid_count - filtered_source_count - no_problem_count - no_generation_count - mismatched_lengths_count - len(type_a_problems) - len(type_b_problems)}")

# %% Split Type A Problems Randomly
print(f"\n--- Splitting Type A Problems (Seed: {SPLIT_SEED}) ---")
random.seed(SPLIT_SEED)
random.shuffle(type_a_problems)

split_point = len(type_a_problems) // 2
group1_problems = type_a_problems[:split_point]
group2_problems = type_a_problems[split_point:]

print(f"Group 1 size: {len(group1_problems)}")
print(f"Group 2 size: {len(group2_problems)}")

# %% Create SFT Dataset
print("\n--- Creating Combined SFT Dataset ---")
sft_data = []

# Group 1: Use correct generations
for prob in group1_problems:
    sft_entry = {
        "messages": [
            {"role": "user", "content": prob["problem"]},
            {"role": "assistant", "content": prob["generations"][prob["landmv_correct_idx"]]}
        ],
        "chosen_gen_idx": prob["landmv_correct_idx"],
        "source": prob["source"],
        "uuid": prob["uuid"],
        "original_index": prob["original_index"],
        "sft_source": "group1_correct",
        "problem_type": prob["problem_type"],
        "group_assignment": 1
    }
    sft_data.append(sft_entry)

# Group 2: Use incorrect generations
for prob in group2_problems:
    sft_entry = {
        "messages": [
            {"role": "user", "content": prob["problem"]},
            {"role": "assistant", "content": prob["generations"][prob["landmv_incorrect_idx"]]}
        ],
        "chosen_gen_idx": prob["landmv_incorrect_idx"],
        "source": prob["source"],
        "uuid": prob["uuid"],
        "original_index": prob["original_index"],
        "sft_source": "group2_incorrect",
        "problem_type": prob["problem_type"],
        "group_assignment": 2
    }
    sft_data.append(sft_entry)

# Type B: Use correct generations
for prob in type_b_problems:
    sft_entry = {
        "messages": [
            {"role": "user", "content": prob["problem"]},
            {"role": "assistant", "content": prob["generations"][prob["landmv_correct_idx"]]}
        ],
        "chosen_gen_idx": prob["landmv_correct_idx"],
        "source": prob["source"],
        "uuid": prob["uuid"],
        "original_index": prob["original_index"],
        "sft_source": "typeB_correct",
        "problem_type": prob["problem_type"],
        "group_assignment": None
    }
    sft_data.append(sft_entry)

print(f"\nSFT Dataset Summary:")
print(f"Total SFT examples: {len(sft_data)}")
print(f"- From Group 1 (correct): {len(group1_problems)}")
print(f"- From Group 2 (incorrect): {len(group2_problems)}")
print(f"- From Type B (correct): {len(type_b_problems)}")

# %% Helper functions for DPO preprocessing
def extract_prompt_and_responses(chosen_msgs, rejected_msgs):
    """Extracts implicit prompt and separates responses from DPO messages."""
    prompt_msgs = []
    divergence_idx = 0
    min_len = min(len(chosen_msgs), len(rejected_msgs))
    for i in range(min_len):
        if chosen_msgs[i].get('role') == rejected_msgs[i].get('role') and \
           chosen_msgs[i].get('content') == rejected_msgs[i].get('content'):
            divergence_idx = i + 1
        else:
            break
    prompt_msgs = chosen_msgs[:divergence_idx]
    actual_chosen = chosen_msgs[divergence_idx:]
    actual_rejected = rejected_msgs[divergence_idx:]
    return prompt_msgs, actual_chosen, actual_rejected

def calculate_prompt_length(prompt_msgs):
    """Calculate token length of prompt messages."""
    if not prompt_msgs:
        return 0
    prompt_str = tokenizer.apply_chat_template(
        prompt_msgs, add_generation_prompt=False, tokenize=False
    )
    prompt_tokens = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
    # Add 1 if last message role is not assistant (for generation prompt)
    add_gen_prompt = 1 if prompt_msgs[-1].get("role") != 'assistant' else 0
    return len(prompt_tokens) + add_gen_prompt

# %% Create DPO Datasets with Length Filtering
print(f"\n--- Creating DPO Datasets with Length Filtering (max {MAX_RESPONSE_LENGTH} tokens) ---")

main_dpo_data = []
ablation_dpo_data = []

# Main DPO: Group 1 (correct → chosen, incorrect → rejected)
print("\nProcessing Main DPO dataset...")
for prob in group1_problems:
    correct_gen = prob["generations"][prob["landmv_correct_idx"]]
    incorrect_gen = prob["generations"][prob["landmv_incorrect_idx"]]
    
    # Create message lists
    chosen = [
        {"role": "user", "content": prob["problem"]},
        {"role": "assistant", "content": correct_gen}
    ]
    rejected = [
        {"role": "user", "content": prob["problem"]},
        {"role": "assistant", "content": incorrect_gen}
    ]
    
    # Extract prompt and responses
    prompt_msgs, chosen_response_msgs, rejected_response_msgs = extract_prompt_and_responses(chosen, rejected)
    
    # Calculate lengths
    prompt_len = calculate_prompt_length(prompt_msgs)
    chosen_len = calculate_response_length(correct_gen) + 1  # +1 for EOS token
    rejected_len = calculate_response_length(incorrect_gen) + 1  # +1 for EOS token
    
    # Check length constraints on response lengths only
    if chosen_len <= MAX_RESPONSE_LENGTH and rejected_len <= MAX_RESPONSE_LENGTH:
        dpo_entry = {
            "prompt": prob["problem"],
            "chosen": chosen,
            "rejected": rejected,
            "source": prob["source"],
            "uuid": prob["uuid"],
            "original_index": prob["original_index"],
            "chosen_gen_idx": prob["landmv_correct_idx"],
            "rejected_gen_idx": prob["landmv_incorrect_idx"],
            "prompt_msgs": prompt_msgs,
            "chosen_response_msgs": chosen_response_msgs,
            "rejected_response_msgs": rejected_response_msgs,
            "prompt_len": prompt_len,
            "chosen_len": chosen_len,
            "rejected_len": rejected_len,
            "group_assignment": 1,
            "dpo_type": "main"
        }
        main_dpo_data.append(dpo_entry)

print(f"Main DPO: {len(main_dpo_data)} pairs after length filtering (from {len(group1_problems)} total)")

# Ablation DPO: Group 2 (incorrect → chosen, correct → rejected)
print("\nProcessing Ablation DPO dataset...")
for prob in group2_problems:
    correct_gen = prob["generations"][prob["landmv_correct_idx"]]
    incorrect_gen = prob["generations"][prob["landmv_incorrect_idx"]]
    
    # Create message lists (note: incorrect is chosen, correct is rejected for ablation)
    chosen = [
        {"role": "user", "content": prob["problem"]},
        {"role": "assistant", "content": incorrect_gen}
    ]
    rejected = [
        {"role": "user", "content": prob["problem"]},
        {"role": "assistant", "content": correct_gen}
    ]
    
    # Extract prompt and responses
    prompt_msgs, chosen_response_msgs, rejected_response_msgs = extract_prompt_and_responses(chosen, rejected)
    
    # Calculate lengths
    prompt_len = calculate_prompt_length(prompt_msgs)
    chosen_len = calculate_response_length(incorrect_gen) + 1  # +1 for EOS token
    rejected_len = calculate_response_length(correct_gen) + 1  # +1 for EOS token
    
    # Check length constraints on response lengths only
    if chosen_len <= MAX_RESPONSE_LENGTH and rejected_len <= MAX_RESPONSE_LENGTH:
        dpo_entry = {
            "prompt": prob["problem"],
            "chosen": chosen,
            "rejected": rejected,
            "source": prob["source"],
            "uuid": prob["uuid"],
            "original_index": prob["original_index"],
            "chosen_gen_idx": prob["landmv_incorrect_idx"],
            "rejected_gen_idx": prob["landmv_correct_idx"],
            "prompt_msgs": prompt_msgs,
            "chosen_response_msgs": chosen_response_msgs,
            "rejected_response_msgs": rejected_response_msgs,
            "prompt_len": prompt_len,
            "chosen_len": chosen_len,
            "rejected_len": rejected_len,
            "group_assignment": 2,
            "dpo_type": "ablation"
        }
        ablation_dpo_data.append(dpo_entry)

print(f"Ablation DPO: {len(ablation_dpo_data)} pairs after length filtering (from {len(group2_problems)} total)")

# %% Balance DPO Datasets
print("\n--- Balancing DPO Datasets ---")
main_count = len(main_dpo_data)
ablation_count = len(ablation_dpo_data)

if main_count != ablation_count:
    min_count = min(main_count, ablation_count)
    print(f"Imbalanced counts: Main={main_count}, Ablation={ablation_count}")
    print(f"Balancing to {min_count} pairs each...")
    
    if main_count > min_count:
        # Randomly sample from main dataset
        random.seed(SPLIT_SEED)
        main_dpo_data = random.sample(main_dpo_data, min_count)
    
    if ablation_count > min_count:
        # Randomly sample from ablation dataset
        random.seed(SPLIT_SEED)
        ablation_dpo_data = random.sample(ablation_dpo_data, min_count)
    
    print(f"After balancing: Main={len(main_dpo_data)}, Ablation={len(ablation_dpo_data)}")
else:
    print(f"Datasets already balanced with {main_count} pairs each.")

# %% Save All Datasets
print("\n--- Saving Datasets ---")

# Save SFT dataset (normal save, no multiple shards)
save_hf_dataset_from_list(
    sft_data,
    output_base_dir,
    "openr1_math_sft_mixed_ablation",
    force_multiple_shards=False
)
del sft_data
gc.collect()

# Save Main DPO dataset (force multiple shards to avoid HuggingFace bug)
save_hf_dataset_from_list(
    main_dpo_data,
    output_base_dir,
    "openr1_math_dpo_main_ablation",
    force_multiple_shards=True
)
del main_dpo_data
gc.collect()

# Save Ablation DPO dataset (force multiple shards to avoid HuggingFace bug)
save_hf_dataset_from_list(
    ablation_dpo_data,
    output_base_dir,
    "openr1_math_dpo_inverted_ablation",
    force_multiple_shards=True
)
del ablation_dpo_data
gc.collect()

print("\n--- Script Finished Successfully ---")
print("\nDatasets created:")
print("1. openr1_math_sft_mixed_ablation - Combined SFT dataset")
print("2. openr1_math_dpo_main_ablation - Standard DPO (correct→chosen)")
print("3. openr1_math_dpo_inverted_ablation - Inverted DPO (incorrect→chosen)")