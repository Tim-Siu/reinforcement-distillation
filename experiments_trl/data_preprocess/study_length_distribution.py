# -*- coding: utf-8 -*-
"""
Analyze DPO Dataset Sequence Lengths (with Optional Filtering & Saving)

This notebook loads a DPO dataset (expected to have 'chosen' and 'rejected'
columns containing conversational data), extracts the implicit prompt,
applies the chat template, tokenizes the components, optionally filters
based on token lengths, saves the filtered dataset in its original format,
and analyzes the length distributions of the filtered data to inform
DPO training parameters like max_prompt_length and max_length.
"""

# %% [markdown]
# ## 1. Setup and Configuration
# Import necessary libraries and define configuration parameters like dataset path, tokenizer name, filtering thresholds, and save path.

# %% Imports
import os
import sys
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk, Dataset, DatasetDict, Features, Value, Sequence
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Dict, List, Any, Tuple, Optional
import shutil # For directory removal

# Add project root to sys.path if running from a subdirectory (optional)
# module_path = os.path.abspath(os.path.join('..')) # Adjust if needed
# if module_path not in sys.path:
#     sys.path.append(module_path)

# %% Configuration

# --- !!! SET THESE PATHS AND PARAMETERS !!! ---
DATASET_PATH = "../../processed_datasets/openr1_math_raw_processed/openr1_math_dpo_LandMV_shuffled_input"
TOKENIZER_NAME_OR_PATH = "../../../models/modified/Qwen2.5-Math-1.5B" # Match DPO model

# --- Optional Filtering ---
APPLY_FILTERING = True # Set to True to enable filtering
# Set thresholds (use None to disable a specific filter)
MAX_PROMPT_LENGTH_FILTER: Optional[int] = 800 # Example: Keep prompts <= 16k tokens
MAX_RESPONSE_LENGTH_FILTER: Optional[int] = 19000  # Example: Keep chosen AND rejected responses <= 4k tokens

# --- Saving Filtered Data ---
SAVE_FILTERED_DATASET = True # Set to True to save the result
# Define output path (directory will be created)
# The script will automatically append filter info to the directory name
FILTERED_DATASET_OUTPUT_BASE_DIR = DATASET_PATH
OVERWRITE_EXISTING_OUTPUT = False # If True, deletes the target directory before saving
# ---

NUM_PROC = os.cpu_count() // 2  # Adjust based on your machine's cores
SAMPLE_SIZE = None # Set to an integer (e.g., 10000) for faster testing, None for full dataset

# Set plot style
sns.set_theme(style="whitegrid")

# --- Validate Filter Config ---
filter_suffix = "" # Initialize suffix for filenames/titles
if APPLY_FILTERING:
    if MAX_PROMPT_LENGTH_FILTER is None and MAX_RESPONSE_LENGTH_FILTER is None:
        print("Warning: APPLY_FILTERING is True, but both filter thresholds are None. No filtering will occur.")
        APPLY_FILTERING = False
    else:
        # Build suffix for reporting/saving only if filtering is active and thresholds exist
        suffix_parts = []
        if MAX_PROMPT_LENGTH_FILTER is not None:
            suffix_parts.append(f"P{MAX_PROMPT_LENGTH_FILTER}")
        if MAX_RESPONSE_LENGTH_FILTER is not None:
            suffix_parts.append(f"R{MAX_RESPONSE_LENGTH_FILTER}")
        if suffix_parts:
            filter_suffix = "_filtered_" + "_".join(suffix_parts)


# %% [markdown]
# ## 2. Load Data and Tokenizer

# %% Load Dataset
print(f"Loading dataset from: {DATASET_PATH}")
# ... (Keep the Dataset/DatasetDict loading logic from the previous version) ...
try:
    # Load the potentially multi-split dataset structure
    raw_data = load_from_disk(DATASET_PATH)
    print("Raw data loaded successfully.")

    # --- Handle DatasetDict vs Dataset ---
    if isinstance(raw_data, DatasetDict):
        print(f"Detected DatasetDict with splits: {list(raw_data.keys())}")
        # --- !!! CHOOSE THE SPLIT TO ANALYZE !!! ---
        split_to_analyze = 'train' # Default to 'train'
        if split_to_analyze not in raw_data:
            split_to_analyze = list(raw_data.keys())[0]
            print(f"Warning: 'train' split not found. Analyzing split: '{split_to_analyze}'")
        # ---
        original_dataset_split = raw_data[split_to_analyze] # Keep reference to original
        print(f"Selected split '{split_to_analyze}' for analysis.")
    elif isinstance(raw_data, Dataset):
        print("Detected single Dataset.")
        original_dataset_split = raw_data # Keep reference to original
        split_to_analyze = "data"
    else:
        raise TypeError(f"Loaded object is neither a Dataset nor a DatasetDict. Type: {type(raw_data)}")

    print(f"\nAnalyzing dataset split '{split_to_analyze}':")
    print("Features:")
    print(original_dataset_split.features)
    dataset_for_processing = original_dataset_split # Work on a copy/reference for processing steps
    print(f"Number of rows in split: {len(dataset_for_processing)}")

    # --- Schema Validation (on the selected split) ---
    required_cols = ['chosen', 'rejected']
    if not all(col in dataset_for_processing.column_names for col in required_cols):
         raise ValueError(f"Dataset split '{split_to_analyze}' missing required columns: {required_cols}. Found: {dataset_for_processing.column_names}")

    def check_message_format(feature_col):
        if not isinstance(feature_col, Sequence): return False
        inner_feature = feature_col.feature
        if not isinstance(inner_feature, Features): return False
        return 'role' in inner_feature and 'content' in inner_feature

    if not check_message_format(dataset_for_processing.features['chosen']) or \
       not check_message_format(dataset_for_processing.features['rejected']):
           print(f"Warning: 'chosen' or 'rejected' columns in split '{split_to_analyze}' might not be Sequence(Features({{ 'role': ..., 'content': ... }})). Check format.")

    if SAMPLE_SIZE:
        print(f"Sampling dataset split '{split_to_analyze}' to {SAMPLE_SIZE} examples.")
        if SAMPLE_SIZE > len(dataset_for_processing):
             print(f"Warning: Sample size {SAMPLE_SIZE} is larger than dataset split size {len(dataset_for_processing)}. Using full split.")
        else:
             dataset_for_processing = dataset_for_processing.shuffle(seed=42).select(range(SAMPLE_SIZE))
             # Also sample the reference dataset if we sampled the processing one
             original_dataset_split = original_dataset_split.select(dataset_for_processing._indices.to_pandas().to_numpy())


except FileNotFoundError:
    print(f"Error: Dataset not found at {DATASET_PATH}. Please check the path.")
    sys.exit(1)
except KeyError as e:
    print(f"Error: Could not find split '{split_to_analyze}' in the loaded DatasetDict: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred loading or processing the dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# %% Load Tokenizer
print(f"\nLoading tokenizer: {TOKENIZER_NAME_OR_PATH}")
# ... (Keep the tokenizer loading logic unchanged) ...
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME_OR_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Setting pad token to eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

# %% [markdown]
# ## 3. Preprocessing: Extract Prompt, Apply Template, Calculate Lengths
#
# Calculate lengths needed for filtering, but keep necessary original columns.

# %% Preprocessing Functions (extract_prompt_and_responses, template_and_tokenize_lengths - remain unchanged)
# ... (Keep the function definitions unchanged) ...
def extract_prompt_and_responses(example: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Extracts implicit prompt and separates responses from a DPO example."""
    chosen_msgs = example['chosen']
    rejected_msgs = example['rejected']
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
    if not actual_chosen and len(chosen_msgs) > divergence_idx:
        actual_chosen = chosen_msgs[divergence_idx:]
    if not actual_rejected and len(rejected_msgs) > divergence_idx:
        actual_rejected = rejected_msgs[divergence_idx:]
    # Keep original columns by returning them unchanged
    output = {
        "prompt_msgs": prompt_msgs,
        "chosen_response_msgs": actual_chosen,
        "rejected_response_msgs": actual_rejected,
        "chosen": chosen_msgs, # Return original
        "rejected": rejected_msgs # Return original
    }
    # Add any other columns from the original dataset if needed for saving later
    # for key in example:
    #     if key not in output:
    #         output[key] = example[key]
    return output


def template_and_tokenize_lengths(example: Dict[str, List], tokenizer: PreTrainedTokenizerBase) -> Dict[str, int]:
    """Applies chat template to message lists and returns token lengths."""
    # ... (Length calculation logic remains the same) ...
    prompt_str = ""
    chosen_str = ""
    rejected_str = ""
    if example.get("prompt_msgs"):
         add_gen_prompt = True
         last_prompt_role = example["prompt_msgs"][-1].get("role")
         if last_prompt_role == 'assistant': add_gen_prompt = False
         prompt_str = tokenizer.apply_chat_template(
             example["prompt_msgs"], add_generation_prompt=False, tokenize=False
         )
    if example.get("chosen_response_msgs"):
         chosen_str = tokenizer.apply_chat_template(
             example["chosen_response_msgs"], add_generation_prompt=False, tokenize=False
         )
    if example.get("rejected_response_msgs"):
         rejected_str = tokenizer.apply_chat_template(
             example["rejected_response_msgs"], add_generation_prompt=False, tokenize=False
         )

    prompt_tokens = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
    prompt_len = len(prompt_tokens) + (1 if example.get("prompt_msgs") and example.get("prompt_msgs")[-1].get("role") != 'assistant' else 0)
    chosen_tokens = tokenizer(chosen_str, add_special_tokens=False)['input_ids']
    chosen_len = len(chosen_tokens) + 1
    rejected_tokens = tokenizer(rejected_str, add_special_tokens=False)['input_ids']
    rejected_len = len(rejected_tokens) + 1

    # Return lengths AND pass through other existing columns
    output = {
        "prompt_len": prompt_len,
        "chosen_len": chosen_len,
        "rejected_len": rejected_len,
    }
    for key in example:
        if key not in output:
            output[key] = example[key]
    return output


# %% Apply Preprocessing and Length Calculation
print(f"\nPreprocessing dataset (Num Proc: {NUM_PROC})...")

# Create a dataset with original columns + message lists + length columns
print("Step 1 & 2: Extracting prompts, calculating lengths (keeping original columns)...")
augmented_dataset = dataset_for_processing.map(
    extract_prompt_and_responses,
    num_proc=NUM_PROC,
    desc="Extracting Prompts (Keeping Originals)",
    # No remove_columns here, keep 'chosen' and 'rejected'
)
augmented_dataset = augmented_dataset.map(
    lambda x: template_and_tokenize_lengths(x, tokenizer),
    num_proc=NUM_PROC,
    desc="Calculating Lengths (Keeping All Columns)",
    # No remove_columns here
)
print("Preprocessing and length calculation complete.")
print("Example augmented row:")
print(augmented_dataset[0])
print("\nAugmented dataset columns:")
print(augmented_dataset.column_names)


# %% [markdown]
# ## 4. Optional Filtering & Saving
#
# Apply filtering based on the calculated lengths. If filtering is applied and saving is enabled, save the filtered dataset containing only the original `chosen` and `rejected` columns.

# %% Apply Filtering and Prepare for Saving/Analysis

dataset_to_analyze = augmented_dataset # Start with the full augmented data
original_format_to_save = None # Will hold the data to be saved

if APPLY_FILTERING:
    print("\n--- Applying Length Filters ---")
    print(f"Max Prompt Length Threshold: {MAX_PROMPT_LENGTH_FILTER if MAX_PROMPT_LENGTH_FILTER is not None else 'None'}")
    print(f"Max Response Length Threshold: {MAX_RESPONSE_LENGTH_FILTER if MAX_RESPONSE_LENGTH_FILTER is not None else 'None'}")

    def filter_by_length(example: Dict[str, int]) -> bool:
        keep = True
        if MAX_PROMPT_LENGTH_FILTER is not None:
            keep = keep and (example['prompt_len'] <= MAX_PROMPT_LENGTH_FILTER)
        if MAX_RESPONSE_LENGTH_FILTER is not None:
            keep = keep and (example['chosen_len'] <= MAX_RESPONSE_LENGTH_FILTER)
            keep = keep and (example['rejected_len'] <= MAX_RESPONSE_LENGTH_FILTER)
        return keep

    original_count = len(augmented_dataset)
    # Filter the dataset that has length columns
    dataset_to_analyze = augmented_dataset.filter(
        filter_by_length,
        num_proc=NUM_PROC,
        desc="Applying Length Filters"
    )
    filtered_count = len(dataset_to_analyze)
    removed_count = original_count - filtered_count

    print(f"Original examples: {original_count}")
    print(f"Filtered examples: {filtered_count}")
    print(f"Removed examples:  {removed_count} ({removed_count/original_count:.2%})")

    if filtered_count == 0:
        print("Error: Dataset is empty after filtering. Cannot save or analyze.")
        sys.exit(1)

    # Prepare the dataset for saving (select original columns)
    if SAVE_FILTERED_DATASET:
         # Select only the original format columns from the filtered dataset
         columns_to_keep_for_saving = ['prompt', 'chosen', 'rejected', 'source', 'uuid', 'original_index', 'chosen_gen_idx', 'rejected_gen_idx', 'prompt_len', 'chosen_len', 'rejected_len']
         # Add any other original columns if they existed and should be saved
         # original_cols = original_dataset_split.column_names
         # columns_to_keep_for_saving = [col for col in original_cols if col in dataset_to_analyze.column_names]

        #  original_format_to_save = dataset_to_analyze.select_columns(columns_to_keep_for_saving)
         original_format_to_save = dataset_to_analyze
         print(f"\nPrepared dataset for saving with columns: {original_format_to_save.column_names}")

else:
    print("\n--- Skipping Length Filtering ---")
    # If not filtering, but saving is enabled, save the original data structure
    # if SAVE_FILTERED_DATASET:
    #     columns_to_keep_for_saving = ['chosen', 'rejected']
    #     # original_cols = original_dataset_split.column_names
    #     # columns_to_keep_for_saving = [col for col in original_cols if col in dataset_to_analyze.column_names]
    #     original_format_to_save = dataset_to_analyze.select_columns(columns_to_keep_for_saving) # dataset_to_analyze is the same as augmented_dataset here
    #     print(f"\nSaving dataset without filtering. Prepared with columns: {original_format_to_save.column_names}")


# %% Save Filtered Dataset
if SAVE_FILTERED_DATASET and original_format_to_save is not None:
    # Construct final save path
    base_name = os.path.basename(DATASET_PATH.rstrip('/'))
    final_save_dir_name = f"{base_name}{filter_suffix}"
    final_save_path = os.path.join(FILTERED_DATASET_OUTPUT_BASE_DIR, final_save_dir_name)

    print(f"\n--- Saving Filtered Dataset ---")
    print(f"Target save directory: {final_save_path}")

    if os.path.exists(final_save_path):
        if OVERWRITE_EXISTING_OUTPUT:
            print(f"Warning: Output directory {final_save_path} exists. Deleting...")
            try:
                shutil.rmtree(final_save_path)
                print("Directory deleted.")
            except OSError as e:
                print(f"Error deleting directory {final_save_path}: {e}")
                sys.exit(1)
        else:
            print(f"Error: Output directory {final_save_path} already exists and OVERWRITE_EXISTING_OUTPUT is False.")
            sys.exit(1)

    try:
        os.makedirs(FILTERED_DATASET_OUTPUT_BASE_DIR, exist_ok=True)
        original_format_to_save_wrapped = DatasetDict({"train": original_format_to_save})
        original_format_to_save_wrapped.save_to_disk(final_save_path)
        print(f"Filtered dataset saved successfully to {final_save_path}")
    except Exception as e:
        print(f"Error saving dataset to {final_save_path}: {e}")

elif SAVE_FILTERED_DATASET:
     print("\nSkipping saving: No data available after filtering or saving was disabled implicitly.")


# %% [markdown]
# ## 5. Analyze Length Distributions (Using Filtered Data for Analysis)
#
# Analyze the distributions on the `dataset_to_analyze` (which is the filtered dataset if filtering was applied, otherwise the full augmented dataset).

# %% Create DataFrame for Analysis
print("\nCreating Pandas DataFrame for Analysis...")
# Use dataset_to_analyze which contains the length columns and represents the filtered data
df = dataset_to_analyze.to_pandas()

# Calculate total lengths
df['total_chosen_len'] = df['prompt_len'] + df['chosen_len']
df['total_rejected_len'] = df['prompt_len'] + df['rejected_len']
df['max_total_len'] = df[['total_chosen_len', 'total_rejected_len']].max(axis=1)

print("DataFrame created for analysis.")
print(df[['prompt_len', 'chosen_len', 'rejected_len', 'max_total_len']].head()) # Show relevant columns


# %% Summary Statistics
print(f"\n--- Length Statistics (Data Used for Analysis{filter_suffix}) ---")
# ... (Keep stats calculation the same) ...
percentiles = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
stats = df[['prompt_len', 'chosen_len', 'rejected_len', 'total_chosen_len', 'total_rejected_len', 'max_total_len']].describe(percentiles=percentiles)
print(stats)

# %% Plot Distributions
print("\n--- Plotting Distributions ---")
# ... (Keep plotting logic the same, using df derived from dataset_to_analyze) ...
fig_title = f"Length Distributions (Split: {split_to_analyze}{filter_suffix})"
plt.figure(figsize=(18, 10))
plt.suptitle(fig_title, fontsize=16, y=1.02) # Add main title

# (Plotting subplots remain the same as before)
# Prompt Length
plt.subplot(2, 3, 1)
sns.histplot(df['prompt_len'], bins=50, kde=True)
plt.title('Prompt Length')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
if not df['prompt_len'].empty:
    plt.axvline(stats.loc['99%', 'prompt_len'], color='r', linestyle='--', label=f"99% Pct: {stats.loc['99%', 'prompt_len']:.0f}")
    plt.legend()

# Chosen Response Length
plt.subplot(2, 3, 2)
sns.histplot(df['chosen_len'], bins=50, kde=True)
plt.title('Chosen Response Length')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
if not df['chosen_len'].empty:
    plt.axvline(stats.loc['99%', 'chosen_len'], color='r', linestyle='--', label=f"99% Pct: {stats.loc['99%', 'chosen_len']:.0f}")
    plt.legend()

# Rejected Response Length
plt.subplot(2, 3, 3)
sns.histplot(df['rejected_len'], bins=50, kde=True)
plt.title('Rejected Response Length')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
if not df['rejected_len'].empty:
    plt.axvline(stats.loc['99%', 'rejected_len'], color='r', linestyle='--', label=f"99% Pct: {stats.loc['99%', 'rejected_len']:.0f}")
    plt.legend()

# Total Chosen Length
plt.subplot(2, 3, 4)
sns.histplot(df['total_chosen_len'], bins=50, kde=True)
plt.title('Total Length (Prompt + Chosen)')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
if not df['total_chosen_len'].empty:
    plt.axvline(stats.loc['99%', 'total_chosen_len'], color='r', linestyle='--', label=f"99% Pct: {stats.loc['99%', 'total_chosen_len']:.0f}")
    plt.legend()

# Total Rejected Length
plt.subplot(2, 3, 5)
sns.histplot(df['total_rejected_len'], bins=50, kde=True)
plt.title('Total Length (Prompt + Rejected)')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
if not df['total_rejected_len'].empty:
    plt.axvline(stats.loc['99%', 'total_rejected_len'], color='r', linestyle='--', label=f"99% Pct: {stats.loc['99%', 'total_rejected_len']:.0f}")
    plt.legend()

# Max Total Length (Per Pair)
plt.subplot(2, 3, 6)
sns.histplot(df['max_total_len'], bins=50, kde=True)
plt.title('Max Total Length (Per Pair)')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
if not df['max_total_len'].empty:
    plt.axvline(stats.loc['99%', 'max_total_len'], color='r', linestyle='--', label=f"99% Pct: {stats.loc['99%', 'max_total_len']:.0f}")
    plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()


# %% [markdown]
# ## 6. Interpretation and Recommendations
#
# Based on the statistics and plots (**reflecting the filtered data if filtering was applied**):
#
# *   **Feasibility Check:** If filtering was applied, check the `Removed examples` percentage. High removal might indicate thresholds are too strict.
# *   **Prompt Length:** Use the filtered `prompt_len` stats (max or 99th percentile) to inform `max_prompt_length`.
# *   **Response Lengths:** Filtered `chosen_len`/`rejected_len` confirm the filtering worked.
# *   **Max Total Length:** Use the filtered `max_total_len` stats (max or 99th percentile) to inform `max_length`. Set `max_length` in `DPOConfig` >= the max observed value here to avoid further truncation.
#
# **Remember:** The saved dataset (if enabled) at `FILTERED_DATASET_OUTPUT_BASE_DIR/your_dataset_name_filtered_...` contains only the `chosen` and `rejected` columns for the examples that passed the filter. Use this saved dataset as input for your DPO training if you want to train only on the filtered subset.

# %% End of Script
print("\nAnalysis complete.")
# %%
