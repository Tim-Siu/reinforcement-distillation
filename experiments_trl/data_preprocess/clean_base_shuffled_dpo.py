# %% Import necessary libraries
import os
from datasets import load_dataset, Dataset, DatasetDict
import gc # Garbage collector
import uuid # For generating UUIDs
import os

# %% Setup Environment and Paths
print("Current working directory:", os.getcwd())
# Set HF_ENDPOINT if needed
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Specify the correct path to the dataset directory containing the raw data
dataset_path = "../hf_datasets/OpenR1-Math-Raw" # !!! ADJUST THIS PATH IF NEEDED !!!
# Specify the split containing the raw data (e.g., 'train')
split_name = "train"

output_base_dir = "../processed_datasets/openr1_math_raw_processed" # Output dir for DPO dataset
os.makedirs(output_base_dir, exist_ok=True)

# Source to exclude
SOURCE_TO_EXCLUDE = "cn_k12"

# --- Configuration ---
SHUFFLE_SEED = 42 # Seed for reproducible shuffling (applied *before* processing)

print(f"Loading dataset from path: {dataset_path}, split: {split_name}")
# Load the specified split of the dataset
try:
    # Load dataset - consider disabling cache if memory issues persist
    # ds_raw = load_dataset(dataset_path, name=None, split=split_name, cache_dir=None)
    ds_raw = load_dataset(dataset_path, name=None, split=split_name, cache_dir="../../hf_cache")
    print(f"Dataset '{split_name}' split loaded with {len(ds_raw)} entries.")
    print("\nDataset Features:")
    print(ds_raw.features)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print(f"Please ensure the dataset path ('{dataset_path}') and split name ('{split_name}') are correct and the dataset exists.")
    raise e

# %% Add Original Index and Shuffle the Raw Dataset First (Crucial for Reproducible DPO Pairs)
print(f"\n--- Preparing and Shuffling Raw Dataset (Seed: {SHUFFLE_SEED}) ---")
print("Adding original index column...")
# Use map with with_indices=True to add the original index before shuffling
try:
    ds_with_index = ds_raw.map(
        lambda example, idx: {'original_index': idx},
        with_indices=True,
        num_proc=os.cpu_count() # Use multiple cores if available
    )
    print("Original index added.")
    # Remove the original raw dataset from memory if possible
    del ds_raw
    gc.collect()

    print("Shuffling the dataset... (This may take time and memory depending on dataset size)")
    # Shuffle the dataset with the added index. This operation creates a new shuffled dataset.
    ds_shuffled = ds_with_index.shuffle(seed=SHUFFLE_SEED)
    print(f"Dataset shuffled. New size: {len(ds_shuffled)}")
    # Remove the intermediate dataset from memory
    del ds_with_index
    gc.collect()
    print("Intermediate datasets removed from memory.")

except Exception as e:
    print(f"Error during index adding or shuffling: {e}")
    print("Ensure you have enough disk space for caching and RAM.")
    raise e


# %% Helper Function for Saving (Unchanged from previous script)

def save_hf_dataset_from_list(data_list, output_dir, dataset_name):
    """Converts a list of dicts to a Dataset and saves it under the 'train' split."""
    if not data_list:
        print(f"Warning: Empty list provided for {dataset_name}. Skipping save.")
        return None

    full_output_path = os.path.join(output_dir, dataset_name)
    os.makedirs(full_output_path, exist_ok=True)

    print(f"Converting list of {len(data_list)} dicts to Dataset for {dataset_name}...")
    try:
        # Convert list directly to Dataset object
        dataset_obj = Dataset.from_list(data_list)
        # Create a DatasetDict with a 'train' split
        dataset_dict = DatasetDict({"train": dataset_obj})
        # Save to disk
        dataset_dict.save_to_disk(full_output_path)
        print(f"Dataset {dataset_name} saved to: {full_output_path}")
        print(f"  To load back: dataset_dict = load_from_disk('{full_output_path}')")
        print(f"                ds = dataset_dict['train']")
        return full_output_path
    except Exception as e:
         print(f"Error converting list to Dataset or saving for {dataset_name}: {e}")
         if data_list:
             print("Structure of the first item:", data_list[0])
         return None

# %% DPO Pair Preparation (LandMV setting)

print(f"\n--- Starting DPO Pair Preparation (LandMV Setting) on SHUFFLED Dataset ---")
print(f"Filtering out invalid problems and source: '{SOURCE_TO_EXCLUDE}'")
print(f"Generating UUIDs and using original index from shuffled data.")
print("Keeping problems with >= 1 LandMV True AND >= 1 LandMV False generation.")

dpo_data_landmv = []
processed_count = 0
filtered_invalid_count = 0
filtered_source_count = 0
mismatched_lengths_count = 0
no_generation_count = 0
no_problem_count = 0
skipped_dpo_condition_count = 0 # Count items not meeting DPO criteria

# Corrected Key
correctness_key_reparsed = "math_verify_reparsed_answer"
print(f"Using correctness key: '{correctness_key_reparsed}'")

# Iterate through the SHUFFLED dataset
for item in ds_shuffled:
    processed_count += 1
    if processed_count % 20000 == 0:
        print(f"  Processed {processed_count} items from shuffled data for DPO...")

    # --- Extract original index added before shuffling ---
    original_index = item.get('original_index', -1) # Get index, default to -1 if missing

    # --- Filter 1: problem_is_valid ---
    problem_valid_status = item.get('problem_is_valid')
    if not (isinstance(problem_valid_status, str) and problem_valid_status.strip().lower() == 'yes'):
        filtered_invalid_count += 1
        continue

    # --- Filter 2: source ---
    if item.get("source") == SOURCE_TO_EXCLUDE:
        filtered_source_count += 1
        continue

    # --- Extract necessary data ---
    problem = item.get("problem")
    generations = item.get("generations", [])
    correctness = item.get("correctness", {})
    llama_v = correctness.get("llama_verification", [])
    math_rep_v = correctness.get(correctness_key_reparsed, []) # Use the variable

    # --- Basic Checks ---
    if not problem:
        no_problem_count += 1
        continue
    if not generations: # Check if generations list is None or empty
        no_generation_count +=1
        continue
    num_gens = len(generations)
    if num_gens == 0:
         no_generation_count +=1
         continue

    # --- Length Consistency Check ---
    if not (isinstance(llama_v, (list, tuple)) and len(llama_v) == num_gens and
            isinstance(math_rep_v, (list, tuple)) and len(math_rep_v) == num_gens):
        mismatched_lengths_count += 1
        continue

    # --- Find LandMV True and LandMV False Indices ---
    landmv_true_indices = []
    landmv_false_indices = []
    try:
        for i, (l, mr) in enumerate(zip(llama_v, math_rep_v)):
            is_landmv_true = bool(l) and bool(mr)
            if is_landmv_true:
                landmv_true_indices.append(i)
            else:
                # LandMV False means NOT (Llama True AND MathRep True)
                landmv_false_indices.append(i)
    except TypeError: # Catch if llama_v or math_rep_v elements aren't bool-able
         mismatched_lengths_count += 1 # Count as a data issue
         continue

    # --- DPO Pair Selection Logic (LandMV) ---
    # Check if we have AT LEAST ONE of each type for this problem
    if landmv_true_indices and landmv_false_indices:
        # We have a valid problem for LandMV DPO
        # Choose the *first* LandMV True as 'chosen'
        chosen_idx = landmv_true_indices[0]
        # Choose the *first* LandMV False as 'rejected'
        rejected_idx = landmv_false_indices[0]

        # --- Generate common data for this DPO pair ---
        item_uuid = str(uuid.uuid4()) # Generate UUID once per DPO pair derived from an item
        item_source = item.get("source") # Get source once

        # --- Format DPO entry ---
        chosen = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": generations[chosen_idx]}
        ]
        rejected = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": generations[rejected_idx]}
        ]
        dpo_entry = {
            "prompt": problem,
            "chosen": chosen,
            "rejected": rejected,
            "source": item_source,
            "uuid": item_uuid,
            "original_index": original_index, # Use index from item
            "chosen_gen_idx": chosen_idx,    # Optional: Track original indices
            "rejected_gen_idx": rejected_idx # Optional: Track original indices
        }
        dpo_data_landmv.append(dpo_entry)
    else:
        # This item doesn't meet the DPO condition (missing either True or False LandMV)
        skipped_dpo_condition_count += 1
        continue # Skip to the next item

# --- Clear shuffled dataset from memory ---
del ds_shuffled
gc.collect()
print("Shuffled dataset removed from memory.")

# --- Final Report ---
print("\n--- DPO Processing Summary (LandMV Setting) ---")
print(f"Total items processed (from shuffled data): {processed_count}")
print(f"Items filtered out (Invalid Problem): {filtered_invalid_count}")
print(f"Items filtered out (Source '{SOURCE_TO_EXCLUDE}'): {filtered_source_count}")
print(f"Items skipped (No Problem Text): {no_problem_count}")
print(f"Items skipped (No Generations): {no_generation_count}")
print(f"Items skipped (Mismatched Correctness Lengths): {mismatched_lengths_count}")
print(f"Items skipped (Did not meet DPO condition: >=1 True AND >=1 False LandMV): {skipped_dpo_condition_count}")
print("-" * 30)
print(f"Final count for LandMV DPO dataset: {len(dpo_data_landmv)}")
print("(Output order is based on the initial shuffle of the raw data)")
print("-" * 30)


# %% Save the final DPO dataset

print("\n--- Saving LandMV DPO Dataset (Order based on shuffled input) ---")

save_hf_dataset_from_list(
    dpo_data_landmv,
    output_base_dir,
    "openr1_math_dpo_LandMV_shuffled_input" # Name reflects DPO type and input shuffle
)
del dpo_data_landmv
gc.collect()


print("\n--- Script Finished ---")