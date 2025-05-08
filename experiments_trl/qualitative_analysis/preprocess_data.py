# preprocess_data.py
import json
import hashlib
import random
from datasets import load_dataset

# Configuration
NUM_TRAIN_SAMPLES = 4
TEST_SAMPLES_CONFIG = {
    "aime_2024": {"name": "HuggingFaceH4/aime_2024", "split": "train", "column": "problem", "count": 1},
    "aime_2025": {"name": "yentinglin/aime_2025", "split": "train", "column": "problem", "count": 1},
    "math_500": {"name": "HuggingFaceH4/MATH-500", "split": "test", "column": "problem", "count": 2},
}
# !!! IMPORTANT: Update this path to your actual REDI Pairs dataset !!!
REDI_PAIRS_DATASET_PATH = "processed_datasets/openr1_math_raw_processed/openr1_math_dpo_LandMV_shuffled_input/openr1_math_dpo_LandMV_shuffled_input_filtered_P800_R19000" # Replace with your actual dataset name/path
OUTPUT_FILE = "qualitative_analysis/data/selected_samples_for_analysis.jsonl"

# Ensure the output directory exists
import os
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def get_content_hash(content_string):
    """Generates a SHA256 hash for the given string content."""
    return hashlib.sha256(content_string.encode('utf-8')).hexdigest()

def main():
    selected_samples = []
    rng = random.Random(42) # For reproducibility

    # 1. Sample from Training Data (REDI Pairs)
    print(f"Loading training dataset from: {REDI_PAIRS_DATASET_PATH}")
    try:
        train_ds = load_dataset(REDI_PAIRS_DATASET_PATH, split="train")
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        print("Please ensure REDI_PAIRS_DATASET_PATH is set correctly and the dataset is accessible.")
        print("Mocking training data for demonstration purposes.")
        train_ds = [
            {
                "prompt": "Solve for x: 2x + 3 = 7",
                "chosen": [{"role": "user", "content": "Solve for x: 2x + 3 = 7"}, {"role": "assistant", "content": "The solution is x = 2."}],
                "rejected": [{"role": "user", "content": "Solve for x: 2x + 3 = 7"}, {"role": "assistant", "content": "The solution is x = 5."}],
                "source": "mock", "uuid": "mock_uuid_1", "original_index": 0, # uuid here is from original data
                "chosen_gen_idx":0, "rejected_gen_idx":1
            }
        ] * NUM_TRAIN_SAMPLES

    if len(train_ds) < NUM_TRAIN_SAMPLES:
        print(f"Warning: Training dataset has only {len(train_ds)} samples, requested {NUM_TRAIN_SAMPLES}.")
        num_to_sample_train = len(train_ds)
    else:
        num_to_sample_train = NUM_TRAIN_SAMPLES
    
    if num_to_sample_train > 0 and len(train_ds) >= num_to_sample_train:
        sampled_train_indices = rng.sample(range(len(train_ds)), num_to_sample_train)
        for i in sampled_train_indices:
            item = train_ds[i]
            
            chosen_text = ""
            if item["chosen"] and isinstance(item["chosen"], list):
                for msg in item["chosen"]:
                    if msg["role"] == "assistant":
                        chosen_text = msg["content"]
                        break
            elif isinstance(item["chosen"], str): # If it's just a string
                chosen_text = item["chosen"]


            rejected_text = ""
            if item["rejected"] and isinstance(item["rejected"], list):
                for msg in item["rejected"]:
                    if msg["role"] == "assistant":
                        rejected_text = msg["content"]
                        break
            elif isinstance(item["rejected"], str): # If it's just a string
                rejected_text = item["rejected"]

            # Create a unique hash for the (prompt, chosen, rejected) triplet
            content_to_hash = f"{item['prompt']}_{chosen_text}_{rejected_text}"
            sample_id = get_content_hash(content_to_hash)

            selected_samples.append({
                "sample_id": sample_id, # Changed from sample_uuid
                "type": "train",
                "source_dataset": REDI_PAIRS_DATASET_PATH,
                "original_item_identifier": item.get("uuid", f"train_idx_{i}"), # Keep original identifier if present
                "prompt_text": item["prompt"],
                "chosen_assistant_text": chosen_text,
                "rejected_assistant_text": rejected_text,
                "chosen_full_conversation": item["chosen"], # Keep original structure
                "rejected_full_conversation": item["rejected"], # Keep original structure
            })
    else:
        print(f"Cannot sample {num_to_sample_train} from training dataset of size {len(train_ds)}")

    # 2. Sample from Test Datasets
    for key, conf in TEST_SAMPLES_CONFIG.items():
        print(f"Loading test dataset: {conf['name']}")
        try:
            test_ds = load_dataset(conf["name"], split=conf["split"])
        except Exception as e:
            print(f"Error loading test dataset {conf['name']}: {e}. Skipping.")
            continue

        if len(test_ds) < conf["count"]:
            print(f"Warning: Test dataset {key} has only {len(test_ds)} samples, requested {conf['count']}.")
            num_to_sample_test = len(test_ds)
        else:
            num_to_sample_test = conf["count"]
        
        if num_to_sample_test > 0 and len(test_ds) >= num_to_sample_test:
            sampled_test_indices = rng.sample(range(len(test_ds)), num_to_sample_test)
            for i in sampled_test_indices:
                item = test_ds[i]
                prompt_content = item[conf["column"]]
                sample_id = get_content_hash(prompt_content) # Hash the prompt for test questions

                selected_samples.append({
                    "sample_id": sample_id, # Changed from sample_uuid
                    "type": "test",
                    "source_dataset": key, 
                    "original_item_identifier": f"{key}_idx_{i}",
                    "prompt_text": prompt_content,
                    "chosen_assistant_text": None, 
                    "rejected_assistant_text": None,
                    "chosen_full_conversation": None,
                    "rejected_full_conversation": None,
                })
        else:
            print(f"Cannot sample {num_to_sample_test} from test dataset {key} of size {len(test_ds)}")


    # 3. Save to JSONL
    with open(OUTPUT_FILE, "w") as f:
        for sample in selected_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(selected_samples)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()