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
"""
This script is used to decontaminate a dataset by checking for n-gram overlap with other datasets.
It uses the same approach presented in https://arxiv.org/abs/2501.19393,
as found in: https://github.com/simplescaling/s1/blob/main/data/decontaminate_util.py
It is a modified version of https://github.com/huggingface/open-r1/blob/52520a6713f8ebe03637cd0b75b9308946a33b7f/scripts/decontaminate.py

Usage:

python decontaminate.py \
    --dataset processed_datasets/openr1_math_raw_processed/openr1_math_sft_LandMV_filtered_shuffled_input \
    --split train \
    --ngram_size 8 \
    --problem_column messages
"""

import collections
import argparse
import os # Added for os.cpu_count()
from tqdm import tqdm
from datasets import load_dataset

def normalize_string(text: str) -> str:
    """Basic string normalization."""
    text = text.lower().strip()
    text = " ".join(text.split())
    return text

def word_ngrams(text: str, n: int) -> list:
    """Generate word-level n-grams from text."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

def build_ngram_lookup(documents: list[str], ngram_size: int = 8) -> dict[str, set[int]]:
    """Build ngram lookup for documents."""
    lookup = collections.defaultdict(set)
    for doc_id, document in enumerate(tqdm(documents, desc="Building n-gram lookup")):
        # Ensure document is a string
        doc_str = str(document) if not isinstance(document, str) else document
        normalized_text = normalize_string(doc_str)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(doc_id)
    return lookup

def build_ngram_single(document: str, ngram_size: int = 8) -> set[str]:
    normalized_text = normalize_string(document)
    ngrams = word_ngrams(normalized_text, ngram_size)
    return set(ngrams)

def get_problem_text_from_row(row: dict, problem_column_spec: str) -> str:
    """
    Extracts the problem text from a row based on the problem_column_spec.
    Handles direct column access and the specific "messages" list structure.
    """
    if not isinstance(row, dict):
        # print(f"Warning: Expected row to be a dict, got {type(row)}. Returning empty string.")
        return ""

    # Special handling for "messages" structure (like in your SFT datasets)
    if problem_column_spec == "messages":
        messages = row.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                    else: print(f"Warning: User message content is not a string: {type(content)}")
            print(f"Warning: 'user' role not found or no content in 'messages' list for row.")
            return "" # 'messages' list found, but no "user" role with string content
        else: print(f"Warning: '{problem_column_spec}' key found but is not a list: {type(messages)}")
        return "" # 'messages' key exists but is not a list or is missing

    # Default to direct column access
    elif problem_column_spec in row:
        text = row.get(problem_column_spec)
        if isinstance(text, str):
            return text
        else: print(f"Warning: Content in column '{problem_column_spec}' is not a string: {type(text)}. Converting to string.")
        return str(text) # Convert to string as a fallback

    print(f"Warning: Problem column_spec '{problem_column_spec}' not found or not usable in row. Row keys: {list(row.keys())}")
    return "" # Return empty string if problem text cannot be found

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decontaminate a local dataset by checking for n-gram overlap.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name or local path of the dataset to check for contamination (e.g., path to a directory saved by dataset.save_to_disk())."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Name of the dataset config to load (if applicable)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to check for contamination, defaults to `train`."
    )
    parser.add_argument(
        "--ngram_size",
        type=int,
        default=8,
        help="Size of n-grams to build, defaults to 8."
    )
    parser.add_argument(
        "--problem_column",
        type=str,
        default="problem",
        help=("Name of the column containing the text to check (e.g., 'problem'). "
              "For chat-formatted datasets where text is in a list under a key like 'messages', "
              "provide that key (e.g., 'messages'); the script will look for 'role':'user'.")
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None, # Default to None, will use os.cpu_count() later
        help="Number of processes to use for dataset mapping. Defaults to number of CPU cores."
    )
    args = parser.parse_args()

    num_processing_cores = args.num_proc if args.num_proc is not None else os.cpu_count()
    print(f"Using {num_processing_cores} processes for dataset operations.")

    print(f"Loading target dataset: {args.dataset}, config: {args.config}, split: {args.split}")
    # load_dataset with a path to a save_to_disk directory and a split will return the Dataset object directly.
    # If args.dataset is a Hub name, it also works as expected.
    try:
        ds = load_dataset(args.dataset, name=args.config, split=args.split, trust_remote_code=True) # Added trust_remote_code for general local scripts
    except Exception as e:
        print(f"Error loading dataset {args.dataset} (split: {args.split}): {e}")
        print("Please ensure the path is correct and the dataset structure is loadable by Hugging Face datasets.")
        raise
    print(f"Loaded target dataset with {len(ds)} rows.")

    print("Loading evaluation datasets...")
    eval_datasets = {
        "aime_2024": (load_dataset("HuggingFaceH4/aime_2024", split="train"), "problem"),
        "aime_2025": (load_dataset("yentinglin/aime_2025", split="train"), "problem"),
        "math_500": (load_dataset("HuggingFaceH4/MATH-500", split="test"), "problem"),
        # "gpqa": (load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True), "Question"),
        # "lcb": (
        #     load_dataset(
        #         "livecodebench/code_generation_lite", split="test", version_tag="v4_v5", trust_remote_code=True
        #     ),
        #     "question_content",
        # ),
    }

    ngram_lookups = {}
    for ds_name, (eval_ds_obj, eval_problem_col) in eval_datasets.items():
        print(f"Building n-gram lookup for {ds_name} using column '{eval_problem_col}'...")
        if eval_problem_col not in eval_ds_obj.column_names:
            print(f"Warning: Column '{eval_problem_col}' not found in eval_dataset '{ds_name}'. Skipping this eval dataset.")
            continue
        ngram_lookups[ds_name] = build_ngram_lookup(eval_ds_obj[eval_problem_col], ngram_size=args.ngram_size)


    # Add contamination flag columns to the dataset
    for eval_name, ngram_lookup in ngram_lookups.items():
        print(f"Checking for contamination against {eval_name}...")

        def find_contaminated(row):
            text_to_check = get_problem_text_from_row(row, args.problem_column)
            if not text_to_check: # If text is empty or could not be extracted
                row[f"contaminated_{eval_name}"] = False
                return row

            ngrams = build_ngram_single(text_to_check, ngram_size=args.ngram_size)
            row[f"contaminated_{eval_name}"] = any(n in ngram_lookup for n in ngrams)
            return row

        ds = ds.map(find_contaminated, num_proc=num_processing_cores)

    print("Identifying overall contaminated samples...")
    contaminated_indices = []
    contamination_col_names = [col for col in ds.column_names if col.startswith("contaminated_")]

    if not contamination_col_names:
        print("No contamination checks were performed (e.g., all eval datasets might have been skipped or had issues).")
    else:
        for i, example in enumerate(tqdm(ds, desc="Aggregating contamination results")):
            is_row_contaminated = False
            for col_name in contamination_col_names:
                if col_name in example and example[col_name]:
                    is_row_contaminated = True
                    break
            if is_row_contaminated:
                contaminated_indices.append(i)

    contaminated_count = len(contaminated_indices)

    print(f"\n--- Decontamination Report ---")
    print(f"Target dataset: {args.dataset} (config: {args.config}, split: {args.split})")
    print(f"N-gram size used: {args.ngram_size}")
    print(f"Column/spec checked in target dataset: '{args.problem_column}'")
    print(f"Total rows in target dataset: {len(ds)}")
    print(f"Number of contaminated rows found: {contaminated_count}")

    if contaminated_count > 0:
        if len(contaminated_indices) <= 100:
            print(f"Indices of contaminated rows: {contaminated_indices}")
        else:
            print(f"Indices of the first 100 contaminated rows: {contaminated_indices[:100]}")
            print(f"... and {len(contaminated_indices) - 100} more.")

    print("\nScript finished. The target dataset was analyzed for contamination; no data was changed or saved.")