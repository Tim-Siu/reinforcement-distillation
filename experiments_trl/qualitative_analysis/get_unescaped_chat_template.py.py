# get_unescaped_chat_template.py
import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Load a tokenizer and print its unescaped chat template.")
    parser.add_argument(
        "model_name_or_path",
        type=str,
        help="The name or path of the Hugging Face model/tokenizer.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow trusting remote code for the tokenizer.",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer for: {args.model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("\n--- Tokenizer's Chat Template ---")
        print(tokenizer.chat_template)
        print("--- End of Template ---")
        print("\nCopy the content between '--- Tokenizer's Chat Template ---' and '--- End of Template ---' into your .jinja file.")
    elif hasattr(tokenizer, 'default_chat_template') and tokenizer.default_chat_template:
        print("\n--- Tokenizer's Default Chat Template ---")
        print(tokenizer.default_chat_template)
        print("--- End of Template ---")
        print("\nCopy the content between '--- Tokenizer's Default Chat Template ---' and '--- End of Template ---' into your .jinja file.")
        print("Note: This model uses 'default_chat_template'. Ensure your VLLM/Transformers version handles this correctly if 'chat_template' is expected.")
    else:
        print("\nNo chat_template or default_chat_template found on the tokenizer object.")
        print("The model might not have a predefined Jinja chat template, or it might be constructed differently.")

if __name__ == "__main__":
    main()

# python get_unescaped_chat_template.py data/Qwen-7B-Math-R1-78k/checkpoint-3030 --trust-remote-code
