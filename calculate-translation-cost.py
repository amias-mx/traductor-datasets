#!/usr/bin/env python3
import os
import argparse
import ast
import tiktoken # Preferred tokenizer
import warnings # To warn about fallbacks
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from typing import Optional, List, Tuple, Any, Set

# --- Constants ---
# Last checked: 2025-04-03
AWS_PRICE_PER_MILLION_CHARS = 15.00
AWS_FREE_TIER_CHARS = 2_000_000
# OpenAI GPT-4o mini Pricing (replace if model changes)
# Source: User prompt based on current pricing as of 2025-04-03
OPENAI_INPUT_PRICE_PER_MILLION_TOKENS = 0.150
OPENAI_CACHED_PRICE_PER_MILLION_TOKENS = 0.075 # For subsequent identical system prompts
OPENAI_OUTPUT_PRICE_PER_MILLION_TOKENS = 0.600

# Default column names for special handling
SYSTEM_COLUMN_NAME = "system"
CONVERSATION_COLUMN_NAME = "conversations"

# --- Helper Functions ---

def extract_value_from_conversation(conv_str: str) -> int:
    """
    Extract translatable character count from conversation JSON/Python literal string.
    Only counts characters within 'value' fields.
    """
    try:
        # Safely parse the string as a Python literal (list of dicts)
        conversations = ast.literal_eval(conv_str)
        if not isinstance(conversations, list):
            raise ValueError("Parsed conversation is not a list.")

        char_count = 0
        for entry in conversations:
            if isinstance(entry, dict) and 'value' in entry and entry['value']:
                # Ensure value is treated as string for length calculation
                char_count += len(str(entry['value']))

        return char_count
    except (ValueError, SyntaxError, TypeError) as e:
        # Warn if parsing fails, estimate may be less accurate
        warnings.warn(
            f"Could not parse conversation string: {e}. "
            f"Falling back to full string length for char count. "
            f"String snippet: '{conv_str[:100]}...'"
        )
        # Fallback: return the original string length
        return len(conv_str)
    except Exception as e:
        # Catch other unexpected errors
        warnings.warn(
            f"Unexpected error parsing conversation: {e}. "
            f"Falling back to full string length. "
            f"String snippet: '{conv_str[:100]}...'"
         )
        return len(conv_str)


def count_tokens(text: str) -> int:
    """
    Count tokens in a text string using tiktoken (cl100k_base for GPT-4/GPT-4o).
    Provides a rough fallback if tiktoken is unavailable.
    """
    try:
        # cl100k_base is the encoding for GPT-4, GPT-3.5-Turbo, text-embedding-ada-002, GPT-4o
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        # Warn if tiktoken fails or is not available
        warnings.warn(
            f"tiktoken failed ({e}), using rough fallback for token count (words / 0.75). "
            f"This estimate can be inaccurate."
         )
        # Fallback: estimate tokens based on word count (very rough)
        words = text.split()
        if not words:
            return 0
        return int(len(words) / 0.75)

# --- Core Calculation ---

def calculate_costs(
    dataset_id: str,
    configs: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
    sample_size: Optional[int] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate the estimated cost of processing the dataset using AWS Translate
    and OpenAI GPT-4o mini.

    Args:
        dataset_id: HuggingFace dataset ID.
        configs: List of config names to process or None for all.
        columns: List of column names to analyze or None for all found.
        sample_size: Number of examples to sample (None for all).

    Returns:
        Tuple of (Estimated AWS cost in USD, Estimated OpenAI cost in USD),
        or (None, None) if an error occurs.
    """
    print(f"Loading dataset: {dataset_id}")
    if configs:
        print(f"Selected configs: {configs}")

    try:
        # Get dataset configs if not specified
        if not configs:
            try:
                available_configs = get_dataset_config_names(dataset_id)
                if available_configs:
                    configs = available_configs
                    print(f"Found configs: {configs}")
                else:
                    configs = [None] # Indicates default config
                    print("No named configs found, using default.")
            except Exception as e:
                print(f"Could not automatically get configs: {e}. Assuming default config.")
                configs = [None]

        # --- Initialize Counters ---
        # Character counts for AWS Translate
        total_char_count = 0
        system_prompt_chars = 0
        conversation_chars = 0
        other_chars = 0
        unique_system_prompts_aws: Set[str] = set() # Track unique prompts for AWS count

        # Token counts for OpenAI
        total_tokens = 0          # Total tokens (input + output)
        regular_input_tokens = 0  # Input tokens not part of cached system prompts
        cached_input_tokens = 0   # Input tokens considered "cached" (repeated system prompts)
        output_tokens = 0         # Estimated output tokens
        unique_system_prompts_openai: Set[str] = set() # Track unique prompts for OpenAI cost

        # --- Process Dataset ---
        for config in configs:
            config_name_str = config if config else "default"
            try:
                ds = load_dataset(dataset_id, config)
            except Exception as e:
                print(f"\nError loading config '{config_name_str}': {e}. Skipping.")
                continue

            # Handle dataset dictionary structure (multiple splits) vs single dataset
            splits_to_process = {}
            if isinstance(ds, dict):
                splits_to_process = ds
            else:
                # Assume 'train' split if not a dict
                splits_to_process["train"] = ds

            for split_name, split_ds in splits_to_process.items():
                print(f"\nProcessing config: '{config_name_str}', split: '{split_name}'")

                current_columns = columns # Use specified columns if provided
                if not current_columns:
                    current_columns = split_ds.column_names
                    # Warning: Using columns from the *first* split. Schemas might differ.
                    print(f"Analyzing all columns found in this split: {current_columns}")
                    if columns is None and len(splits_to_process) > 1:
                       warnings.warn("Column list derived from the first split. "
                                     "Analysis might be inaccurate if subsequent splits have different columns.")

                # Apply sampling if requested
                num_examples = len(split_ds)
                if sample_size is not None and sample_size < num_examples:
                    split_ds = split_ds.select(range(sample_size))
                    print(f"Using sample of {len(split_ds)} examples (out of {num_examples})")
                else:
                    print(f"Processing all {num_examples} examples")

                # Process each example
                for example in tqdm(split_ds, desc=f"Analyzing {split_name}", unit=" examples"):
                    for col in current_columns:
                        if col in example and example[col] is not None:
                            # Ensure content is string
                            content_str = str(example[col])
                            char_count_col = len(content_str)

                            # --- AWS Character Counting ---
                            total_char_count += char_count_col
                            if col == SYSTEM_COLUMN_NAME:
                                # Only count unique system prompts once for AWS estimate
                                if content_str not in unique_system_prompts_aws:
                                    unique_system_prompts_aws.add(content_str)
                                    system_prompt_chars += char_count_col
                                # Note: Subsequent identical prompts still add to total_char_count
                            elif col == CONVERSATION_COLUMN_NAME:
                                # For conversations, count only 'value' fields
                                conv_val_chars = extract_value_from_conversation(content_str)
                                conversation_chars += conv_val_chars
                                # Adjust total count: subtract full string, add value chars
                                total_char_count -= char_count_col
                                total_char_count += conv_val_chars
                            else:
                                other_chars += char_count_col

                            # --- OpenAI Token Counting ---
                            token_count_col = count_tokens(content_str)
                            total_tokens += token_count_col

                            if col == SYSTEM_COLUMN_NAME:
                                # For OpenAI, first occurrence is regular input, subsequent are cached
                                if content_str not in unique_system_prompts_openai:
                                    unique_system_prompts_openai.add(content_str)
                                    regular_input_tokens += token_count_col
                                else:
                                    cached_input_tokens += token_count_col
                            elif col == CONVERSATION_COLUMN_NAME:
                                # Estimate conversation tokens: ~1/3 input, ~2/3 output
                                # This is a heuristic and actual ratio depends on conversation flow.
                                input_conv_tokens = int(token_count_col * 0.33)
                                output_conv_tokens = token_count_col - input_conv_tokens
                                regular_input_tokens += input_conv_tokens
                                output_tokens += output_conv_tokens
                            else:
                                # Other columns are treated as input tokens
                                regular_input_tokens += token_count_col

        # --- Calculate Costs ---

        # AWS Translate Cost
        # Apply free tier
        aws_billable_chars = max(0, total_char_count - AWS_FREE_TIER_CHARS)
        aws_total_cost = (aws_billable_chars / 1_000_000) * AWS_PRICE_PER_MILLION_CHARS

        # OpenAI GPT-4o mini Cost
        openai_regular_input_cost = (regular_input_tokens / 1_000_000) * OPENAI_INPUT_PRICE_PER_MILLION_TOKENS
        openai_cached_input_cost = (cached_input_tokens / 1_000_000) * OPENAI_CACHED_PRICE_PER_MILLION_TOKENS
        openai_output_cost = (output_tokens / 1_000_000) * OPENAI_OUTPUT_PRICE_PER_MILLION_TOKENS
        openai_total_cost = openai_regular_input_cost + openai_cached_input_cost + openai_output_cost

        # --- Print Results ---
        print("\n" + "=" * 25 + " Cost Analysis " + "=" * 25)

        # AWS details
        print("\n--- AWS Translation Details ---")
        print(f"Total characters processed:      {total_char_count:15,}")
        # The following breakdown is approximate if conversation parsing failed
        print(f"  Unique system prompt chars:    {system_prompt_chars:15,} (from {len(unique_system_prompts_aws):,} unique prompts)")
        print(f"  Conversation 'value' chars:    {conversation_chars:15,} (estimated)")
        print(f"  Other field chars:             {other_chars:15,}")
        print("-" * 35)
        print(f"AWS Free Tier characters:        {AWS_FREE_TIER_CHARS:15,}")
        print(f"Billable characters (estimated): {aws_billable_chars:15,}")
        print(f"AWS Price per Million Chars:     ${AWS_PRICE_PER_MILLION_CHARS:.2f}")

        # OpenAI details
        print("\n--- OpenAI GPT-4o mini Details ---")
        print(f"Total tokens processed (est.):   {total_tokens:15,}")
        print(f"  Regular input tokens (est.):   {regular_input_tokens:15,} (${openai_regular_input_cost:,.2f})")
        print(f"  Cached input tokens (est.):    {cached_input_tokens:15,} (${openai_cached_input_cost:,.2f}) (Repeated system prompts)")
        print(f"  Output tokens (est.):          {output_tokens:15,} (${openai_output_cost:,.2f}) (From conversations)")
        print("-" * 35)
        print(f"OpenAI Input Price / 1M tokens:  ${OPENAI_INPUT_PRICE_PER_MILLION_TOKENS:.3f}")
        print(f"OpenAI Cached Price / 1M tokens: ${OPENAI_CACHED_PRICE_PER_MILLION_TOKENS:.3f}")
        print(f"OpenAI Output Price / 1M tokens: ${OPENAI_OUTPUT_PRICE_PER_MILLION_TOKENS:.3f}")

        # Side-by-side cost comparison
        print("\n" + "=" * 27 + " Summary " + "=" * 28)
        print("┌───────────────────────────────────────────────────────────────┐")
        print("│ Service                 │ Price Basis           │ Total Cost Est. │")
        print("├───────────────────────────────────────────────────────────────┤")
        print(f"│ AWS Translate           │ ${AWS_PRICE_PER_MILLION_CHARS:>5.2f} / M Chars      │ ${aws_total_cost:<15,.2f} │")
        print(f"│ OpenAI GPT-4o mini      │ Various Token Rates   │ ${openai_total_cost:<15,.2f} │")
        print("└───────────────────────────────────────────────────────────────┘")

        # Comparison verdict
        print("\n--- Comparison Verdict ---")
        if aws_total_cost == openai_total_cost:
            print("Both services have an estimated cost of ${aws_total_cost:,.2f}.")
        elif aws_total_cost < openai_total_cost:
            diff = openai_total_cost - aws_total_cost
            savings_percent = (diff / openai_total_cost) * 100 if openai_total_cost > 0 else 0
            print(f"AWS Translate is estimated to be ${diff:,.2f} cheaper ({savings_percent:.1f}% savings vs OpenAI).")
        else:
            diff = aws_total_cost - openai_total_cost
            savings_percent = (diff / aws_total_cost) * 100 if aws_total_cost > 0 else 0
            print(f"OpenAI GPT-4o mini is estimated to be ${diff:,.2f} cheaper ({savings_percent:.1f}% savings vs AWS).")
        print("-" * 26)

        return aws_total_cost, openai_total_cost

    except FileNotFoundError as e:
         print(f"\nError: Dataset '{dataset_id}' not found. Please check the ID. Details: {e}")
         return None, None
    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None, None

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(
        description='Estimate AWS Translate and OpenAI GPT-4o mini costs for HuggingFace datasets.'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='HuggingFace dataset ID (e.g., username/dataset_name)'
    )
    parser.add_argument(
        '--configs',
        help='Comma-separated config names to analyze (default: all available configs)'
    )
    parser.add_argument(
        '--columns',
        help='Comma-separated column names to analyze (default: all columns found in the first split)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        help='Analyze only the first N examples per split'
    )
    parser.add_argument(
        '--extrapolate',
        action='store_true',
        help='Extrapolate costs from the sample to the full dataset size (requires --sample)'
    )
    args = parser.parse_args()

    # Process arguments
    configs_list = args.configs.split(',') if args.configs else None
    columns_list = args.columns.split(',') if args.columns else None

    # Calculate costs based on sample or full data
    aws_cost, openai_cost = calculate_costs(
        args.dataset,
        configs_list,
        columns_list,
        args.sample
    )

    # Extrapolate if requested and possible
    if args.extrapolate and args.sample and aws_cost is not None and openai_cost is not None:
        if args.sample <= 0:
             print("\nCannot extrapolate with sample size <= 0.")
             return

        try:
            print("\nCalculating full dataset size for extrapolation...")
            # Note: This calculation assumes the structure used during sampling applies
            # It might be inaccurate if only specific configs/splits were sampled non-exhaustively
            total_examples = 0
            processed_configs = configs_list if configs_list else []
            if not processed_configs: # If no configs specified, try getting all
                 try:
                      available_configs = get_dataset_config_names(args.dataset)
                      processed_configs = available_configs if available_configs else [None]
                 except:
                      processed_configs = [None] # Fallback to default

            for config in processed_configs:
                 config_name_str = config if config else "default"
                 try:
                    ds_info = load_dataset(args.dataset, config, download_mode='reuse_cache_if_exists')
                    if isinstance(ds_info, dict):
                         total_examples += sum(len(split) for split in ds_info.values())
                    else:
                         total_examples += len(ds_info)
                 except Exception as e:
                    print(f"Warning: Could not load config '{config_name_str}' to get full size: {e}")
                    print("Extrapolation might be inaccurate if sampled configs differ significantly.")

            if total_examples == 0:
                print("Could not determine total dataset size for extrapolation.")
                return

            # Check if sample size exceeds total examples (can happen if sampling across multiple small splits)
            actual_sample_size = args.sample * len(processed_configs) # Rough upper bound if sampling N from each config
            # A more precise way would be to sum min(args.sample, len(split)) across all splits processed during calculate_costs
            # For simplicity, we use args.sample directly assuming it was applied per split/config.
            if args.sample >= total_examples:
                 print("\nSample size is greater than or equal to total examples. No extrapolation needed.")
            else:
                 # Calculate extrapolation factor
                 # This assumes the sample is representative of the whole dataset.
                 factor = total_examples / args.sample
                 extrapolated_aws_cost = aws_cost * factor
                 extrapolated_openai_cost = openai_cost * factor

                 print("\n" + "=" * 22 + " Extrapolated Costs " + "=" * 22)
                 print(f"Sample size per split/config: {args.sample:,}")
                 print(f"Estimated full dataset size:  {total_examples:,} examples")
                 print(f"Extrapolation factor:         {factor:.2f}x")
                 print("(Note: Assumes sample is representative)")
                 print("┌─────────────────────────────────────────────────────────────────────┐")
                 print("│ Service                 │ Sample Cost     │ Extrapolated Cost Est.  │")
                 print("├─────────────────────────────────────────────────────────────────────┤")
                 print(f"│ AWS Translate           │ ${aws_cost:<15,.2f} │ ${extrapolated_aws_cost:<23,.2f} │")
                 print(f"│ OpenAI GPT-4o mini      │ ${openai_cost:<15,.2f} │ ${extrapolated_openai_cost:<23,.2f} │")
                 print("└─────────────────────────────────────────────────────────────────────┘")

                 # Compare extrapolated costs
                 print("\n--- Extrapolated Comparison Verdict ---")
                 if extrapolated_aws_cost == extrapolated_openai_cost:
                     print(f"Both services have an extrapolated estimated cost of ${extrapolated_aws_cost:,.2f}.")
                 elif extrapolated_aws_cost < extrapolated_openai_cost:
                     diff = extrapolated_openai_cost - extrapolated_aws_cost
                     savings_percent = (diff / extrapolated_openai_cost) * 100 if extrapolated_openai_cost > 0 else 0
                     print(f"AWS Translate is estimated to be ${diff:,.2f} cheaper ({savings_percent:.1f}% savings vs OpenAI).")
                 else:
                     diff = extrapolated_aws_cost - extrapolated_openai_cost
                     savings_percent = (diff / extrapolated_aws_cost) * 100 if extrapolated_aws_cost > 0 else 0
                     print(f"OpenAI GPT-4o mini is estimated to be ${diff:,.2f} cheaper ({savings_percent:.1f}% savings vs AWS).")
                 print("-" * 37)

        except Exception as e:
            print(f"\nError during cost extrapolation: {e}")

if __name__ == "__main__":
    main()