#!/usr/bin/env python3
import os
import argparse
import ast
from tqdm import tqdm
from datasets import load_dataset

def extract_value_from_conversation(conv_str):
    """Extract translatable content from conversation JSON string"""
    try:
        # Parse the conversation string as a Python literal
        conversations = ast.literal_eval(conv_str)
        
        # Sum up all the characters in 'value' fields
        char_count = 0
        for entry in conversations:
            if 'value' in entry and entry['value']:
                char_count += len(str(entry['value']))
        
        return char_count
    except Exception as e:
        print(f"Error parsing conversation: {str(e)}")
        # If parsing fails, return the original string length as fallback
        return len(conv_str)

def calculate_translation_cost(dataset_id, configs=None, columns=None, sample_size=None):
    """
    Calculate the estimated cost of translating the dataset using AWS Translate.
    
    Args:
        dataset_id: HuggingFace dataset ID
        configs: List of config names to process or None for all
        columns: List of column names to translate or None for all
        sample_size: Number of examples to sample (None for all)
        
    Returns:
        Total estimated cost in USD
    """
    # Load dataset
    print(f"Loading dataset: {dataset_id}")
    if configs:
        print(f"Selected configs: {configs}")
    
    try:
        # Get configs if not specified
        if not configs:
            try:
                from datasets import get_dataset_config_names
                available_configs = get_dataset_config_names(dataset_id)
                if available_configs:
                    configs = available_configs
                    print(f"Found configs: {configs}")
                else:
                    configs = [None]  # Default config
            except:
                configs = [None]  # Default config
        
        total_char_count = 0
        unique_system_prompts = set()
        system_prompt_chars = 0
        conversation_chars = 0
        other_chars = 0
        
        # Process each config
        for config in configs:
            ds = load_dataset(dataset_id, config)
            
            # Handle dataset dictionary structure
            if isinstance(ds, dict):
                # Multiple splits
                for split_name, split_ds in ds.items():
                    print(f"Processing {config if config else 'default'} - {split_name}")
                    
                    # Get columns if not specified
                    if not columns:
                        columns = split_ds.column_names
                        print(f"All columns will be analyzed: {columns}")
                    
                    # Apply sample if needed
                    if sample_size is not None and sample_size < len(split_ds):
                        split_ds = split_ds.select(range(min(sample_size, len(split_ds))))
                        print(f"Using sample of {len(split_ds)} examples")
                    else:
                        print(f"Processing all {len(split_ds)} examples")
                    
                    # Process each example
                    for example in tqdm(split_ds, desc=f"Analyzing {split_name}", unit="examples"):
                        # Process each column
                        for col in columns:
                            if col in example:
                                if col == "system":
                                    # System prompts get cached, so count each unique prompt only once
                                    system_text = str(example[col])
                                    if system_text not in unique_system_prompts:
                                        unique_system_prompts.add(system_text)
                                        system_chars = len(system_text)
                                        system_prompt_chars += system_chars
                                        total_char_count += system_chars
                                elif col == "conversations":
                                    # For conversations, extract and sum up 'value' fields
                                    conv_chars = extract_value_from_conversation(str(example[col]))
                                    conversation_chars += conv_chars
                                    total_char_count += conv_chars
                                else:
                                    # For other columns, just count the characters
                                    other_chars = len(str(example[col]))
                                    other_chars += other_chars
                                    total_char_count += other_chars
            else:
                # Single dataset (no splits)
                print(f"Processing {config if config else 'default'}")
                
                # Get columns if not specified
                if not columns:
                    columns = ds.column_names
                    print(f"All columns will be analyzed: {columns}")
                
                # Apply sample if needed
                if sample_size is not None and sample_size < len(ds):
                    ds = ds.select(range(min(sample_size, len(ds))))
                    print(f"Using sample of {len(ds)} examples")
                else:
                    print(f"Processing all {len(ds)} examples")
                
                # Process each example
                for example in tqdm(ds, desc="Analyzing", unit="examples"):
                    # Process each column
                    for col in columns:
                        if col in example:
                            if col == "system":
                                # System prompts get cached, so count each unique prompt only once
                                system_text = str(example[col])
                                if system_text not in unique_system_prompts:
                                    unique_system_prompts.add(system_text)
                                    system_chars = len(system_text)
                                    system_prompt_chars += system_chars
                                    total_char_count += system_chars
                            elif col == "conversations":
                                # For conversations, extract and sum up 'value' fields
                                conv_chars = extract_value_from_conversation(str(example[col]))
                                conversation_chars += conv_chars
                                total_char_count += conv_chars
                            else:
                                # For other columns, just count the characters
                                other_chars = len(str(example[col]))
                                other_chars += other_chars
                                total_char_count += other_chars
        
        # Calculate cost
        price_per_million = 15.00  # AWS price per million characters
        free_tier = 2000000  # 2 million free characters
        
        # Apply free tier
        billable_chars = max(0, total_char_count - free_tier)
        total_cost = (billable_chars / 1000000) * price_per_million
        
        # Print results
        print("\n===== Translation Cost Analysis =====")
        print(f"Total characters to translate: {total_char_count:,}")
        print(f"  - System prompts: {system_prompt_chars:,} (from {len(unique_system_prompts)} unique prompts)")
        print(f"  - Conversations: {conversation_chars:,}")
        print(f"  - Other fields: {other_chars:,}")
        print(f"Free tier characters: {free_tier:,}")
        print(f"Billable characters: {billable_chars:,}")
        print(f"AWS Translate cost (${price_per_million:.2f} per million chars): ${total_cost:.2f}")
        
        return total_cost
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Calculate AWS Translate cost for HuggingFace datasets')
    parser.add_argument('--dataset', required=True, help='HuggingFace dataset ID')
    parser.add_argument('--configs', help='Comma-separated config names (default: all)')
    parser.add_argument('--columns', help='Comma-separated column names to translate (default: all)')
    parser.add_argument('--sample', type=int, help='Analyze only a sample of examples and extrapolate (default: all)')
    parser.add_argument('--extrapolate', action='store_true', help='Extrapolate from sample to full dataset')
    args = parser.parse_args()
    
    # Process args
    configs = args.configs.split(',') if args.configs else None
    columns = args.columns.split(',') if args.columns else None
    
    # Calculate cost on sample
    cost = calculate_translation_cost(args.dataset, configs, columns, args.sample)
    
    # Extrapolate if requested
    if args.extrapolate and args.sample and cost is not None:
        try:
            # Get full dataset size
            ds = load_dataset(args.dataset, configs[0] if configs else None)
            if isinstance(ds, dict):
                total_examples = sum(len(split) for split in ds.values())
            else:
                total_examples = len(ds)
            
            # Calculate extrapolation factor
            factor = total_examples / args.sample
            extrapolated_cost = cost * factor
            
            print("\n===== Extrapolated Cost =====")
            print(f"Sample size: {args.sample:,} examples")
            print(f"Full dataset size: {total_examples:,} examples")
            print(f"Extrapolation factor: {factor:.2f}x")
            print(f"Estimated total cost: ${extrapolated_cost:.2f}")
        except Exception as e:
            print(f"Error extrapolating cost: {str(e)}")

if __name__ == "__main__":
    main()