#!/usr/bin/env python3
import os
import argparse
import ast
import tiktoken
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

def count_tokens(text):
    """Count tokens in a text string using tiktoken"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        return len(encoding.encode(text))
    except:
        # If tiktoken is not available, estimate tokens as words/0.75
        return int(len(text.split()) / 0.75)

def calculate_costs(dataset_id, configs=None, columns=None, sample_size=None):
    """
    Calculate the estimated cost of processing the dataset using AWS Translate and OpenAI GPT-4o mini.
    
    Args:
        dataset_id: HuggingFace dataset ID
        configs: List of config names to process or None for all
        columns: List of column names to translate or None for all
        sample_size: Number of examples to sample (None for all)
        
    Returns:
        Tuple of (AWS cost in USD, OpenAI cost in USD)
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
        
        # Character counts for AWS Translate
        total_char_count = 0
        unique_system_prompts = set()
        system_prompt_chars = 0
        conversation_chars = 0
        other_chars = 0
        
        # Token counts for OpenAI
        total_tokens = 0
        system_tokens = 0
        cached_tokens = 0  # For system prompts that would be cached
        input_tokens = 0
        output_tokens = 0
        
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
                                content_str = str(example[col])
                                
                                if col == "system":
                                    # System prompts get cached for OpenAI
                                    system_text = content_str
                                    token_count = count_tokens(system_text)
                                    
                                    # For AWS
                                    if system_text not in unique_system_prompts:
                                        unique_system_prompts.add(system_text)
                                        system_chars = len(system_text)
                                        system_prompt_chars += system_chars
                                        total_char_count += system_chars
                                    
                                    # For OpenAI
                                    system_tokens += token_count
                                    if system_text not in unique_system_prompts:
                                        # Count as regular input first time
                                        input_tokens += token_count
                                    else:
                                        # Count as cached input for subsequent occurrences
                                        cached_tokens += token_count
                                    total_tokens += token_count
                                    
                                elif col == "conversations":
                                    # For conversations, extract and sum up 'value' fields
                                    conv_chars = extract_value_from_conversation(content_str)
                                    conversation_chars += conv_chars
                                    total_char_count += conv_chars
                                    
                                    # For OpenAI, assume conversation text is split between input/output
                                    conv_tokens = count_tokens(content_str)
                                    # Estimate: typically in conversation data, user messages are inputs and 
                                    # assistant messages are outputs, with roughly 1:2 token ratio
                                    input_conv_tokens = int(conv_tokens * 0.33)  # Rough estimate
                                    output_conv_tokens = conv_tokens - input_conv_tokens
                                    input_tokens += input_conv_tokens
                                    output_tokens += output_conv_tokens
                                    total_tokens += conv_tokens
                                    
                                else:
                                    # For other columns, just count the characters/tokens
                                    other_char_count = len(content_str)
                                    other_chars += other_char_count
                                    total_char_count += other_char_count
                                    
                                    # For OpenAI, count as input tokens
                                    other_tokens = count_tokens(content_str)
                                    input_tokens += other_tokens
                                    total_tokens += other_tokens
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
                            content_str = str(example[col])
                            
                            if col == "system":
                                # System prompts get cached for OpenAI
                                system_text = content_str
                                token_count = count_tokens(system_text)
                                
                                # For AWS
                                if system_text not in unique_system_prompts:
                                    unique_system_prompts.add(system_text)
                                    system_chars = len(system_text)
                                    system_prompt_chars += system_chars
                                    total_char_count += system_chars
                                
                                # For OpenAI
                                system_tokens += token_count
                                if system_text not in unique_system_prompts:
                                    # Count as regular input first time
                                    input_tokens += token_count
                                else:
                                    # Count as cached input for subsequent occurrences
                                    cached_tokens += token_count
                                total_tokens += token_count
                                
                            elif col == "conversations":
                                # For conversations, extract and sum up 'value' fields
                                conv_chars = extract_value_from_conversation(content_str)
                                conversation_chars += conv_chars
                                total_char_count += conv_chars
                                
                                # For OpenAI, assume conversation text is split between input/output
                                conv_tokens = count_tokens(content_str)
                                # Estimate: typically in conversation data, user messages are inputs and 
                                # assistant messages are outputs, with roughly 1:2 token ratio
                                input_conv_tokens = int(conv_tokens * 0.33)  # Rough estimate
                                output_conv_tokens = conv_tokens - input_conv_tokens
                                input_tokens += input_conv_tokens
                                output_tokens += output_conv_tokens
                                total_tokens += conv_tokens
                                
                            else:
                                # For other columns, just count the characters/tokens
                                other_char_count = len(content_str)
                                other_chars += other_char_count
                                total_char_count += other_char_count
                                
                                # For OpenAI, count as input tokens
                                other_tokens = count_tokens(content_str)
                                input_tokens += other_tokens
                                total_tokens += other_tokens
        
        # Calculate AWS Translate cost
        aws_price_per_million = 15.00  # AWS price per million characters
        aws_free_tier = 2000000  # 2 million free characters
        
        # Apply free tier
        aws_billable_chars = max(0, total_char_count - aws_free_tier)
        aws_total_cost = (aws_billable_chars / 1000000) * aws_price_per_million
        
        # Calculate OpenAI GPT-4o mini cost
        openai_input_price = 0.150  # per million tokens
        openai_cached_price = 0.075  # per million tokens
        openai_output_price = 0.600  # per million tokens
        
        # No free tier for OpenAI
        openai_input_cost = (input_tokens / 1000000) * openai_input_price
        openai_cached_cost = (cached_tokens / 1000000) * openai_cached_price
        openai_output_cost = (output_tokens / 1000000) * openai_output_price
        openai_total_cost = openai_input_cost + openai_cached_cost + openai_output_cost
        
        # Print results in a side-by-side format
        print("\n===== Cost Analysis =====")
        
        # Print AWS details
        print("\nAWS Translation Details:")
        print(f"Total characters: {total_char_count:,}")
        print(f"  - System prompts: {system_prompt_chars:,} (from {len(unique_system_prompts)} unique prompts)")
        print(f"  - Conversations: {conversation_chars:,}")
        print(f"  - Other fields: {other_chars:,}")
        print(f"Free tier characters: {aws_free_tier:,}")
        print(f"Billable characters: {aws_billable_chars:,}")
        
        # Print OpenAI details
        print("\nOpenAI GPT-4o mini Details:")
        print(f"Total tokens: {total_tokens:,}")
        print(f"  - Regular input tokens: {input_tokens:,} (${openai_input_cost:.2f})")
        print(f"  - Cached input tokens: {cached_tokens:,} (${openai_cached_cost:.2f})")
        print(f"  - Output tokens: {output_tokens:,} (${openai_output_cost:.2f})")
        
        # Print side-by-side cost comparison
        print("\n┌───────────────────────────────────────────────────────────────┐")
        print("│ Service                  │ Price               │ Total Cost    │")
        print("├───────────────────────────────────────────────────────────────┤")
        print(f"│ AWS Translate            │ ${aws_price_per_million:.2f}/M chars      │ ${aws_total_cost:<14.2f} │")
        print(f"│ OpenAI GPT-4o mini       │ Various rates       │ ${openai_total_cost:<14.2f} │")
        print("└───────────────────────────────────────────────────────────────┘")
        
        # Show which service is cheaper
        print("\n===== Comparison =====")
        if aws_total_cost < openai_total_cost:
            diff = openai_total_cost - aws_total_cost
            print(f"AWS Translate is ${diff:.2f} cheaper ({(diff/openai_total_cost)*100:.1f}% savings)")
        else:
            diff = aws_total_cost - openai_total_cost
            print(f"OpenAI GPT-4o mini is ${diff:.2f} cheaper ({(diff/aws_total_cost)*100:.1f}% savings)")
        
        return aws_total_cost, openai_total_cost
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Calculate AWS Translate and OpenAI GPT-4o mini costs for HuggingFace datasets')
    parser.add_argument('--dataset', required=True, help='HuggingFace dataset ID')
    parser.add_argument('--configs', help='Comma-separated config names (default: all)')
    parser.add_argument('--columns', help='Comma-separated column names to process (default: all)')
    parser.add_argument('--sample', type=int, help='Analyze only a sample of examples and extrapolate (default: all)')
    parser.add_argument('--extrapolate', action='store_true', help='Extrapolate from sample to full dataset')
    args = parser.parse_args()
    
    # Process args
    configs = args.configs.split(',') if args.configs else None
    columns = args.columns.split(',') if args.columns else None
    
    # Calculate costs on sample
    aws_cost, openai_cost = calculate_costs(args.dataset, configs, columns, args.sample)
    
    # Extrapolate if requested
    if args.extrapolate and args.sample and aws_cost is not None and openai_cost is not None:
        try:
            # Get full dataset size
            ds = load_dataset(args.dataset, configs[0] if configs else None)
            if isinstance(ds, dict):
                total_examples = sum(len(split) for split in ds.values())
            else:
                total_examples = len(ds)
            
            # Calculate extrapolation factor
            factor = total_examples / args.sample
            extrapolated_aws_cost = aws_cost * factor
            extrapolated_openai_cost = openai_cost * factor
            
            print("\n===== Extrapolated Costs =====")
            print(f"Sample size: {args.sample:,} examples")
            print(f"Full dataset size: {total_examples:,} examples")
            print(f"Extrapolation factor: {factor:.2f}x")
            print(f"┌───────────────────────────────────────────────────────────────────┐")
            print(f"│ Service                  │ Sample Cost      │ Extrapolated Cost   │")
            print(f"├───────────────────────────────────────────────────────────────────┤")
            print(f"│ AWS Translate            │ ${aws_cost:<16.2f} │ ${extrapolated_aws_cost:<20.2f} │")
            print(f"│ OpenAI GPT-4o mini       │ ${openai_cost:<16.2f} │ ${extrapolated_openai_cost:<20.2f} │")
            print(f"└───────────────────────────────────────────────────────────────────┘")
            
            # Compare extrapolated costs
            print("\n===== Cost Comparison =====")
            if extrapolated_aws_cost < extrapolated_openai_cost:
                diff = extrapolated_openai_cost - extrapolated_aws_cost
                print(f"AWS Translate is ${diff:.2f} cheaper ({(diff/extrapolated_openai_cost)*100:.1f}% savings)")
            else:
                diff = extrapolated_aws_cost - extrapolated_openai_cost
                print(f"OpenAI GPT-4o mini is ${diff:.2f} cheaper ({(diff/extrapolated_aws_cost)*100:.1f}% savings)")
                
        except Exception as e:
            print(f"Error extrapolating cost: {str(e)}")

if __name__ == "__main__":
    main()