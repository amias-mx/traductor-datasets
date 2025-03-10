#!/usr/bin/env python3
import os
import tempfile
import shutil
import json
import ast
import time
from tqdm import tqdm
import boto3
import subprocess
import argparse
import yaml
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from huggingface_hub import login
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def create_aws_translate_client(region_name=None):
    """Creates and returns an AWS Translate client."""
    return boto3.client("translate", region_name=region_name)

def translate_text(text, client, source_lang="en", target_lang="es", max_retries=3):
    """Translates a single text string using AWS Translate."""
    if not text or not isinstance(text, str):
        return text
        
    for attempt in range(max_retries):
        try:
            # AWS Translate has a 10,000 byte limit per request
            MAX_LENGTH = 9000  # Using 9000 to be safe with UTF-8 encoding
            
            # If text is within limits, translate directly
            if len(text) <= MAX_LENGTH:
                response = client.translate_text(
                    Text=text,
                    SourceLanguageCode=source_lang,
                    TargetLanguageCode=target_lang
                )
                return response["TranslatedText"]
                
            # For longer texts, split by sentences and translate in chunks
            chunks = []
            current_chunk = ""
            sentences = text.replace('\n', '. ').replace('? ', '. ').replace('! ', '. ').split('. ')
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                if len(current_chunk) + len(sentence) + 2 <= MAX_LENGTH:
                    current_chunk += sentence + '. '
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + '. '
                    
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            # Translate each chunk
            translated_chunks = []
            for chunk in chunks:
                response = client.translate_text(
                    Text=chunk,
                    SourceLanguageCode=source_lang,
                    TargetLanguageCode=target_lang
                )
                translated_chunks.append(response["TranslatedText"])
                
            return ' '.join(translated_chunks)
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error translating after {max_retries} attempts: {str(e)}")
                return text
            print(f"Attempt {attempt + 1} failed, retrying...")

def batch_translate_texts(texts, client, source_lang="en", target_lang="es", max_retries=3, batch_size=5):
    """
    Batch translate texts using AWS Translate BatchTranslate API.
    
    This function improves efficiency by combining multiple short texts
    into a single API call using a special delimiter.
    """
    if not texts:
        return texts
        
    # Filter out None or empty strings
    valid_texts = [(i, text) for i, text in enumerate(texts) if text and isinstance(text, str)]
    
    if not valid_texts:
        return texts
        
    result = list(texts)  # Create a copy of the original list
    
    # Process in batches
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i+batch_size]
        indices = [idx for idx, _ in batch]
        batch_texts = [text for _, text in batch]
        
        # Special delimiter unlikely to appear in normal text and preserved during translation
        delimiter = "|||SPLIT|||"
        
        # Only combine short texts that won't exceed limits when combined
        if all(len(text) < 1000 for _, text in batch) and sum(len(text) for _, text in batch) < 8000:
            # Combine texts with delimiter
            combined_text = delimiter.join(batch_texts)
            
            # Translate the combined text
            translated_combined = translate_text(combined_text, client, source_lang, target_lang, max_retries)
            
            # Split the translated text
            if translated_combined and delimiter in translated_combined:
                translated_texts = translated_combined.split(delimiter)
                
                # Update results
                for j, translated in enumerate(translated_texts):
                    if j < len(indices):
                        result[indices[j]] = translated
            else:
                # Fallback to individual translation if delimiter handling failed
                for idx, text in batch:
                    result[idx] = translate_text(text, client, source_lang, target_lang, max_retries)
        else:
            # For longer texts, translate individually
            for idx, text in batch:
                result[idx] = translate_text(text, client, source_lang, target_lang, max_retries)
    
    return result

def translate_python_literal_conversation(conv_str, client, max_retries=3):
    """Translates conversations in Python literal format."""
    try:
        # Parse using Python's built-in literal parser
        conversations = ast.literal_eval(conv_str)
        
        # Process each conversation entry
        for entry in conversations:
            if 'value' in entry and entry['value']:
                # Only translate the 'value' field
                entry['value'] = translate_text(entry['value'], client, max_retries=max_retries)
        
        # Return as Python literal string
        return str(conversations)
    except Exception as e:
        print(f"Error processing conversation: {str(e)}")
        return conv_str

def translate_batch(batch, selected_cols, client, max_retries, translation_cache=None, 
                use_parallel=False, workers=10, use_batching=False, batch_size=5):
    """
    Translates a batch of texts for selected columns
    Supports both parallel and sequential processing with optional caching
    """
    for col in selected_cols:
        if col in batch:
            # Check if column is 'conversations' which has Python literals
            if col == "conversations":
                # Print debug info for the first item in the batch
                if len(batch[col]) > 0:
                    try:
                        conv_str = str(batch[col][0])
                        print(f"Processing conversation (ID: {batch.get('id', ['unknown'])[0]}):")
                        # Print first 500 chars to avoid overwhelming the console
                        print(f"{conv_str[:500]}...")
                        if len(conv_str) > 500:
                            print(f"[...truncated, full length: {len(conv_str)} chars]")
                        print("-" * 80)
                    except Exception as e:
                        print(f"Error displaying conversation: {e}")
                
                # Translate conversations (with or without parallelization)
                if use_parallel:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = [
                            executor.submit(translate_python_literal_conversation, str(txt), client, max_retries)
                            for txt in batch[col]
                        ]
                        batch[col] = [future.result() for future in concurrent.futures.as_completed(futures)]
                else:
                    batch[col] = [
                        translate_python_literal_conversation(str(txt), client, max_retries=max_retries) 
                        for txt in batch[col]
                    ]
            
            # For system column with same value, translate once
            elif col == "system" and all(txt == batch[col][0] for txt in batch[col]):
                system_text = str(batch[col][0])
                
                # Check cache for system prompt
                translated = None
                cache_key = f"en:es:{system_text}"
                
                if translation_cache is not None and cache_key in translation_cache:
                    translated = translation_cache[cache_key]
                    print(f"USING CACHED system prompt (hash: {hash(system_text) % 10000}):")
                else:
                    print(f"NEW system prompt (hash: {hash(system_text) % 10000}):")
                    # Translate and cache
                    translated = translate_text(system_text, client, max_retries=max_retries)
                    if translation_cache is not None:
                        translation_cache[cache_key] = translated
                
                # Print first 100 chars to avoid overwhelming console
                print(f"First 100 chars: {system_text[:100]}...")
                print("-" * 80)
                
                batch[col] = [translated] * len(batch[col])
            else:
                # Process regular text fields
                texts = [str(txt) for txt in batch[col]]
                
                if use_batching:
                    # Use batching for short texts
                    batch[col] = batch_translate_texts(
                        texts, client, max_retries=max_retries, batch_size=batch_size
                    )
                elif use_parallel:
                    # Use parallel processing
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        future_to_idx = {
                            executor.submit(translate_text, text, client, "en", "es", max_retries): i
                            for i, text in enumerate(texts)
                        }
                        
                        result = [None] * len(texts)
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            try:
                                result[idx] = future.result()
                            except Exception as e:
                                print(f"Error in translation: {str(e)}")
                                result[idx] = texts[idx]  # Keep original on error
                        
                        batch[col] = result
                else:
                    # Simple sequential processing
                    batch[col] = [
                        translate_text(str(txt), client, max_retries=max_retries) 
                        for txt in batch[col]
                    ]
    
    return batch

def save_progress(work_dir, config, split, current_row, selected_cols, dataset_id=None, configs=None):
    """Save current progress to a JSON file"""
    progress = {
        'dataset_id': dataset_id,
        'configs': configs,
        'config': config,
        'split': split,
        'current_row': current_row,
        'selected_cols': selected_cols,
        'timestamp': time.time()
    }
    
    try:
        # First ensure the directory exists
        if not os.path.exists(work_dir):
            os.makedirs(work_dir, exist_ok=True)
            print(f"Created directory: {work_dir}")
            
        progress_file = os.path.join(work_dir, 'progress.json')
        with open(progress_file, 'w') as f:
            json.dump(progress, f)
        
        # Verify the file was written
        if os.path.exists(progress_file):
            print(f"Progress saved to {progress_file}")
        else:
            print(f"Warning: Progress file was not saved properly")
    except Exception as e:
        print(f"Error saving progress: {str(e)}")

def load_progress(work_dir):
    """Load progress from JSON file"""
    try:
        progress_file = os.path.join(work_dir, 'progress.json')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Progress file not found at: {progress_file}")
            return None
    except Exception as e:
        print(f"Error loading progress: {str(e)}")
        return None

def get_full_repo_url(repo_name):
    """Convert repo name to full Hugging Face URL"""
    return f"https://huggingface.co/datasets/{repo_name}"

def prepare_repository(repo_dir, repo_name, token=None):
    """Set up git repository with proper LFS handling"""
    os.makedirs(repo_dir, exist_ok=True)
    
    try:
        # Initialize git-lfs
        subprocess.run(['git', 'lfs', 'install'], check=True)
        
        # Clone the repository
        repo_url = get_full_repo_url(repo_name)
        try:
            subprocess.run(['git', 'clone', repo_url, repo_dir], check=True)
        except subprocess.CalledProcessError:
            # If clone fails, initialize new repository
            subprocess.run(['git', 'init'], cwd=repo_dir, check=True)
            subprocess.run(['git', 'remote', 'add', 'origin', repo_url], cwd=repo_dir, check=True)
        
        # Set up git-lfs tracking for parquet files
        subprocess.run(['git', 'lfs', 'track', "*.parquet"], cwd=repo_dir, check=True)
        
        # Setup git config if token is provided
        if token:
            subprocess.run(['git', 'config', 'user.name', 'Hugging Face'], cwd=repo_dir, check=True)
            subprocess.run(['git', 'config', 'user.email', 'no-reply@huggingface.co'], cwd=repo_dir, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up repository: {str(e)}")
        return False

def create_readme_and_yaml(work_dir, selected_configs, dataset_id):
    """Creates README.md and dataset.yaml files with dataset information"""
    # Dataset YAML Configurations
    dataset_yaml_content = {
        "configs": []
    }

    for cfg in selected_configs:
        cfg_name = cfg if cfg else "default"
        
        # Find all available splits for this config
        config_dir = os.path.join(work_dir, "configs", cfg_name)
        splits = []
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith(".parquet") and not file.startswith("chunk_"):
                    splits.append(file.replace(".parquet", ""))
        
        config_entry = {
            "config_name": cfg_name,
            "data_files": []
        }

        # Add found splits
        for split in splits:
            config_entry["data_files"].append({
                "split": split,
                "path": f"configs/{cfg_name}/{split}.parquet"
            })

        dataset_yaml_content["configs"].append(config_entry)

    # Serialize YAML content
    dataset_yaml_str = yaml.dump(dataset_yaml_content, allow_unicode=True, default_flow_style=False).strip()

    # YAML Header
    yaml_header = f"""
{dataset_yaml_str}
language:
- es
license: cc-by-4.0"""

    # README Content
    readme_content = f"""\
---
{yaml_header}
---
# Dataset Translation

This repository contains the Spanish translation of dataset subsets from 
[{dataset_id}](https://huggingface.co/datasets/{dataset_id}).

Each subset is preserved as a separate config, maintaining the original structure.

**Note**: The translations are generated using machine translation and may contain
typical automated translation artifacts.
"""

    # Write files
    with open(os.path.join(work_dir, "README.md"), "w", encoding='utf-8') as f:
        f.write(readme_content)
    with open(os.path.join(work_dir, "dataset.yaml"), "w", encoding='utf-8') as f:
        f.write(dataset_yaml_str)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate HuggingFace datasets with incremental saving')
    parser.add_argument('--test', action='store_true', help='Test mode: process only two subsets with 10 rows each')
    parser.add_argument('--retries', type=int, default=3, help='Number of retries for AWS Translate')
    parser.add_argument('--resume', action='store_true', help='Resume from a previous run')
    parser.add_argument('--chunk-size', type=int, default=100, help='Number of examples to process before saving')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing for translation')
    parser.add_argument('--batch', action='store_true', help='Use batch translation for multiple texts')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads for parallel processing')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of texts to combine in batch translation')
    parser.add_argument('--global-cache', action='store_true', help='Enable global caching of repetitive content', default=True)
    args = parser.parse_args()

    # Get or create work directory
    if args.resume:
        work_dir = input("Enter the path to the previous work directory: ").strip()
        # Remove quotes if they were added
        work_dir = work_dir.strip('"\'')
        
        if not os.path.exists(work_dir):
            print(f"Work directory not found: {work_dir}")
            print("Please check the path and try again.")
            return
        elif not os.path.isdir(work_dir):
            print(f"The path exists but is not a directory: {work_dir}")
            return
        else:
            print(f"Found work directory: {work_dir}")
    else:
        # Create temporary directory
        try:
            work_dir = tempfile.mkdtemp(prefix='translation_work_')
            
            # Test write access
            test_file = os.path.join(work_dir, 'test_file.txt')
            with open(test_file, 'w') as f:
                f.write('Test write access')
                
            # Verify the directory exists and has write access
            if os.path.exists(test_file):
                os.remove(test_file)
                print(f"\nCreated temporary work directory: {work_dir}")
                print("If process is interrupted, use this path with --resume to continue")
            else:
                print(f"Failed to verify write access to temporary directory")
                return
        except Exception as e:
            print(f"Error creating temporary directory: {str(e)}")
            print("Using current directory as fallback")
            work_dir = os.path.join(os.getcwd(), f"translation_work_{int(time.time())}")
            os.makedirs(work_dir, exist_ok=True)
            print(f"Created work directory: {work_dir}")

    # Load progress if resuming
    progress = None
    if args.resume:
        progress = load_progress(work_dir)
        if progress is None:
            print("No valid progress file found. Starting a new translation process.")
            print("Continue with this directory? (y/n):")
            if input().strip().lower() != 'y':
                print("Exiting. Please restart with a new directory.")
                return
        else:
            print(f"Resuming translation of dataset: {progress['dataset_id']}")
            print(f"Config: {progress['config']}")
            print(f"Split: {progress['split']}")
            print(f"Continuing from example: {progress['current_row']}")
            print(f"Columns: {progress['selected_cols']}")
            print("")

    # Get dataset info if not resuming
    if not progress:
        dataset_id = input("Enter Hugging Face dataset ID: ").strip()
        if not dataset_id:
            print("No dataset ID provided")
            return

        try:
            configs = get_dataset_config_names(dataset_id)
            print("\nAvailable configs:")
            for i, config in enumerate(configs):
                print(f"  [{i}] {config}")
            if not configs:
                print("  [default]")
            
            if args.test:
                print("\nTest mode: Will process only 2 configs with 10 rows each")
                selected_configs = configs[:2] if configs else [None]
            else:
                config_input = input("Enter config numbers to process (comma-separated) or 'all': ").strip()
                if config_input.lower() == 'all':
                    selected_configs = configs if configs else [None]
                elif config_input.strip():
                    indices = [int(i.strip()) for i in config_input.split(',')]
                    selected_configs = [configs[i] for i in indices]
                else:
                    selected_configs = [None]
                    
            print(f"\nSelected configs: {selected_configs}")
            
            # Get columns
            ds = load_dataset(dataset_id, selected_configs[0] if configs else None)
            
            # Simplify to handle dictionary datasets (with splits) consistently
            if isinstance(ds, dict):
                splits = list(ds.keys())
                first_split = ds[splits[0]]
            else:
                splits = ["train"]  # Default split name if not a dictionary
                first_split = ds
                
            print(f"\nAvailable splits: {splits}")
                
            columns = first_split.column_names
            if not columns:
                print("Error: No columns found in dataset")
                return
                
            print("\nAvailable columns:")
            for i, col in enumerate(columns):
                print(f"  [{i}] {col}")
            
            cols_input = input("Enter column numbers to translate (comma-separated) or 'all': ").strip()
            if not cols_input:
                print("Error: No columns selected")
                return
                
            if cols_input.lower() == 'all':
                selected_cols = columns
            else:
                indices = [int(i.strip()) for i in cols_input.split(',')]
                if not all(0 <= i < len(columns) for i in indices):
                    print("Error: Some column numbers are out of range")
                    return
                selected_cols = [columns[i] for i in indices]
                
            print(f"\nSelected columns: {selected_cols}")
            
            # Write initial progress
            save_progress(work_dir, selected_configs[0], splits[0], 0, 
                         selected_cols, dataset_id, selected_configs)
                
        except Exception as e:
            print(f"Error setting up dataset: {str(e)}")
            return
    else:
        dataset_id = progress['dataset_id']
        selected_configs = progress['configs']
        selected_cols = progress['selected_cols']

    # Get HF info
    repo_name = input("Enter HF repository name (username/repo): ").strip()
    token = input("Enter HF token (or press Enter if logged in): ").strip()

    # Initialize AWS client
    try:
        client = create_aws_translate_client()
    except Exception as e:
        print(f"AWS client error: {str(e)}")
        return

    # Create repository directory
    repo_dir = tempfile.mkdtemp(prefix='hf_repo_')
    
    # Initialize global translation cache for repetitive content
    translation_cache = {} if args.global_cache else None
    
    # Set the chunk size - smaller in test mode
    chunk_size = 5 if args.test else args.chunk_size
    print(f"Using chunk size of {chunk_size} examples")
    
    # Print parallelization info
    if args.parallel:
        print(f"Using parallel processing with {args.workers} workers")
        if args.batch:
            print(f"Using batch translation with batch size of {args.batch_size}")
    
    # Load the initial config to check system prompt before main processing
    if not progress and not args.test and args.global_cache:
        try:
            print("\nPre-caching system prompts...")
            ds = load_dataset(dataset_id, selected_configs[0] if selected_configs else None)
            
            # Handle both dictionary and non-dictionary datasets
            if isinstance(ds, dict):
                first_split = next(iter(ds.values()))
            else:
                first_split = ds
                
            # Check if 'system' is in the columns and pre-cache it
            if 'system' in selected_cols and 'system' in first_split.column_names:
                # Get the first row's system prompt
                system_text = str(first_split[0]['system'])
                print(f"Found system prompt (hash: {hash(system_text) % 10000}):")
                print(f"First 100 chars: {system_text[:100]}...")
                
                # Translate and cache
                translated = translate_text(system_text, client, max_retries=args.retries)
                cache_key = f"en:es:{system_text}"
                translation_cache[cache_key] = translated
                print("System prompt pre-cached for faster processing")
            else:
                print("No system prompt found for pre-caching")
                
        except Exception as e:
            print(f"Error during pre-caching: {e}")
            print("Continuing with normal processing...")
            
    print("\nBeginning translation process...")
    
    try:
        # Determine starting point from progress file
        start_config = progress['config'] if progress else selected_configs[0]
        config_idx = selected_configs.index(start_config)
        
        # Process each config
        for config in selected_configs[config_idx:]:
            # Load dataset
            print(f"\nLoading dataset with config: {config}")
            ds = load_dataset(dataset_id, config)
            
            # Make naming consistent
            config_name = config if config else "default"
            
            # Handle both dictionary and non-dictionary datasets
            if isinstance(ds, dict):
                splits_to_process = list(ds.keys())
            else:
                splits_to_process = ["train"]
                ds = {"train": ds}  # Convert to dictionary format for consistent handling
            
            # If resuming, find which split to start with
            if progress and progress['config'] == config:
                split_idx = splits_to_process.index(progress['split'])
                splits_to_process = splits_to_process[split_idx:]
            
            # Process each split
            for split_name in splits_to_process:
                print(f"\nTranslating {config_name} - {split_name}")
                split_ds = ds[split_name]
                
                # Apply test limit if needed
                if args.test:
                    split_ds = split_ds.select(range(min(10, len(split_ds))))
                
                # Get total examples
                total_examples = len(split_ds)
                print(f"Total examples in this split: {total_examples}")
                
                # Determine starting point
                start_idx = 0
                if progress and progress['config'] == config and progress['split'] == split_name:
                    start_idx = progress['current_row']
                    print(f"Resuming from example {start_idx}/{total_examples}")
                
                # Create save directory
                save_path = os.path.join(work_dir, "configs", config_name)
                os.makedirs(save_path, exist_ok=True)
                
                # Process in chunks
                for chunk_start in range(start_idx, total_examples, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_examples)
                    print(f"\nProcessing examples {chunk_start} to {chunk_end-1}")
                    
                    # Select subset for this chunk
                    chunk_ds = split_ds.select(range(chunk_start, chunk_end))
                    
                    try:
                        # Choose translation method based on args
                        translated_chunk = chunk_ds.map(
                            lambda x: translate_batch(
                                x, 
                                selected_cols, 
                                client, 
                                args.retries,
                                translation_cache=translation_cache,
                                use_parallel=args.parallel,
                                workers=args.workers,
                                use_batching=args.batch,
                                batch_size=args.batch_size
                            ),
                            batched=True,
                            batch_size=64,
                            desc=f"Translating {split_name} (examples {chunk_start}-{chunk_end-1})"
                        )
                        
                        # Save chunk to a temporary file
                        chunk_file = f"{save_path}/chunk_{split_name}_{chunk_start}_{chunk_end}.parquet"
                        translated_chunk.to_parquet(chunk_file)
                        print(f"Saved chunk to {chunk_file}")
                        
                        # Update progress
                        save_progress(work_dir, config, split_name, chunk_end, selected_cols, dataset_id, selected_configs)
                        print(f"Progress updated: {chunk_end}/{total_examples} examples processed")
                        
                    except KeyboardInterrupt:
                        print("\nProcess interrupted.")
                        save_progress(work_dir, config, split_name, chunk_start, selected_cols, dataset_id, selected_configs)
                        print(f"Progress saved at example {chunk_start}")
                        raise
                
                # After processing all chunks, combine them
                print(f"\nCombining chunks for {config_name} - {split_name}...")
                try:
                    # List all chunk files
                    chunk_files = []
                    for filename in os.listdir(save_path):
                        if filename.startswith(f"chunk_{split_name}_") and filename.endswith(".parquet"):
                            chunk_files.append(os.path.join(save_path, filename))
                    
                    # Sort chunk files by their starting index
                    chunk_files.sort(key=lambda x: int(x.split('_')[-2]))
                    
                    if chunk_files:
                        # Combine chunks into a single dataset
                        all_chunks = []
                        for chunk_file in chunk_files:
                            chunk_ds = load_dataset("parquet", data_files=chunk_file)
                            if isinstance(chunk_ds, dict):
                                chunk_ds = next(iter(chunk_ds.values()))
                            all_chunks.append(chunk_ds)
                        
                        # Concatenate and save
                        if all_chunks:
                            combined_ds = concatenate_datasets(all_chunks)
                            final_path = f"{save_path}/{split_name}.parquet"
                            combined_ds.to_parquet(final_path)
                            print(f"Saved complete translated data to {final_path}")
                            
                            # Clean up chunk files
                            for chunk_file in chunk_files:
                                os.remove(chunk_file)
                            print("Cleaned up temporary chunk files")
                except Exception as e:
                    print(f"Error combining chunks: {str(e)}")
                    print("Individual chunk files are preserved for manual recovery if needed")
        
        # Create the final metadata files
        print("\nCreating README.md and dataset.yaml...")
        create_readme_and_yaml(work_dir, selected_configs, dataset_id)
        
        # Handle repository
        print("\nPreparing to upload to Hugging Face...")
        if token:
            login(token=token)
        
        # Set up repository
        if not prepare_repository(repo_dir, repo_name, token):
            print("Failed to set up repository. Saving files locally instead.")
            backup_dir = os.path.join(os.getcwd(), "translated_dataset_backup")
            shutil.copytree(work_dir, backup_dir)
            print(f"Files saved to: {backup_dir}")
            return
        
        # Copy files to repository
        print("\nCopying files to repository...")
        for item in os.listdir(work_dir):
            src = os.path.join(work_dir, item)
            dst = os.path.join(repo_dir, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        # Push to Hugging Face
        print("\nPushing to Hugging Face...")
        try:
            subprocess.run(['git', 'add', '.'], cwd=repo_dir, check=True)
            subprocess.run(['git', 'commit', '-m', 'Add translated dataset'], cwd=repo_dir, check=True)
            subprocess.run(['git', 'push', '-u', 'origin', 'main'], cwd=repo_dir, check=True)
            print("\nUpload completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error pushing to repository: {str(e)}")
            backup_dir = os.path.join(os.getcwd(), "translated_dataset_backup")
            shutil.copytree(work_dir, backup_dir)
            print(f"Files saved locally to: {backup_dir}")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted. Progress saved.")
        print(f"\nTo resume translation, run:")
        print(f"python {os.path.basename(__file__)} --resume")
        print(f"And enter this work directory path when prompted:")
        print(f"{work_dir}")
        
        if not args.resume:
            backup_dir = os.path.join(os.getcwd(), "translated_dataset_backup")
            try:
                shutil.copytree(work_dir, backup_dir)
                print(f"Files also backed up locally to: {backup_dir}")
            except Exception as e:
                print(f"Could not create backup: {str(e)}")

if __name__ == "__main__":
    main()