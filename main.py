#!/usr/bin/env python3
import sys
import os
import tempfile
import shutil
import json
from tqdm import tqdm
import boto3
import subprocess
import argparse
import yaml
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import login

def create_aws_translate_client(region_name=None):
    """
    Creates and returns an AWS Translate client.
    If region_name is None, environment variables or ~/.aws/config will be used.
    """
    return boto3.client("translate", region_name=region_name)

def translate_text(text: str, client, source_lang="en", target_lang="es", max_retries=3) -> str:
    """
    Translates a single text string using AWS Translate.
    Handles texts longer than AWS limits by splitting them.
    Includes retry logic for failed requests.
    """
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
            
            # Simple sentence splitting
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

def save_progress(work_dir, config, split, current_row, selected_cols, dataset_id=None, configs=None):
    """Save current progress to a JSON file"""
    progress = {
        'dataset_id': dataset_id,
        'configs': configs,
        'config': config,
        'split': split,
        'current_row': current_row,
        'selected_cols': selected_cols
    }
    with open(os.path.join(work_dir, 'progress.json'), 'w') as f:
        json.dump(progress, f)

def load_progress(work_dir):
    """Load progress from JSON file"""
    try:
        with open(os.path.join(work_dir, 'progress.json'), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def translate_batch(batch, selected_cols, client, max_retries):
    """Translates a batch of texts for selected columns"""
    for col in selected_cols:
        if col in batch:
            batch[col] = [translate_text(str(txt), client, max_retries=max_retries) for txt in batch[col]]
    return batch

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

def create_readme_and_yaml(work_dir, selected_configs, datasets_dict, dataset_id):
    """
    Creates README.md and dataset.yaml files with dataset information
    """
    # Dataset YAML Configurations
    dataset_yaml_content = {
        "configs": []
    }

    for cfg in selected_configs:
        cfg_name = cfg if cfg else "default"
        ds = datasets_dict.get(cfg_name, {})

        config_entry = {
            "config_name": cfg_name,
            "data_files": []
        }

        # Handle both Dataset and DatasetDict cases
        splits = ds.keys() if isinstance(ds, dict) else ['train']
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
    parser = argparse.ArgumentParser(description='Translate HuggingFace datasets')
    parser.add_argument('--test', action='store_true', help='Test mode: process only two subsets with 10 rows each')
    parser.add_argument('--retries', type=int, default=3, help='Number of retries for AWS Translate')
    parser.add_argument('--resume', action='store_true', help='Resume from last saved progress')
    args = parser.parse_args()

    # Get or create work directory
    if args.resume:
        work_dir = input("Enter the path to the previous work directory: ").strip()
        if not os.path.exists(work_dir):
            print("Work directory not found")
            return
    else:
        work_dir = tempfile.mkdtemp(prefix='translation_work_')

    # Load progress if resuming
    progress = load_progress(work_dir) if args.resume else None

    # Get dataset info if not resuming
    if not progress:
        dataset_id = input("Enter Hugging Face dataset ID: ").strip()
        if not dataset_id:
            print("No dataset ID provided")
            return

        try:
            configs = get_dataset_config_names(dataset_id)
            print("\nAvailable configs:", configs if configs else ["default"])
            
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
            if isinstance(ds, dict):
                if not ds:
                    print("Error: Dataset is empty")
                    return
                first_split = next(iter(ds.values()))
            else:
                first_split = ds
                
            columns = first_split.column_names
            if not columns:
                print("Error: No columns found in dataset")
                return
                
            print("\nAvailable columns:", columns)
            print("Column numbers:", {i: col for i, col in enumerate(columns)})
            
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

    try:
        # Process each config
        datasets_dict = {}
        start_config = progress['config'] if progress else selected_configs[0]
        config_idx = selected_configs.index(start_config)

        for config in tqdm(selected_configs[config_idx:], desc="Configs"):
            try:
                # Load dataset
                ds = load_dataset(dataset_id, config)
                cfg_name = config if config else "main"
                
                if args.test:
                    # In test mode, limit to 10 rows
                    if isinstance(ds, dict):
                        ds = {k: v.select(range(min(10, len(v)))) for k, v in ds.items()}
                    else:
                        ds = ds.select(range(min(10, len(ds))))
                
                datasets_dict[cfg_name] = ds
                
                # Process each split
                for split_name, split_ds in (ds.items() if isinstance(ds, dict) else [('train', ds)]):
                    print(f"\nTranslating {cfg_name} - {split_name}")
                    
                    try:
                        translated = split_ds.map(
                            lambda x: translate_batch(x, selected_cols, client, args.retries),
                            batched=True,
                            batch_size=8,
                            desc=f"Translating {split_name}"
                        )
                        
                        # Save progress after each batch
                        save_progress(work_dir, config, split_name, 0, selected_cols, dataset_id, selected_configs)
                        
                        # Save the translated data
                        save_path = os.path.join(work_dir, "configs", cfg_name)
                        os.makedirs(save_path, exist_ok=True)
                        translated.to_parquet(f"{save_path}/{split_name}.parquet")
                        
                    except KeyboardInterrupt:
                        print("\nProcess interrupted. Progress saved.")
                        return

            except Exception as e:
                print(f"Error processing {config}: {str(e)}")
                continue

        # Create necessary files
        print("\nCreating README.md and dataset.yaml...")
        create_readme_and_yaml(work_dir, selected_configs, datasets_dict, dataset_id)

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
        if not args.resume:
            backup_dir = os.path.join(os.getcwd(), "translated_dataset_backup")
            shutil.copytree(work_dir, backup_dir)
            print(f"Files saved locally to: {backup_dir}")
    finally:
        # Clean up temporary directories only if not resuming
        if not args.resume:
            try:
                shutil.rmtree(repo_dir)
                shutil.rmtree(work_dir)
            except:
                pass

if __name__ == "__main__":
    main()