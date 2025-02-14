#!/usr/bin/env python3
import sys
import os
import tempfile
import shutil
from tqdm import tqdm
import time
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

def translate_text(text: str, client, source_lang="en", target_lang="es") -> str:
    """
    Translates a single text string using AWS Translate.
    Includes rate limiting and error handling.
    """
    if not text or not isinstance(text, str):
        return text
    try:
        # Rate limiting to avoid AWS Translate throttling
        time.sleep(0.1)
        response = client.translate_text(
            Text=text,
            SourceLanguageCode=source_lang,
            TargetLanguageCode=target_lang
        )
        return response["TranslatedText"]
    except Exception as e:
        print(f"Error translating: {str(e)}")
        return text

def translate_batch(batch, selected_cols, client):
    """
    Translates a batch of texts for selected columns.
    """
    for col in selected_cols:
        if col in batch:
            batch[col] = [translate_text(str(txt), client) for txt in batch[col]]
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

def save_dataset_as_parquet(dataset, save_path):
    """Save dataset in Parquet format"""
    # Clean any problematic columns
    if '_format_kwargs' in dataset.features:
        dataset = dataset.remove_columns(['_format_kwargs'])
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save as parquet
    dataset.to_parquet(f"{save_path}/data.parquet")

def create_dataset_yaml(work_dir, selected_configs, ds):
    """Creates the dataset.yaml file with proper config structure"""
    yaml_content = {
        "configs": [
            {
                "config_name": cfg if cfg else "main",
                "data_files": {
                    split: f"{cfg if cfg else 'main'}/{split}/data.parquet"
                    for split in (ds.keys() if isinstance(ds, dict) else ['train'])
                }
            }
            for cfg in selected_configs
        ]
    }

    yaml_path = os.path.join(work_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True)

def create_readme(work_dir, dataset_id):
    """Creates a README.md file with dataset information"""
    readme_content = f"""\
---
license: cc-by-4.0
language:
- es
---
# Dataset Translation

This repository contains the Spanish translation of dataset subsets from 
[{dataset_id}](https://huggingface.co/datasets/{dataset_id}).

Each subset is preserved as a separate config, maintaining the original structure.

**Note**: The translations are generated using machine translation and may contain
typical automated translation artifacts.
"""
    
    readme_path = os.path.join(work_dir, "README.md")
    with open(readme_path, "w", encoding='utf-8') as f:
        f.write(readme_content)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate HuggingFace datasets')
    parser.add_argument('--test', action='store_true', help='Test mode: process only one subset')
    args = parser.parse_args()

    # 1. Get dataset info
    dataset_id = input("Enter Hugging Face dataset ID: ").strip()
    if not dataset_id:
        print("No dataset ID provided")
        return

    # 2. Get configs
    try:
        configs = get_dataset_config_names(dataset_id)
        
        if args.test:
            print("\nTest mode: Will only process the first config")
            selected_configs = [configs[0]] if configs else [None]
        else:
            selected_configs = configs if configs else [None]
            
        print(f"\nSelected configs: {selected_configs}")
        
    except Exception as e:
        print(f"Error getting configs: {str(e)}")
        return

    # 3. Load first config to get columns
    try:
        ds = load_dataset(dataset_id, selected_configs[0] if configs else None)
        first_split = next(iter(ds.values())) if isinstance(ds, dict) else ds
        columns = first_split.column_names
        print("\nAvailable columns:", columns)
        cols_to_translate = input("Enter column numbers to translate (comma-separated) or 'all': ")
        selected_cols = columns if cols_to_translate.lower() == 'all' else \
                       [columns[int(i)] for i in cols_to_translate.split(',')]
        
        print(f"\nSelected columns: {selected_cols}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # 4. Get HF info
    repo_name = input("Enter HF repository name (username/repo): ").strip()
    token = input("Enter HF token (or press Enter if logged in): ").strip()

    # 5. Initialize AWS client
    try:
        client = create_aws_translate_client()
    except Exception as e:
        print(f"AWS client error: {str(e)}")
        return

    # 6. Create temporary directories
    work_dir = tempfile.mkdtemp(prefix='translation_work_')
    repo_dir = tempfile.mkdtemp(prefix='hf_repo_')

    try:
        # 7. Process each config
        for config in tqdm(selected_configs, desc="Configs"):
            try:
                # Load dataset
                ds = load_dataset(dataset_id, config)
                
                # Process each split
                for split_name, split_ds in (ds.items() if isinstance(ds, dict) else [('train', ds)]):
                    print(f"\nTranslating {config if config else 'main'} - {split_name}")
                    translated = split_ds.map(
                        lambda x: translate_batch(x, selected_cols, client),
                        batched=True,
                        batch_size=8,
                        desc=f"Translating {split_name}"
                    )
                    
                    # Save in Parquet format
                    save_path = os.path.join(work_dir, 
                                           config if config else 'main',
                                           split_name)
                    save_dataset_as_parquet(translated, save_path)

            except Exception as e:
                print(f"Error processing {config}: {str(e)}")
                continue

        # 8. Create necessary files
        print("\nCreating dataset.yaml...")
        create_dataset_yaml(work_dir, selected_configs, ds)
        
        print("Creating README.md...")
        create_readme(work_dir, dataset_id)

        # 9. Handle repository
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

    finally:
        # Clean up
        try:
            shutil.rmtree(work_dir)
            shutil.rmtree(repo_dir)
        except:
            pass

if __name__ == "__main__":
    main()