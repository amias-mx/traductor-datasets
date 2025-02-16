#!/usr/bin/env python3
import sys
import os
import tempfile
import shutil
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

def translate_text(text: str, client, source_lang="en", target_lang="es") -> str:
    """
    Translates a single text string using AWS Translate.
    Handles texts longer than AWS limits by splitting them.
    """
    if not text or not isinstance(text, str):
        return text
        
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

def create_readme_and_yaml(work_dir, selected_configs, datasets_dict, dataset_id):
    """
    Creates a README.md file and a dataset.yaml file with dataset information.
    Combines functionality to avoid duplication.
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

    # Write README.md
    readme_path = os.path.join(work_dir, "README.md")
    with open(readme_path, "w", encoding='utf-8') as f:
        f.write(readme_content)

    # Write dataset.yaml
    yaml_path = os.path.join(work_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding='utf-8') as f:
        f.write(dataset_yaml_str)


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
        print("\nAvailable configs:", configs if configs else ["default"])
        
        if args.test:
            print("\nTest mode: Will only process up to 2 configs")
            selected_configs = configs[:2] if configs else [None]
        else:
            config_input = input("Enter config numbers to process (comma-separated) or 'all': ").strip()
            if config_input.lower() == 'all':
                selected_configs = configs if configs else [None]
            elif config_input.strip():
                try:
                    indices = [int(i.strip()) for i in config_input.split(',')]
                    selected_configs = [configs[i] for i in indices]
                except (ValueError, IndexError):
                    print("Invalid config selection")
                    return
            else:
                selected_configs = [None]  # Default config
            
        print(f"\nSelected configs: {selected_configs}")
        
    except Exception as e:
        print(f"Error getting configs: {str(e)}")
        return

    # 3. Load first config to get columns
    try:
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
        
        cols_to_translate = input("Enter column numbers to translate (comma-separated) or 'all': ").strip()
        if not cols_to_translate:
            print("Error: No columns selected")
            return
            
        try:
            if cols_to_translate.lower() == 'all':
                selected_cols = columns
            else:
                indices = [int(i.strip()) for i in cols_to_translate.split(',')]
                if not all(0 <= i < len(columns) for i in indices):
                    print("Error: Some column numbers are out of range")
                    return
                selected_cols = [columns[i] for i in indices]
        except ValueError:
            print("Error: Invalid column numbers")
            return
            
        if not selected_cols:
            print("Error: No valid columns selected")
            return
            
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
        datasets_dict = {}
        for config in tqdm(selected_configs, desc="Configs"):
            try:
                # Load dataset
                ds = load_dataset(dataset_id, config)
                cfg_name = config if config else "main"
                datasets_dict[cfg_name] = ds
                
                # Process each split
                for split_name, split_ds in (ds.items() if isinstance(ds, dict) else [('train', ds)]):
                    print(f"\nTranslating {cfg_name} - {split_name}")
                    translated = split_ds.map(
                        lambda x: translate_batch(x, selected_cols, client),
                        batched=True,
                        batch_size=8,
                        desc=f"Translating {split_name}"
                    )
                    
                # Save in Parquet format with proper directory structure
                save_path = os.path.join(work_dir, "configs", cfg_name)
                os.makedirs(save_path, exist_ok=True)
                translated.to_parquet(f"{save_path}/{split_name}.parquet")

            except Exception as e:
                print(f"Error processing {config}: {str(e)}")
                continue

        # 8. Create necessary files
        print("\nCreating README.md and dataset.yaml...")
        create_readme_and_yaml(work_dir, selected_configs, datasets_dict, dataset_id)

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
