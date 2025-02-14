#!/usr/bin/env python3
import sys
import os
import tempfile
import shutil
from tqdm import tqdm
import time
import boto3
from datasets import load_dataset, get_dataset_config_names

def create_aws_translate_client(region_name=None):
    return boto3.client("translate", region_name=region_name)

def translate_text(text: str, client, source_lang="en", target_lang="es") -> str:
    if not text or not isinstance(text, str):
        return text
    try:
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
    for col in selected_cols:
        if col in batch:
            batch[col] = [translate_text(str(txt), client) for txt in batch[col]]
    return batch

def main():
    # 1. Get dataset info
    dataset_id = input("Enter Hugging Face dataset ID: ").strip()
    if not dataset_id:
        print("No dataset ID provided")
        return

    # 2. Get configs
    try:
        configs = get_dataset_config_names(dataset_id)
        selected_configs = configs if configs else [None]
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
                    # Translate with progress bar
                    print(f"\nTranslating {config if config else 'main'} - {split_name}")
                    translated = split_ds.map(
                        lambda x: translate_batch(x, selected_cols, client),
                        batched=True,
                        batch_size=8,
                        desc=f"Translating {split_name}"
                    )
                    
                    # Save split
                    save_path = os.path.join(work_dir, 
                                           config if config else 'main',
                                           split_name)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    translated.save_to_disk(save_path)

            except Exception as e:
                print(f"Error processing {config}: {str(e)}")
                continue

        # 8. Create dataset.yaml
        yaml_content = {
            "configs": [
                {
                    "config_name": cfg if cfg else "main",
                    "data_files": {
                        split: f"{cfg if cfg else 'main'}/{split}/*.arrow"
                        for split in (ds.keys() if isinstance(ds, dict) else ['train'])
                    }
                }
                for cfg in selected_configs
            ]
        }

        with open(os.path.join(work_dir, "dataset.yaml"), "w") as f:
            import yaml
            yaml.dump(yaml_content, f)

        # 9. Handle repository
        from huggingface_hub import Repository, login
        if token:
            login(token=token)

        # Clone the repository first
        repo = Repository(local_dir=repo_dir, clone_from=repo_name)
        
        # Copy all translated files to the repo directory
        for item in os.listdir(work_dir):
            src = os.path.join(work_dir, item)
            dst = os.path.join(repo_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        # Add, commit and push
        repo.git_add(".")
        repo.git_commit("Add translated dataset")
        repo.git_push()

    finally:
        # Clean up
        try:
            shutil.rmtree(work_dir)
            shutil.rmtree(repo_dir)
        except:
            pass

if __name__ == "__main__":
    main()