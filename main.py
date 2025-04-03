#!/usr/bin/env python3
import os
import tempfile
import shutil
import json
import ast
import time
import warnings # To warn about issues
from tqdm import tqdm
import boto3
import subprocess
import argparse
import yaml
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from huggingface_hub import login
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import openai
import tiktoken # Needed for OpenAI token counting

# --- Constants ---
AWS_TRANSLATE_MAX_BYTES = 9900 # Slightly less than 10k limit for safety
OPENAI_DEFAULT_MODEL = "gpt-4o-mini" # User requested model
OPENAI_MAX_TOKENS = 127000 # Context window for gpt-4o-mini (input+output), use slightly less
TOKENIZER_NAME = "cl100k_base" # Tokenizer for gpt-4o-mini

# --- Helper: Tokenizer ---
try:
    encoding = tiktoken.get_encoding(TOKENIZER_NAME)
    tiktoken_available = True
except Exception:
    warnings.warn(f"Could not load tiktoken tokenizer '{TOKENIZER_NAME}'. "
                  f"OpenAI token counts will be rough estimates.")
    tiktoken_available = False

def count_tokens(text: str) -> int:
    """Counts tokens using tiktoken or estimates if unavailable."""
    if tiktoken_available:
        return len(encoding.encode(text))
    else:
        # Rough fallback: ~4 chars per token
        return len(text) // 4

# --- Client Creation ---

def create_aws_translate_client(region_name=None):
    """Creates and returns an AWS Translate client."""
    return boto3.client("translate", region_name=region_name)

def create_openai_client(api_key):
    """Creates and returns an OpenAI client."""
    return openai.OpenAI(api_key=api_key)

# --- Translation Functions ---

def translate_text(text: str, client: Any, source_lang="en", target_lang="es", max_retries=3, provider="aws", openai_model=OPENAI_DEFAULT_MODEL) -> str:
    """Translates a single text string using the selected provider."""
    if not text or not isinstance(text, str):
        return text # Return non-strings or empty strings as is

    if provider == "openai":
        return translate_text_openai(text, client, source_lang, target_lang, max_retries, openai_model)
    else: # default to AWS
        return translate_text_aws(text, client, source_lang, target_lang, max_retries)

def translate_text_aws(text: str, client: Any, source_lang="en", target_lang="es", max_retries=3) -> str:
    """Translates a single text string using AWS Translate, handling byte limits."""
    if not text:
        return text

    text_bytes = text.encode('utf-8')
    if len(text_bytes) <= AWS_TRANSLATE_MAX_BYTES:
        # Text is within limits, translate directly
        for attempt in range(max_retries):
            try:
                response = client.translate_text(
                    Text=text,
                    SourceLanguageCode=source_lang,
                    TargetLanguageCode=target_lang
                )
                return response["TranslatedText"]
            except Exception as e:
                if attempt == max_retries - 1:
                    warnings.warn(f"AWS Translate failed after {max_retries} attempts: {e}. Returning original text snippet: '{text[:100]}...'")
                    return text
                time.sleep(2 ** attempt) # Exponential backoff
        return text # Should not be reached, but as fallback
    else:
        # Text exceeds limit, split into chunks (simple paragraph/sentence split)
        warnings.warn(f"Text exceeds AWS byte limit ({len(text_bytes)} > {AWS_TRANSLATE_MAX_BYTES}). Splitting text.")
        translated_chunks = []
        current_chunk = ""
        # Simple split by paragraphs first, then sentences as fallback
        paragraphs = text.split('\n\n')
        sentences_to_process = []
        for p in paragraphs:
            if p.strip():
                 # Further split long paragraphs by sentences (basic split)
                 sentences_to_process.extend(s + '.' for s in p.split('. ') if s.strip())

        for sentence in sentences_to_process:
            sentence_bytes = sentence.encode('utf-8')
            current_chunk_bytes = current_chunk.encode('utf-8')

            if len(current_chunk_bytes) + len(sentence_bytes) <= AWS_TRANSLATE_MAX_BYTES:
                current_chunk += sentence
            else:
                # Translate the current chunk if it's not empty
                if current_chunk:
                    translated_chunks.append(translate_text_aws(current_chunk.strip(), client, source_lang, target_lang, max_retries))
                # Start a new chunk with the current sentence
                # If sentence itself is too long, it will be handled recursively (though inefficiently)
                current_chunk = sentence

        # Translate the last remaining chunk
        if current_chunk:
            translated_chunks.append(translate_text_aws(current_chunk.strip(), client, source_lang, target_lang, max_retries))

        # Join translated chunks. Decide joiner based on original structure maybe?
        # Simple space join for now. Could use '\n\n' if paragraphs were dominant.
        return ' '.join(translated_chunks)


def translate_text_openai(text: str, client: Any, source_lang="en", target_lang="es", max_retries=3, model=OPENAI_DEFAULT_MODEL) -> str:
    """Translates a single text string using OpenAI API, handling token limits."""
    if not text:
        return text

    # Estimate tokens needed for the prompt itself (system + user message)
    # This is approximate, actual usage depends on model specifics
    system_prompt = f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. Preserve formatting, markdown, and special characters. Only return the translated text without explanations."
    prompt_tokens = count_tokens(system_prompt) + count_tokens(text)

    if prompt_tokens <= OPENAI_MAX_TOKENS:
        # Text likely fits within limit, attempt direct translation
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.1 # Lower temperature for more deterministic translation
                )
                # Check if response is valid
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                     return response.choices[0].message.content.strip()
                else:
                     raise ValueError("OpenAI response was empty or invalid.")

            except Exception as e:
                warnings.warn(f"OpenAI API error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    warnings.warn(f"OpenAI Translate failed after {max_retries} attempts. Returning original text snippet: '{text[:100]}...'")
                    return text
                time.sleep(2 ** attempt) # Exponential backoff
        return text # Fallback

    else:
        # Text likely exceeds token limit, split into chunks
        warnings.warn(f"Text likely exceeds OpenAI token limit ({prompt_tokens} > {OPENAI_MAX_TOKENS}). Splitting text.")
        translated_chunks = []
        current_chunk = ""
        # Simple split logic (similar to AWS)
        paragraphs = text.split('\n\n')
        sentences_to_process = []
        for p in paragraphs:
            if p.strip():
                 sentences_to_process.extend(s + '.' for s in p.split('. ') if s.strip())

        for sentence in sentences_to_process:
            # Estimate tokens for potential chunk + sentence
            chunk_tokens = count_tokens(current_chunk)
            sentence_tokens = count_tokens(sentence)
            system_tokens = count_tokens(system_prompt) # Re-estimate system prompt for safety

            # Need to leave room for output tokens as well, hard to estimate precisely
            # Assume output is roughly same length as input for this check
            if system_tokens + chunk_tokens + sentence_tokens + (chunk_tokens + sentence_tokens) < OPENAI_MAX_TOKENS:
                current_chunk += sentence
            else:
                if current_chunk:
                    translated_chunks.append(translate_text_openai(current_chunk.strip(), client, source_lang, target_lang, max_retries, model))
                # If sentence itself is too long, it might fail recursively
                current_chunk = sentence

        if current_chunk:
            translated_chunks.append(translate_text_openai(current_chunk.strip(), client, source_lang, target_lang, max_retries, model))

        # Join translated chunks (simple space join)
        return ' '.join(translated_chunks)

# --- Batch Translation (AWS Optimized) ---

def batch_translate_texts(texts: List[str], client: Any, source_lang="en", target_lang="es", max_retries=3, batch_size=5, provider="aws", openai_model=OPENAI_DEFAULT_MODEL) -> List[str]:
    """
    Batch translate texts using selected translation provider.
    Currently, batching optimization is only implemented for AWS.
    """
    if not texts:
        return texts

    results = list(texts) # Create a mutable copy
    valid_texts_indices = [i for i, text in enumerate(texts) if text and isinstance(text, str)]

    if not valid_texts_indices:
        return results

    if provider == "openai":
        # OpenAI: Translate individually (batching not implemented here)
        for i in valid_texts_indices:
            results[i] = translate_text_openai(texts[i], client, source_lang, target_lang, max_retries, openai_model)
        return results

    # AWS: Attempt batching for short texts, otherwise individual calls
    for i in range(0, len(valid_texts_indices), batch_size):
        batch_indices = valid_texts_indices[i : i + batch_size]
        current_batch_texts = [texts[idx] for idx in batch_indices]

        # Simple check: if all texts are short and combined bytes are within limit
        total_bytes = sum(len(t.encode('utf-8')) for t in current_batch_texts)
        delimiter = " |||TRANS-SPLIT||| " # Slightly more unique delimiter
        delimiter_bytes = len(delimiter.encode('utf-8')) * (len(current_batch_texts) - 1)

        can_batch = True
        if not current_batch_texts:
            can_batch = False
        # Individual text length check (heuristic, 1000 chars)
        if any(len(t) > 1000 for t in current_batch_texts):
             can_batch = False
        # Combined byte check
        if total_bytes + delimiter_bytes > AWS_TRANSLATE_MAX_BYTES:
             can_batch = False

        if can_batch:
            combined_text = delimiter.join(current_batch_texts)
            translated_combined = translate_text_aws(combined_text, client, source_lang, target_lang, max_retries)

            # Split result, handle potential errors
            if translated_combined and delimiter in translated_combined:
                translated_texts = translated_combined.split(delimiter)
                if len(translated_texts) == len(current_batch_texts):
                    for j, idx in enumerate(batch_indices):
                        results[idx] = translated_texts[j].strip()
                    continue # Go to next batch
                else:
                    warnings.warn(f"Batch translation split mismatch (expected {len(current_batch_texts)}, got {len(translated_texts)}). Falling back to individual translation for this batch.")
            else:
                 warnings.warn(f"Batch translation failed or delimiter missing. Falling back to individual translation for this batch.")

        # Fallback: Translate individually if batching failed or wasn't attempted
        for idx in batch_indices:
             results[idx] = translate_text_aws(texts[idx], client, source_lang, target_lang, max_retries)

    return results


# --- Special Format Handling ---

def translate_python_literal_conversation(conv_str: str, client: Any, max_retries=3, provider="aws", openai_model=OPENAI_DEFAULT_MODEL) -> str:
    """Translates 'value' fields within a Python literal list-of-dicts conversation."""
    try:
        # Safely parse the string as a Python literal
        conversations = ast.literal_eval(conv_str)
        if not isinstance(conversations, list):
             raise ValueError("Parsed conversation is not a list.")

        translated = False
        for entry in conversations:
            # Check if entry is a dict and has 'value' field with content
            if isinstance(entry, dict) and 'value' in entry and entry['value'] and isinstance(entry['value'], str):
                original_value = entry['value']
                entry['value'] = translate_text(original_value, client, max_retries=max_retries, provider=provider, openai_model=openai_model)
                if entry['value'] != original_value:
                     translated = True

        # Return modified structure as Python literal string if translation occurred
        # Otherwise return original to avoid format changes if nothing translated
        return str(conversations) if translated else conv_str

    except (ValueError, SyntaxError, TypeError) as e:
        warnings.warn(f"Could not parse/process conversation literal: {e}. Returning original string: '{conv_str[:100]}...'")
        return conv_str
    except Exception as e:
         warnings.warn(f"Unexpected error processing conversation literal: {e}. Returning original: '{conv_str[:100]}...'")
         return conv_str


# --- Batch Processing Function for .map() ---

def translate_batch(
    batch: dict, # Batch format from datasets .map(batched=True)
    selected_cols: List[str],
    client: Any,
    max_retries: int,
    translation_cache: Optional[dict],
    use_parallel: bool,
    workers: int,
    use_batching: bool, # AWS specific batching
    batch_size: int, # AWS specific batching
    provider: str,
    openai_model: str
    ) -> dict:
    """
    Translates a batch of texts for selected columns using various strategies.
    Modifies the batch dictionary in-place (as expected by .map).
    """
    for col in selected_cols:
        if col in batch:
            texts_to_process = batch[col]
            results = [None] * len(texts_to_process) # Placeholder for results

            # --- Handle System Prompts (Caching) ---
            if col == "system" and translation_cache is not None and len(texts_to_process) > 0:
                # Check if all prompts in this batch are identical
                first_prompt = str(texts_to_process[0]) if texts_to_process[0] is not None else ""
                all_same = all((str(txt) if txt is not None else "") == first_prompt for txt in texts_to_process)

                if all_same and first_prompt:
                    cache_key = f"{provider}:{first_prompt}"
                    if cache_key in translation_cache:
                        translated = translation_cache[cache_key]
                        # print(f"Using CACHED translation for system prompt (hash: {hash(first_prompt) % 10000})") # Optional: verbose logging
                    else:
                        # print(f"Translating NEW system prompt (hash: {hash(first_prompt) % 10000})") # Optional: verbose logging
                        translated = translate_text(first_prompt, client, max_retries=max_retries, provider=provider, openai_model=openai_model)
                        translation_cache[cache_key] = translated
                    results = [translated] * len(texts_to_process)
                    batch[col] = results
                    continue # Skip other processing for this column

            # --- Handle Conversation Column ---
            # Handles 'conversation' (singular) and 'conversations' (plural)
            if col in ["conversation", "conversations"]:
                # Process each conversation string individually
                # Parallelization can be applied here
                def process_conv(conv_str):
                     return translate_python_literal_conversation(str(conv_str) if conv_str else "", client, max_retries, provider, openai_model)

                if use_parallel and len(texts_to_process) > 1:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        results = list(executor.map(process_conv, texts_to_process))
                else:
                    results = [process_conv(txt) for txt in texts_to_process]
                batch[col] = results
                continue # Skip other processing for this column

            # --- Handle Regular Text Columns ---
            # Filter out None values before translation
            valid_indices = [i for i, txt in enumerate(texts_to_process) if txt is not None and isinstance(txt, str)]
            valid_texts = [str(texts_to_process[i]) for i in valid_indices]
            translated_texts = [None] * len(valid_texts) # Results for valid texts

            if not valid_texts: # No valid text to translate in this column batch
                 batch[col] = texts_to_process # Return original batch
                 continue

            # Choose translation strategy
            if use_batching and provider == "aws" and len(valid_texts) > 1:
                # Use AWS-specific batching function
                translated_texts = batch_translate_texts(valid_texts, client, max_retries=max_retries, batch_size=batch_size, provider=provider)
            elif use_parallel and len(valid_texts) > 1:
                # Parallel individual translation
                with ThreadPoolExecutor(max_workers=workers) as executor:
                     future_to_idx = {
                          executor.submit(translate_text, text, client, "en", "es", max_retries, provider, openai_model): i
                          for i, text in enumerate(valid_texts)
                     }
                     for future in concurrent.futures.as_completed(future_to_idx):
                          idx = future_to_idx[future]
                          try:
                              translated_texts[idx] = future.result()
                          except Exception as e:
                              warnings.warn(f"Error translating text in parallel: {e}. Keeping original.")
                              translated_texts[idx] = valid_texts[idx] # Keep original on error
            else:
                # Sequential individual translation
                for i, text in enumerate(valid_texts):
                     translated_texts[i] = translate_text(text, client, "en", "es", max_retries, provider, openai_model)

            # Place translated results back into the original batch structure, keeping None values
            final_results = list(texts_to_process) # Start with original
            for i, original_idx in enumerate(valid_indices):
                 final_results[original_idx] = translated_texts[i]
            batch[col] = final_results

    return batch


# --- Progress Saving/Loading ---

def save_progress(work_dir, config, split, current_row, selected_cols, dataset_id=None, configs=None, provider="aws", openai_model=None):
    """Save current progress to a JSON file"""
    progress = {
        'dataset_id': dataset_id,
        'configs': configs,
        'config': config,
        'split': split,
        'current_row': current_row,
        'selected_cols': selected_cols,
        'provider': provider,
        'openai_model': openai_model, # Save model used
        'timestamp': time.time()
    }
    progress_file = os.path.join(work_dir, 'progress.json')
    try:
        os.makedirs(work_dir, exist_ok=True)
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
        # print(f"Progress saved to {progress_file}") # Optional: verbose logging
    except Exception as e:
        warnings.warn(f"Error saving progress to {progress_file}: {e}")

def load_progress(work_dir):
    """Load progress from JSON file"""
    progress_file = os.path.join(work_dir, 'progress.json')
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Error loading progress file {progress_file}: {e}")
            return None
    return None

# --- Repository Handling ---

def get_full_repo_url(repo_name):
    """Convert repo name to full Hugging Face URL"""
    if not '/' in repo_name:
         warnings.warn("Repository name should be in format 'username/repo_name'. Assuming username is part of the name.")
    return f"https://huggingface.co/datasets/{repo_name}"

def prepare_repository(repo_dir, repo_name, token=None):
    """Set up git repository with proper LFS handling"""
    os.makedirs(repo_dir, exist_ok=True)
    try:
        print("Setting up git LFS...")
        subprocess.run(['git', 'lfs', 'install', '--system', '--skip-repo'], check=True, capture_output=True)

        repo_url = get_full_repo_url(repo_name)
        print(f"Cloning or initializing repository: {repo_url}")

        # Try cloning, if fails, init new repo
        try:
            subprocess.run(['git', 'clone', repo_url, repo_dir], check=True, capture_output=True)
            print("Repository cloned.")
            # Ensure LFS is tracked even in existing repo
            subprocess.run(['git', 'lfs', 'track', "*.parquet"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(['git', 'add', '.gitattributes'], cwd=repo_dir, capture_output=True) # Add if changed
            subprocess.run(['git', 'commit', '-m', 'Ensure LFS tracking for parquet'], cwd=repo_dir, capture_output=True) # Commit if changed
        except subprocess.CalledProcessError:
            print("Clone failed (repo might not exist or empty). Initializing new repository.")
            subprocess.run(['git', 'init'], cwd=repo_dir, check=True, capture_output=True)
            # Setup LFS tracking
            subprocess.run(['git', 'lfs', 'track', "*.parquet"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(['git', 'add', '.gitattributes'], cwd=repo_dir, check=True, capture_output=True) # Add .gitattributes
            # Setup remote
            subprocess.run(['git', 'remote', 'add', 'origin', repo_url], cwd=repo_dir, check=True, capture_output=True)
            # Setup dummy user for commits if needed (will use global config if not set here)
            if token:
                subprocess.run(['git', 'config', 'user.name', 'HF Translator Bot'], cwd=repo_dir, check=True)
                subprocess.run(['git', 'config', 'user.email', '<>'], cwd=repo_dir, check=True)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up repository: {e}")
        print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
        return False
    except FileNotFoundError:
         print("Error: 'git' or 'git-lfs' command not found. Please install them and ensure they are in your PATH.")
         return False


def create_readme_and_yaml(work_dir, selected_configs, dataset_id, provider="aws", openai_model=None):
    """Creates README.md and dataset.yaml files with dataset information"""
    # Generate dataset_info.yaml content
    dataset_yaml_content = {"configs": []}
    has_default_config = False

    for cfg in selected_configs:
        cfg_name = cfg if cfg else "default"
        if cfg_name == "default":
            has_default_config = True

        config_dir_rel = os.path.join("data", cfg_name) # HF standard path is data/<config_name>
        config_dir_abs = os.path.join(work_dir, config_dir_rel)

        splits = []
        if os.path.exists(config_dir_abs):
            for file in os.listdir(config_dir_abs):
                if file.endswith(".parquet") and not file.startswith("chunk_"):
                    splits.append(file.replace(".parquet", ""))

        if splits:
            config_entry = {
                "config_name": cfg_name,
                "data_files": []
            }
            for split in sorted(splits): # Sort splits for consistency
                config_entry["data_files"].append({
                    "split": split,
                    "path": f"{config_dir_rel}/{split}.parquet"
                })
            dataset_yaml_content["configs"].append(config_entry)

    # Serialize YAML content
    try:
         # Use safe_dump and allow unicode
         dataset_yaml_str = yaml.safe_dump(dataset_yaml_content, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()
    except Exception as e:
         warnings.warn(f"Could not serialize dataset info to YAML: {e}")
         dataset_yaml_str = "# Error generating YAML"


    # README Content
    translation_method = "AWS Translate"
    if provider == "openai":
        translation_method = f"OpenAI API (Model: {openai_model or OPENAI_DEFAULT_MODEL})"

    readme_content = f"""---
# Basic card metadata
license: mit # License for the *translation script*, data license depends on original source
language:
- en # Original language
- es # Translated language
# Add other relevant tags
tags:
- translation
- parallel-corpus
---

# Spanish Translation of {dataset_id}

This repository contains a Spanish translation of the Hugging Face dataset [{dataset_id}](https://huggingface.co/datasets/{dataset_id}).

The translation was performed using the `traductor-datasets` tool.

**Translation Method**: {translation_method}

**Original Dataset Structure**: The original dataset structure (configs and splits) has been preserved.

**Note**: These translations were generated automatically and may contain errors or artifacts typical of machine translation. They are provided as-is. Please refer to the original dataset for authoritative content.

## Loading the Dataset

```python
from datasets import load_dataset

# Load a specific config and split (e.g., 'default' config, 'train' split)
ds = load_dataset("your_hf_username/{repo_name}", name="default", split="train")

# Load the default config (all splits)
ds_dict = load_dataset("your_hf_username/{repo_name}", name="default")
```

Replace `your_hf_username/{repo_name}` with the actual repository path, and adjust `name` and `split` as needed based on the available configs/splits.
"""

    # Write files
    try:
        with open(os.path.join(work_dir, "README.md"), "w", encoding='utf-8') as f:
            f.write(readme_content)
        # Save as dataset_infos.yaml which is sometimes preferred by HF tools
        with open(os.path.join(work_dir, "dataset_infos.yaml"), "w", encoding='utf-8') as f:
            f.write(dataset_yaml_str)
    except Exception as e:
         warnings.warn(f"Error writing metadata files: {e}")

# --- Main Execution Function ---

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate HuggingFace datasets with incremental saving and upload.')
    parser.add_argument('--test', action='store_true', help='Test mode: process only a small number of rows per split (e.g., 10)')
    parser.add_argument('--retries', type=int, default=3, help='Number of retries for translation API calls')
    parser.add_argument('--resume', action='store_true', help='Resume from a previous run (requires work directory path)')
    parser.add_argument('--chunk-size', type=int, default=100, help='Number of examples to process before saving progress')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing (ThreadPoolExecutor) for translation calls')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads for parallel processing')
    parser.add_argument('--batch', action='store_true', help='Use batch translation for AWS (groups multiple short texts)')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of texts to combine in AWS batch translation')
    # Defaulting global cache to True as it's generally beneficial for system prompts
    # Use --no-global-cache to disable it if needed (requires adding the action='store_false' argument)
    # parser.add_argument('--no-global-cache', dest='use_global_cache', action='store_false', help='Disable global caching of identical content')
    # parser.set_defaults(use_global_cache=True)
    # Simple flag for now:
    parser.add_argument('--global-cache', action='store_true', default=True, help='Enable global caching (default: True)')
    parser.add_argument('--provider', choices=['aws', 'openai'], default='aws', help='Translation provider to use (default: aws)')
    parser.add_argument('--openai-model', default=OPENAI_DEFAULT_MODEL, help=f'OpenAI model to use (default: {OPENAI_DEFAULT_MODEL})')

    args = parser.parse_args()
    use_global_cache = args.global_cache # Assign based on parsed args


    # --- Setup Work Directory ---
    work_dir = None
    if args.resume:
        saved_work_dir = input("Enter the path to the previous work directory: ").strip().strip('"\'')
        if os.path.isdir(saved_work_dir):
            work_dir = saved_work_dir
            print(f"Resuming using work directory: {work_dir}")
        else:
            print(f"Error: Work directory not found or not a directory: {saved_work_dir}")
            return
    else:
        try:
            work_dir = tempfile.mkdtemp(prefix='translation_work_')
            print(f"\nCreated temporary work directory: {work_dir}")
            print("If process is interrupted, use this path with --resume to continue.")
        except Exception as e:
            print(f"Error creating temporary directory: {e}. Using fallback.")
            fallback_dir = os.path.join(os.getcwd(), f"translation_work_{int(time.time())}")
            try:
                os.makedirs(fallback_dir, exist_ok=True)
                work_dir = fallback_dir
                print(f"Created fallback work directory: {work_dir}")
            except Exception as e_fallback:
                 print(f"Error creating fallback directory: {e_fallback}. Cannot continue.")
                 return

    # --- Load Progress or Get Setup Info ---
    progress = None
    provider = args.provider
    openai_model = args.openai_model

    if args.resume:
        progress = load_progress(work_dir)
        if progress:
            print(f"\nResuming translation for dataset: {progress.get('dataset_id', 'N/A')}")
            # Override provider/model from progress if available
            provider = progress.get('provider', args.provider)
            openai_model = progress.get('openai_model', args.openai_model)
            print(f"  Provider: {provider}")
            if provider == 'openai':
                print(f"  OpenAI Model: {openai_model}")
            print(f"  Config: {progress.get('config', 'N/A')}")
            print(f"  Split: {progress.get('split', 'N/A')}")
            print(f"  Resuming from example row: {progress.get('current_row', 0)}")
            print(f"  Columns being translated: {progress.get('selected_cols', 'N/A')}")
        else:
            print("Warning: Could not load progress file. Starting fresh setup.")
            args.resume = False # Force fresh setup

    # --- Initialize Translation Client ---
    client = None
    if provider == "openai":
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key and not args.resume: # Only ask if not resuming and not set
            openai_api_key = input("Enter OpenAI API key: ").strip()

        if openai_api_key:
            try:
                client = create_openai_client(openai_api_key)
                print(f"OpenAI client initialized (Model: {openai_model})")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}. Check API key and connectivity.")
                return # Cannot proceed without client
        else:
             print("Error: OpenAI provider selected, but no API key found (checked OPENAI_API_KEY env var).")
             return
    else: # AWS
        try:
            # Region can be configured via env vars (AWS_DEFAULT_REGION) or ~/.aws/config
            client = create_aws_translate_client()
            print("AWS Translate client initialized.")
        except Exception as e:
            print(f"Error initializing AWS Translate client: {e}. Check credentials and region configuration.")
            return

    # --- Get Dataset Info (if not resuming) ---
    if not args.resume:
        dataset_id = input("Enter Hugging Face dataset ID (e.g., username/dataset_name): ").strip()
        if not dataset_id: print("Dataset ID required."); return

        try:
            print("\nFetching dataset configurations...")
            available_configs = get_dataset_config_names(dataset_id)
            if not available_configs:
                print("No named configs found, assuming default config.")
                available_configs = [None] # Use None for default

            print("Available configs:")
            for i, cfg in enumerate(available_configs):
                print(f"  [{i}] {cfg if cfg else 'default'}")

            if args.test:
                print("\nTest mode enabled: processing only the first available config.")
                selected_configs_input = "0"
            else:
                selected_configs_input = input("Enter config numbers (comma-separated) or 'all': ").strip()

            selected_configs = []
            if selected_configs_input.lower() == 'all':
                selected_configs = available_configs
            else:
                try:
                    indices = [int(i.strip()) for i in selected_configs_input.split(',')]
                    selected_configs = [available_configs[i] for i in indices if 0 <= i < len(available_configs)]
                except ValueError:
                    print("Invalid config number format.")
                    return
            if not selected_configs: print("No valid configs selected."); return
            print(f"\nSelected configs: {[cfg if cfg else 'default' for cfg in selected_configs]}")


            # Get columns from the first selected config/split
            print("\nFetching column names...")
            try:
                # Load a tiny bit of the first config to get columns
                ds_sample = load_dataset(dataset_id, selected_configs[0], split='all[:1]') # Load 1 example
                if isinstance(ds_sample, dict): # Handle DatasetDict
                     first_split_key = next(iter(ds_sample))
                     available_columns = ds_sample[first_split_key].column_names
                else: # Handle Dataset
                     available_columns = ds_sample.column_names

            except Exception as e:
                 print(f"Could not load sample to determine columns: {e}. Please specify columns manually.")
                 available_columns = []


            print("\nAvailable columns:")
            if available_columns:
                for i, col in enumerate(available_columns):
                    print(f"  [{i}] {col}")
                cols_input = input("Enter column numbers to translate (comma-separated) or 'all': ").strip()

                selected_cols = []
                if cols_input.lower() == 'all':
                    selected_cols = available_columns
                else:
                    try:
                        indices = [int(i.strip()) for i in cols_input.split(',')]
                        selected_cols = [available_columns[i] for i in indices if 0 <= i < len(available_columns)]
                    except ValueError:
                        print("Invalid column number format.")
                        return
            else: # Could not auto-detect columns
                 cols_input_manual = input("Enter column names to translate (comma-separated): ").strip()
                 if not cols_input_manual: print("Column names required."); return
                 selected_cols = [c.strip() for c in cols_input_manual.split(',')]


            if not selected_cols: print("No valid columns selected."); return
            print(f"\nSelected columns for translation: {selected_cols}")

            # Save initial state before starting loops
            save_progress(work_dir, selected_configs[0], None, 0, selected_cols, dataset_id, selected_configs, provider, openai_model)

        except Exception as e:
            print(f"Error during dataset setup: {e}")
            return
    else: # Resuming
        dataset_id = progress.get('dataset_id')
        selected_configs = progress.get('configs')
        selected_cols = progress.get('selected_cols')
        if not all([dataset_id, selected_configs, selected_cols]):
             print("Error: Could not load necessary info from progress file.")
             return

    # --- Get Hugging Face Repo Info ---
    repo_name = input(f"Enter HF repo name to upload translated dataset (e.g., your_username/{dataset_id}-es): ").strip()
    if not repo_name or '/' not in repo_name:
        print("Invalid repository name format (should be 'username/repo_name').")
        return
    token = input("Enter HF token with write access (or press Enter if logged in via CLI): ").strip() or None


    # --- Setup Translation Environment ---
    repo_dir = os.path.join(work_dir, "hf_repo") # Place repo inside work_dir
    translation_cache = {} if use_global_cache else None
    chunk_size = 5 if args.test else args.chunk_size # Smaller chunk for testing
    print(f"\nUsing chunk size: {chunk_size} examples per save.")
    if args.parallel: print(f"Using parallel processing with {args.workers} workers.")
    if args.batch and provider == 'aws': print(f"Using AWS batching with batch size {args.batch_size}.")


    # --- Main Translation Loop ---
    print("\nStarting translation process...")
    try:
        start_config_idx = 0
        if args.resume and progress:
            try:
                start_config_idx = selected_configs.index(progress.get('config'))
            except ValueError:
                print(f"Warning: Config '{progress.get('config')}' from progress file not found in selected configs. Starting from the first selected config.")

        # Process each config
        for config_idx in range(start_config_idx, len(selected_configs)):
            config = selected_configs[config_idx]
            config_name_str = config if config else "default"
            print(f"\n--- Processing Config: {config_name_str} ---")

            try:
                 # Load dataset config (consider streaming for very large ones if memory becomes an issue)
                 ds = load_dataset(dataset_id, config, download_mode='reuse_cache_if_exists')
            except Exception as e:
                 print(f"Error loading dataset config '{config_name_str}': {e}. Skipping.")
                 continue

            # Handle splits within the config
            splits_to_process = {}
            if isinstance(ds, dict):
                 splits_to_process = ds
            else:
                 splits_to_process["train"] = ds # Assume 'train' if not a dict

            start_split_idx = 0
            split_names = list(splits_to_process.keys())
            if args.resume and progress and progress.get('config') == config:
                 try:
                      start_split_idx = split_names.index(progress.get('split'))
                 except ValueError:
                      print(f"Warning: Split '{progress.get('split')}' not found in config '{config_name_str}'. Processing all splits.")


            for split_idx in range(start_split_idx, len(split_names)):
                split_name = split_names[split_idx]
                split_ds = splits_to_process[split_name]
                print(f"\nTranslating Split: {split_name} ({len(split_ds)} examples)")

                # Apply test limit if needed
                if args.test:
                    split_ds = split_ds.select(range(min(10, len(split_ds))))
                    print(f"Test mode: processing only {len(split_ds)} examples.")

                total_examples = len(split_ds)
                start_row = 0
                if args.resume and progress and progress.get('config') == config and progress.get('split') == split_name:
                    start_row = progress.get('current_row', 0)
                    print(f"Resuming {split_name} from row {start_row}")

                # Define save path using HF standard (data/<config>/<split>.parquet)
                config_output_dir = os.path.join(work_dir, "data", config_name_str)
                os.makedirs(config_output_dir, exist_ok=True)
                final_split_path = os.path.join(config_output_dir, f"{split_name}.parquet")
                temp_chunk_dir = os.path.join(config_output_dir, f"chunks_{split_name}")
                os.makedirs(temp_chunk_dir, exist_ok=True)


                # Process in chunks
                for chunk_start in range(start_row, total_examples, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_examples)
                    print(f"\nProcessing rows {chunk_start} to {chunk_end-1}...")

                    # Get the chunk
                    chunk_ds = split_ds.select(range(chunk_start, chunk_end))

                    # Translate the chunk
                    try:
                         translated_chunk_ds = chunk_ds.map(
                              translate_batch,
                              batched=True,
                              batch_size=64, # How many rows .map processes together
                              fn_kwargs={
                                  "selected_cols": selected_cols,
                                  "client": client,
                                  "max_retries": args.retries,
                                  "translation_cache": translation_cache,
                                  "use_parallel": args.parallel,
                                  "workers": args.workers,
                                  "use_batching": args.batch,
                                  "batch_size": args.batch_size, # arg for AWS batching
                                  "provider": provider,
                                  "openai_model": openai_model
                              },
                              desc=f"Translating {split_name} ({chunk_start}-{chunk_end-1})"
                         )

                         # Save translated chunk to a temporary file
                         chunk_file = os.path.join(temp_chunk_dir, f"chunk_{chunk_start}_{chunk_end}.parquet")
                         translated_chunk_ds.to_parquet(chunk_file)
                         # print(f"Saved chunk: {chunk_file}") # Optional verbose logging

                         # Update and save progress
                         save_progress(work_dir, config, split_name, chunk_end, selected_cols, dataset_id, selected_configs, provider, openai_model)

                    except KeyboardInterrupt:
                         print("\nProcess interrupted by user.")
                         # Save progress at the start of the interrupted chunk
                         save_progress(work_dir, config, split_name, chunk_start, selected_cols, dataset_id, selected_configs, provider, openai_model)
                         print(f"Progress saved at row {chunk_start}. To resume, use the same work directory:\n{work_dir}")
                         return # Exit cleanly
                    except Exception as e_map:
                         print(f"\nError during translation map operation for chunk {chunk_start}-{chunk_end-1}: {e_map}")
                         print("Skipping this chunk, progress saved before chunk.")
                         # Save progress at the start of the failed chunk
                         save_progress(work_dir, config, split_name, chunk_start, selected_cols, dataset_id, selected_configs, provider, openai_model)
                         continue # Try next chunk

                # --- Combine Chunks for the Split ---
                print(f"\nCombining translated chunks for split: {split_name}...")
                try:
                    chunk_files = [os.path.join(temp_chunk_dir, f) for f in os.listdir(temp_chunk_dir) if f.startswith("chunk_") and f.endswith(".parquet")]
                    # Sort chunks numerically by start index
                    chunk_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

                    if not chunk_files:
                         print(f"Warning: No translated chunk files found for split {split_name}. Skipping combination.")
                         continue

                    # Load all chunks and concatenate
                    all_chunk_datasets = [load_dataset("parquet", data_files=cf, split='train') for cf in chunk_files] # Load each chunk as 'train' split
                    combined_ds = concatenate_datasets(all_chunk_datasets)

                    # Save the final combined split file
                    combined_ds.to_parquet(final_split_path)
                    print(f"Saved combined translated data to: {final_split_path}")

                    # Clean up temporary chunk files and directory
                    for cf in chunk_files:
                        os.remove(cf)
                    os.rmdir(temp_chunk_dir)
                    print("Cleaned up temporary chunk files.")

                except Exception as e_combine:
                    print(f"Error combining chunks for split {split_name}: {e_combine}")
                    print("Temporary chunk files preserved in:", temp_chunk_dir)

            # Reset progress split to None after finishing all splits in a config,
            # so resume starts next config correctly
            save_progress(work_dir, config, None, 0, selected_cols, dataset_id, selected_configs, provider, openai_model)


        # --- Final Steps after All Configs/Splits ---
        print("\nTranslation process completed for all selected configs.")

        # Create README and dataset_infos.yaml
        print("\nCreating metadata files (README.md, dataset_infos.yaml)...")
        create_readme_and_yaml(work_dir, selected_configs, dataset_id, provider, openai_model)

        # --- Upload to Hugging Face Hub ---
        print(f"\nPreparing repository '{repo_name}' for upload...")
        if token:
            try:
                login(token=token)
                print("Logged in to Hugging Face Hub using provided token.")
            except Exception as e_login:
                print(f"Warning: Failed to login using token: {e_login}. Upload might fail if not logged in via CLI.")

        if not prepare_repository(repo_dir, repo_name, token):
            print("Error: Failed to setup repository.")
            # Backup locally if repo setup failed
            backup_dir = os.path.join(os.getcwd(), f"translated_{dataset_id.replace('/','_')}_backup")
            try:
                 if os.path.exists(backup_dir): shutil.rmtree(backup_dir) # Remove old backup
                 shutil.copytree(work_dir, backup_dir, ignore=shutil.ignore_patterns('hf_repo')) # Exclude repo clone dir
                 print(f"\nTranslation results saved locally (excluding repo clone) to:\n{backup_dir}")
            except Exception as e_backup:
                 print(f"Error creating local backup: {e_backup}")
            return

        # Copy translated data ('data' dir) and metadata files into the repo dir
        print("Copying translated files to local repository clone...")
        try:
             data_src = os.path.join(work_dir, "data")
             data_dst = os.path.join(repo_dir, "data")
             if os.path.exists(data_dst): shutil.rmtree(data_dst)
             if os.path.exists(data_src): shutil.copytree(data_src, data_dst)

             shutil.copy2(os.path.join(work_dir, "README.md"), repo_dir)
             shutil.copy2(os.path.join(work_dir, "dataset_infos.yaml"), repo_dir)
        except Exception as e_copy:
             print(f"Error copying files to repository directory: {e_copy}")
             return

        # Push to Hugging Face
        print("\nAttempting to push translated dataset to Hugging Face Hub...")
        try:
            # Add all changes
            subprocess.run(['git', 'add', '.'], cwd=repo_dir, check=True, capture_output=True)
            # Check if there are changes to commit
            status_result = subprocess.run(['git', 'status', '--porcelain'], cwd=repo_dir, check=True, capture_output=True, text=True)
            if status_result.stdout:
                 commit_msg = f"Add translated dataset ({provider})"
                 subprocess.run(['git', 'commit', '-m', commit_msg], cwd=repo_dir, check=True, capture_output=True)
                 print("Committing changes...")
                 # Push
                 print(f"Pushing to {repo_name}...")
                 push_result = subprocess.run(['git', 'push', '-u', 'origin', 'main'], cwd=repo_dir, check=True, capture_output=True, text=True)
                 print("Push successful!")
                 print(f"Dataset available at: https://huggingface.co/datasets/{repo_name}")
            else:
                 print("No changes detected to commit or push.")

        except subprocess.CalledProcessError as e_push:
            print(f"\nError pushing to Hugging Face repository: {e_push}")
            print(f"Stderr: {e_push.stderr.decode() if e_push.stderr else 'N/A'}")
            # Backup locally on push failure
            backup_dir = os.path.join(os.getcwd(), f"translated_{dataset_id.replace('/','_')}_backup")
            try:
                 if os.path.exists(backup_dir): shutil.rmtree(backup_dir) # Remove old backup
                 shutil.copytree(work_dir, backup_dir, ignore=shutil.ignore_patterns('hf_repo'))
                 print(f"\nTranslation results saved locally (excluding repo clone) to:\n{backup_dir}")
            except Exception as e_backup:
                 print(f"Error creating local backup: {e_backup}")
        except Exception as e_git:
             print(f"An unexpected error occurred during git operations: {e_git}")


    except KeyboardInterrupt:
        # Should be caught within the loop now, but as a final fallback
        print("\nProcess interrupted.")
        print(f"Partial progress may be saved in the work directory:\n{work_dir}")
    except Exception as e_main:
        print(f"\nAn unexpected error occurred in the main process: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        # Optional: Offer to clean up work directory?
        # For now, keep it for resume/debugging purposes.
        print(f"\nWork directory (for potential resume or inspection): {work_dir}")


if __name__ == "__main__":
    main()