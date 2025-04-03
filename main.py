#!/usr/bin/env python3
import os
import tempfile
import shutil
import json
import ast
import time
import warnings # To warn about issues
import math # For chunking calculation
from tqdm import tqdm
import boto3
import subprocess
import argparse
import yaml
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets, get_dataset_split_names # Added get_dataset_split_names
from huggingface_hub import login
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import openai
import tiktoken # Needed for OpenAI token counting
from typing import Optional, List, Tuple, Any, Dict # Added Any, Dict

# --- Constants ---
AWS_TRANSLATE_MAX_BYTES = 9900 # Slightly less than 10k limit for safety
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
# Context window for gpt-4o-mini is 128k tokens, leave buffer
OPENAI_MAX_TOKENS = 127000
# Tokenizer for gpt-4o-mini and others
TOKENIZER_NAME = "cl100k_base"
# Practical character limit per API call for large text chunking
PRACTICAL_CHAR_LIMIT = 8000

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

    try:
        text_bytes = text.encode('utf-8')
    except Exception as e:
         warnings.warn(f"Could not encode text to UTF-8: {e}. Returning original snippet: '{text[:100]}...'")
         return text

    if len(text_bytes) <= AWS_TRANSLATE_MAX_BYTES:
        # Text is within limits, translate directly
        for attempt in range(max_retries):
            # <<< DEBUG PRINT ADDED >>>
            # print(f"          [DEBUG][AWS] Attempt {attempt+1}/{max_retries}: Sending request for text (len: {len(text)}): {text[:50]}...")
            try:
                response = client.translate_text(
                    Text=text,
                    SourceLanguageCode=source_lang,
                    TargetLanguageCode=target_lang
                )
                # <<< DEBUG PRINT ADDED >>>
                # print(f"          [DEBUG][AWS] Received response (Attempt {attempt+1}).")
                return response["TranslatedText"]
            except Exception as e:
                # <<< DEBUG PRINT ADDED >>>
                # print(f"          [DEBUG][AWS] ERROR during call (Attempt {attempt+1}): {e}")
                warnings.warn(f"AWS Translate API error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    warnings.warn(f"AWS Translate failed after {max_retries} attempts. Returning original text snippet: '{text[:100]}...'")
                    return text
                 # <<< DEBUG PRINT ADDED >>>
                # print(f"          [DEBUG][AWS] Retrying after delay...")
                time.sleep(1.5 ** attempt) # Exponential backoff slightly gentler for AWS
        return text # Should not be reached, but as fallback
    else:
        # Text exceeds limit, split into chunks (simple paragraph/sentence split)
        # <<< DEBUG PRINT ADDED >>>
        # print(f"          [DEBUG][AWS] Text exceeds byte limit ({len(text_bytes)} > {AWS_TRANSLATE_MAX_BYTES}). Splitting.")
        warnings.warn(f"Text exceeds AWS byte limit ({len(text_bytes)} > {AWS_TRANSLATE_MAX_BYTES}). Splitting text.")
        translated_chunks = []
        current_chunk = ""
        # Simple split by paragraphs first, then sentences as fallback
        # Prioritize paragraph breaks for potentially better context joining
        paragraphs = text.split('\n\n')
        sentences_to_process = []
        for p in paragraphs:
            if p.strip():
                 # Further split long paragraphs by sentences (basic split)
                 sentences_to_process.extend(s.strip() + '.' for s in p.split('. ') if s.strip()) # Add period back

        # <<< DEBUG PRINT ADDED >>>
        # print(f"          [DEBUG][AWS] Splitting into {len(sentences_to_process)} potential sentences/parts...")

        for sentence in sentences_to_process:
            sentence_bytes = sentence.encode('utf-8')
            current_chunk_bytes = current_chunk.encode('utf-8')

            if len(current_chunk_bytes) + len(sentence_bytes) + 1 <= AWS_TRANSLATE_MAX_BYTES: # +1 for potential space
                current_chunk += sentence + " " # Add space between sentences
            else:
                # Translate the current chunk if it's not empty
                if current_chunk:
                    # <<< DEBUG PRINT ADDED >>>
                    # print(f"            [DEBUG][AWS] Translating chunk (bytes: {len(current_chunk.encode('utf-8'))})...")
                    translated_chunks.append(translate_text_aws(current_chunk.strip(), client, source_lang, target_lang, max_retries))
                # Start a new chunk with the current sentence
                # If sentence itself is too long, it will be handled recursively (inefficiently)
                # <<< DEBUG PRINT ADDED >>>
                # print(f"            [DEBUG][AWS] Starting new chunk with sentence (bytes: {len(sentence_bytes)})...")
                current_chunk = sentence + " "

        # Translate the last remaining chunk
        if current_chunk:
            # <<< DEBUG PRINT ADDED >>>
            # print(f"            [DEBUG][AWS] Translating final chunk (bytes: {len(current_chunk.encode('utf-8'))})...")
            translated_chunks.append(translate_text_aws(current_chunk.strip(), client, source_lang, target_lang, max_retries))

        # Join translated chunks. Preserve paragraph breaks roughly if possible.
        # This is heuristic. A better approach would track original separators.
        # For now, join with space.
        # <<< DEBUG PRINT ADDED >>>
        # print(f"          [DEBUG][AWS] Finished splitting. Joining {len(translated_chunks)} translated chunks.")
        return ' '.join(translated_chunks).strip()


def translate_text_openai(text: str, client: Any, source_lang="en", target_lang="es", max_retries=3, model=OPENAI_DEFAULT_MODEL) -> str:
    """Translates a single text string using OpenAI API, handling token limits AND practical size limits."""
    if not text:
        return text

    # Use character length for simple chunking decision first
    if len(text) <= PRACTICAL_CHAR_LIMIT:
        # Text is within practical limits, try direct translation (with token check just in case)
        system_prompt = f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. Preserve formatting, markdown, and special characters. Only return the translated text without explanations."
        prompt_tokens = count_tokens(system_prompt) + count_tokens(text)

        if prompt_tokens <= OPENAI_MAX_TOKENS:
            # Fits within model limits too
            for attempt in range(max_retries):
                # <<< DEBUG PRINT ADDED >>>
                # print(f"          [DEBUG] Attempt {attempt+1}/{max_retries}: Sending request to OpenAI for text (len: {len(text)}): {text[:50]}...")
                try:
                    # Add a timeout (e.g., 180 seconds for potentially large but under-limit chunks)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text}
                        ],
                        temperature=0.1,
                        timeout=180.0
                    )
                    # <<< DEBUG PRINT ADDED >>>
                    # print(f"          [DEBUG] Received response from OpenAI (Attempt {attempt+1}).")
                    if response.choices and response.choices[0].message and response.choices[0].message.content:
                        translated_text = response.choices[0].message.content.strip()
                        # <<< DEBUG PRINT ADDED >>>
                        # print(f"          [DEBUG] Translation successful: {translated_text[:50]}...")
                        return translated_text
                    else:
                        # <<< DEBUG PRINT ADDED >>>
                        # print(f"          [DEBUG] ERROR: OpenAI response invalid or empty content (Attempt {attempt+1}).")
                        raise ValueError("OpenAI response was empty or invalid.")
                except Exception as e:
                    # <<< DEBUG PRINT ADDED >>>
                    # print(f"          [DEBUG] ERROR during OpenAI call (Attempt {attempt+1}): {e}")
                    warnings.warn(f"OpenAI API error on attempt {attempt + 1}/{max_retries}: {e}")
                    # Check for timeout explicitly if needed
                    # if isinstance(e, openai.Timeout):
                    #    print("          [DEBUG] Request timed out.")
                    if attempt == max_retries - 1:
                        warnings.warn(f"OpenAI Translate failed after {max_retries} attempts. Returning original text snippet: '{text[:100]}...'")
                        return text
                    # <<< DEBUG PRINT ADDED >>>
                    # print(f"          [DEBUG] Retrying after delay...")
                    time.sleep(2 ** attempt) # Exponential backoff
            # <<< DEBUG PRINT ADDED >>>
            # print(f"          [DEBUG] Returning original text after all retries failed.")
            return text # Fallback after retries
        else:
             # Input tokens exceed max limit even though chars might be low (rare)
             # <<< DEBUG PRINT ADDED >>>
             # print(f"          [DEBUG] ERROR: Text input tokens ({prompt_tokens}) exceed limit ({OPENAI_MAX_TOKENS}). Skipping.")
             warnings.warn(f"Text input tokens ({prompt_tokens}) exceed limit ({OPENAI_MAX_TOKENS}) despite character count. Skipping translation.")
             return text # Skip translation if token limit exceeded

    else:
        # Text exceeds practical char limit, split it first
        # <<< DEBUG PRINT ADDED >>>
        # print(f"          [DEBUG] Text exceeds practical limit ({len(text)} > {PRACTICAL_CHAR_LIMIT}). Splitting into chunks.")
        warnings.warn(f"Text exceeds practical limit ({len(text)} > {PRACTICAL_CHAR_LIMIT}). Splitting text for translation.")
        translated_chunks = []

        # Simple chunking by character count
        num_chunks = math.ceil(len(text) / PRACTICAL_CHAR_LIMIT)
        # <<< DEBUG PRINT ADDED >>>
        # print(f"          [DEBUG] Splitting into ~{num_chunks} chunks.")

        for i in range(num_chunks):
            start_index = i * PRACTICAL_CHAR_LIMIT
            end_index = start_index + PRACTICAL_CHAR_LIMIT
            text_chunk = text[start_index:end_index]

            if text_chunk: # Ensure chunk is not empty
                # <<< DEBUG PRINT ADDED >>>
                # print(f"            [DEBUG] Translating chunk {i+1}/{num_chunks} (len: {len(text_chunk)})...")
                # Recursively call translate_text_openai for the chunk
                translated_chunk = translate_text_openai(text_chunk, client, source_lang, target_lang, max_retries, model)
                translated_chunks.append(translated_chunk)

        # <<< DEBUG PRINT ADDED >>>
        # print(f"          [DEBUG] Finished splitting. Joining {len(translated_chunks)} translated chunks.")
        # Join translated chunks - use empty string join
        return "".join(translated_chunks)

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
        # Individual text length heuristic (chars)
        if any(len(t) > 1500 for t in current_batch_texts): # Reduced limit for batching
             can_batch = False
        # Combined byte check
        if total_bytes + delimiter_bytes > AWS_TRANSLATE_MAX_BYTES:
             can_batch = False

        if can_batch:
            # <<< DEBUG PRINT ADDED >>>
            # print(f"    [DEBUG][AWS Batch] Attempting to batch translate {len(current_batch_texts)} texts...")
            combined_text = delimiter.join(current_batch_texts)
            translated_combined = translate_text_aws(combined_text, client, source_lang, target_lang, max_retries)

            # Split result, handle potential errors
            if translated_combined and delimiter in translated_combined:
                translated_texts = translated_combined.split(delimiter)
                if len(translated_texts) == len(current_batch_texts):
                    for j, idx in enumerate(batch_indices):
                        results[idx] = translated_texts[j].strip()
                    # <<< DEBUG PRINT ADDED >>>
                    # print(f"    [DEBUG][AWS Batch] Batch successful.")
                    continue # Go to next batch
                else:
                    warnings.warn(f"AWS Batch translation split mismatch (expected {len(current_batch_texts)}, got {len(translated_texts)}). Falling back.")
            else:
                 warnings.warn(f"AWS Batch translation failed or delimiter missing. Falling back.")

        # Fallback: Translate individually if batching failed or wasn't attempted
        # <<< DEBUG PRINT ADDED >>>
        # print(f"    [DEBUG][AWS Batch] Using individual translation for {len(batch_indices)} texts in this batch.")
        for idx in batch_indices:
             results[idx] = translate_text_aws(texts[idx], client, source_lang, target_lang, max_retries)

    return results

# --- Special Format Handling ---

def translate_python_literal_conversation(conv_str: str, client: Any, max_retries=3, provider="aws", openai_model=OPENAI_DEFAULT_MODEL) -> str:
    """Translates 'value' fields within a Python literal list-of-dicts conversation."""
    # <<< DEBUG PRINT ADDED >>>
    # print(f"      [DEBUG] Attempting to parse conversation literal: {conv_str[:100]}...")
    if not isinstance(conv_str, str): # Handle potential non-string input if data is malformed
        warnings.warn(f"Non-string input to translate_python_literal_conversation: {type(conv_str)}. Returning as is.")
        return conv_str # Or str(conv_str) if preferred

    try:
        # Safely parse the string as a Python literal
        conversations = ast.literal_eval(conv_str)
        if not isinstance(conversations, list):
             raise ValueError("Parsed conversation is not a list.")
        # <<< DEBUG PRINT ADDED >>>
        # print(f"      [DEBUG] Parsed successfully. Num entries: {len(conversations)}")

        translated = False
        # <<< DEBUG PRINT ADDED >>>
        # print(f"      [DEBUG] Starting loop through conversation entries...")
        for i, entry in enumerate(conversations):
            # Check if entry is a dict and has 'value' field with content
            if isinstance(entry, dict) and 'value' in entry and entry['value'] and isinstance(entry['value'], str):
                original_value = entry['value']
                # <<< DEBUG PRINT ADDED >>>
                # print(f"        [DEBUG] Translating value for entry {i} (len: {len(original_value)}): {original_value[:50]}...")
                # Call the main translate_text function
                entry['value'] = translate_text(original_value, client, max_retries=max_retries, provider=provider, openai_model=openai_model)
                # <<< DEBUG PRINT ADDED >>>
                # print(f"        [DEBUG] Finished translating value for entry {i}.")
                if entry['value'] != original_value:
                     translated = True

        # <<< DEBUG PRINT ADDED >>>
        # print(f"      [DEBUG] Finished loop. Returning {'modified' if translated else 'original'} conversation string.")
        # Return modified structure as Python literal string if translation occurred
        # Otherwise return original to avoid format changes if nothing translated
        return str(conversations) if translated else conv_str

    except (ValueError, SyntaxError, TypeError) as e:
        # <<< DEBUG PRINT ADDED >>>
        # print(f"      [DEBUG] ERROR during parsing: {e}")
        warnings.warn(f"Could not parse/process conversation literal: {e}. Returning original string: '{conv_str[:100]}...'")
        return conv_str
    except Exception as e:
         # <<< DEBUG PRINT ADDED >>>
         # print(f"      [DEBUG] UNEXPECTED ERROR during processing: {e}")
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
    # <<< DEBUG PRINT ADDED >>>
    # print(f"  [DEBUG] translate_batch called for {len(batch[selected_cols[0]]) if selected_cols else 0} rows.")
    for col in selected_cols:
        if col in batch:
            # <<< DEBUG PRINT ADDED >>>
            # print(f"    Processing column: {col}")
            texts_to_process = batch[col]
            results = [None] * len(texts_to_process) # Placeholder for results

            # --- Handle System Prompts (Caching) ---
            if col == "system" and translation_cache is not None and len(texts_to_process) > 0:
                # <<< DEBUG PRINT ADDED >>>
                # print(f"      -> Checking for system prompt cache...")
                first_prompt = str(texts_to_process[0]) if texts_to_process[0] is not None else ""
                # Check if first prompt is valid and if all others are identical
                all_same = False
                if first_prompt:
                    all_same = all((str(txt) if txt is not None else "") == first_prompt for txt in texts_to_process)

                if all_same: # Handles case where first_prompt is "" correctly
                    cache_key = f"{provider}:{first_prompt}"
                    if cache_key in translation_cache:
                        translated = translation_cache[cache_key]
                    else:
                        # print(f"Translating NEW system prompt (hash: {hash(first_prompt) % 10000})") # Optional: verbose logging
                        translated = translate_text(first_prompt, client, max_retries=max_retries, provider=provider, openai_model=openai_model)
                        translation_cache[cache_key] = translated
                    results = [translated] * len(texts_to_process)
                    batch[col] = results
                    continue # Skip other processing for this column
                else:
                     # Fall through to regular processing if prompts in batch differ
                     # <<< DEBUG PRINT ADDED >>>
                     # print(f"      -> System prompts in batch differ, processing individually...")
                     pass


            # --- Handle Conversation Column ---
            # Handles 'conversation' (singular) and 'conversations' (plural)
            if col in ["conversation", "conversations"]:
                # <<< DEBUG PRINT ADDED >>>
                # print(f"      -> Processing as conversation column...")
                def process_conv(conv_str):
                     # Ensure input is string, handle None or other types gracefully
                     return translate_python_literal_conversation(str(conv_str) if conv_str is not None else "", client, max_retries, provider, openai_model)

                if use_parallel and len(texts_to_process) > 1:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        results = list(executor.map(process_conv, texts_to_process))
                else:
                    results = [process_conv(txt) for txt in texts_to_process]
                batch[col] = results
                continue # Skip other processing for this column

            # --- Handle Regular Text Columns ---
            # <<< DEBUG PRINT ADDED >>>
            # print(f"      -> Processing as regular text column...")
            # Filter out None values before translation, keep track of original indices
            valid_indices = [i for i, txt in enumerate(texts_to_process) if txt is not None and isinstance(txt, str)]
            valid_texts = [str(texts_to_process[i]) for i in valid_indices] # Ensure string type
            translated_texts = [None] * len(valid_texts) # Results for valid texts

            if not valid_texts: # No valid text to translate in this column batch
                 batch[col] = texts_to_process # Return original batch
                 continue

            # Choose translation strategy
            if use_batching and provider == "aws" and len(valid_texts) > 1:
                translated_texts = batch_translate_texts(valid_texts, client, source_lang="en", target_lang="es", max_retries=max_retries, batch_size=batch_size, provider=provider)
            elif use_parallel and len(valid_texts) > 1:
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
                              warnings.warn(f"Error translating text in parallel for column '{col}': {e}. Keeping original.")
                              translated_texts[idx] = valid_texts[idx] # Keep original on error
            else:
                # Sequential individual translation
                for i, text in enumerate(valid_texts):
                     translated_texts[i] = translate_text(text, client, "en", "es", max_retries, provider, openai_model)

            # Place translated results back into the original batch structure, preserving None/non-string values
            final_results = list(texts_to_process) # Start with original batch content
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
            warnings.warn(f"Error loading progress file {progress_file}: {e}. Starting fresh.")
            return None
    return None

# --- Repository Handling ---

def get_full_repo_url(repo_name):
    """Convert repo name to full Hugging Face URL"""
    if '/' not in repo_name:
         warnings.warn("Repository name should be in format 'username/repo_name'. Assuming username is part of the name.")
    # Ensure it points to datasets
    if not repo_name.startswith("datasets/"):
         if repo_name.count('/') == 1:
              return f"https://huggingface.co/datasets/{repo_name}"
         else: # Handle cases where user might include 'datasets/' already or have multiple slashes
              parts = [p for p in repo_name.split('/') if p]
              if len(parts) >= 2:
                   return f"https://huggingface.co/datasets/{'/'.join(parts[-2:])}"
              else: # Fallback if format is very unexpected
                   return f"https://huggingface.co/datasets/{repo_name}"
    else: # Already includes 'datasets/' prefix - use as is but construct URL
         return f"https://huggingface.co/{repo_name}"


def prepare_repository(repo_dir, repo_name, token=None):
    """Set up git repository with proper LFS handling"""
    os.makedirs(repo_dir, exist_ok=True)
    try:
        print("Setting up git LFS...")
        # Run globally first to ensure LFS filters are installed system-wide if possible
        try:
            subprocess.run(['git', 'lfs', 'install', '--system', '--skip-repo'], check=True, capture_output=True)
        except Exception as e_lfs_global:
             print(f"Info: Could not run 'git lfs install --system' (may require admin rights): {e_lfs_global}. Proceeding with local repo install.")
             subprocess.run(['git', 'lfs', 'install', '--local'], check=True, capture_output=True)


        repo_url = get_full_repo_url(repo_name)
        print(f"Cloning or initializing repository: {repo_name} from {repo_url}")

        # Try cloning, if fails, init new repo
        try:
            # Clone with specific depth or filter later if needed for large repos
            subprocess.run(['git', 'clone', repo_url, repo_dir], check=True, stderr=subprocess.PIPE) # Show stderr on error
            print("Repository cloned.")
            # Ensure LFS is tracked even in existing repo
            subprocess.run(['git', 'lfs', 'track', "data/**/*.parquet"], cwd=repo_dir, check=True, capture_output=True) # Track parquet files in data/ subdir
            subprocess.run(['git', 'add', '.gitattributes'], cwd=repo_dir, capture_output=True) # Add if changed/created
            # Check if commit needed
            status_result = subprocess.run(['git', 'status', '--porcelain', '.gitattributes'], cwd=repo_dir, check=True, capture_output=True, text=True)
            if status_result.stdout:
                 subprocess.run(['git', 'commit', '-m', 'Ensure LFS tracking for parquet files'], cwd=repo_dir, capture_output=True)
                 print("Committed .gitattributes LFS tracking.")

        except subprocess.CalledProcessError as e_clone:
            print(f"Clone failed (repo might not exist, be empty, or access denied): {e_clone.stderr.decode() if e_clone.stderr else e_clone}")
            print("Initializing new repository.")
            subprocess.run(['git', 'init'], cwd=repo_dir, check=True, capture_output=True)
            # Setup LFS tracking
            subprocess.run(['git', 'lfs', 'track', "data/**/*.parquet"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(['git', 'add', '.gitattributes'], cwd=repo_dir, check=True, capture_output=True) # Add .gitattributes
            # Commit .gitattributes *before* adding remote if git complains
            subprocess.run(['git', 'commit', '-m', 'Initial commit with LFS tracking'], cwd=repo_dir, check=True, capture_output=True)
            # Setup remote
            subprocess.run(['git', 'remote', 'add', 'origin', repo_url], cwd=repo_dir, check=True, capture_output=True)
            # Setup dummy user for commits if needed (will use global config if not set here)
            if token:
                subprocess.run(['git', 'config', 'user.name', 'HF Translator Bot'], cwd=repo_dir, check=True)
                subprocess.run(['git', 'config', 'user.email', '<>'], cwd=repo_dir, check=True)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up repository: {e}")
        print(f"Command: '{e.cmd}'")
        print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
        return False
    except FileNotFoundError:
         print("Error: 'git' or 'git-lfs' command not found. Please install them and ensure they are in your PATH.")
         return False
    except Exception as e_setup:
         print(f"An unexpected error occurred during repository setup: {e_setup}")
         return False


def create_readme_and_yaml(work_dir, selected_configs, dataset_id, provider, openai_model, repo_name): # Added repo_name
    """Creates README.md and dataset_infos.yaml files with dataset information"""
    # Generate dataset_info.yaml content
    dataset_yaml_content = {"configs": []}

    for cfg in selected_configs:
        cfg_name = cfg if cfg else "default"

        # Path structure expected by HF datasets: data/<config_name>/<split>.parquet
        config_data_dir_rel = os.path.join("data", cfg_name)
        config_data_dir_abs = os.path.join(work_dir, config_data_dir_rel)

        splits = []
        if os.path.exists(config_data_dir_abs):
            for file in os.listdir(config_data_dir_abs):
                # Look for final parquet files, ignore chunk files
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
                    # Use relative path for YAML
                    "path": f"{config_data_dir_rel}/{split}.parquet"
                })
            dataset_yaml_content["configs"].append(config_entry)

    # Serialize YAML content
    try:
         # Use safe_dump and allow unicode, ensure flow style is block
         dataset_yaml_str = yaml.safe_dump(dataset_yaml_content, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()
    except Exception as e:
         warnings.warn(f"Could not serialize dataset info to YAML: {e}")
         dataset_yaml_str = "# Error generating YAML content"


    # README Content
    translation_method = "AWS Translate"
    if provider == "openai":
        translation_method = f"OpenAI API (Model: {openai_model or OPENAI_DEFAULT_MODEL})"

    # Use the passed repo_name in the example usage
    readme_content = f"""---
# Basic card metadata generated by traductor-datasets script
# Consider adding more specific metadata based on the dataset's nature
license: unknown # License of translated data depends on original source and translation TOS
language:
- en # Original language assumed
- es # Translated language
tags:
- translation
- parallel-corpus
- machine-translated
---

# Spanish Translation of {dataset_id}

This repository contains a Spanish translation of the Hugging Face dataset [{dataset_id}](https://huggingface.co/datasets/{dataset_id}).

The translation was performed using the `traductor-datasets` tool.

**Translation Method**: {translation_method}

**Original Dataset Structure**: The original dataset structure (configs and splits) has been preserved in the `data/` directory.

**Note**: These translations were generated automatically and may contain errors or artifacts typical of machine translation. They are provided as-is without warranty. Please refer to the original dataset for authoritative content. Use of this translated data may be subject to the license of the original dataset and the terms of service of the translation provider ({provider}).

## Loading the Dataset

```python
from datasets import load_dataset

# Example: Load the 'default' config, 'train' split
# Replace '{repo_name}' with the actual repository path if different
# Replace 'default' and 'train' with desired config/split if needed
ds = load_dataset("{repo_name}", name="default", split="train")

# Example: Load all splits for the 'default' config
ds_dict = load_dataset("{repo_name}", name="default")

print(ds)
# print(ds_dict)
```

Replace `{repo_name}` with the actual repository path (e.g., `amias-mx/OpenThoughts-114k-spanish`), and adjust `name` and `split` as needed based on the available configs/splits defined in `dataset_infos.yaml`.
"""

    # Write files
    try:
        readme_path = os.path.join(work_dir, "README.md")
        yaml_path = os.path.join(work_dir, "dataset_infos.yaml") # Standard filename
        with open(readme_path, "w", encoding='utf-8') as f:
            f.write(readme_content)
        with open(yaml_path, "w", encoding='utf-8') as f:
            f.write(dataset_yaml_str)
        print(f"Generated {os.path.basename(readme_path)} and {os.path.basename(yaml_path)}")
    except Exception as e:
         warnings.warn(f"Error writing metadata files: {e}")

# --- Main Execution Function ---

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate HuggingFace datasets with incremental saving and upload.')
    # Execution Control
    parser.add_argument('--test', action='store_true', help='Test mode: process only ~10 rows per split')
    parser.add_argument('--resume', action='store_true', help='Resume from a previous run (requires work directory path)')
    # Dataset/Columns Specification (Optional, interactive otherwise)
    parser.add_argument('--dataset', help='HuggingFace dataset ID (e.g., username/dataset_name)')
    parser.add_argument('--columns', help='Comma-separated column names to translate (if skipping interactive selection)')
    parser.add_argument('--target-repo', help='Target HuggingFace repo name (e.g., username/repo-name-es)')
    # Translation Parameters
    parser.add_argument('--provider', choices=['aws', 'openai'], default='aws', help='Translation provider (default: aws)')
    parser.add_argument('--openai-model', default=OPENAI_DEFAULT_MODEL, help=f'OpenAI model (default: {OPENAI_DEFAULT_MODEL})')
    parser.add_argument('--retries', type=int, default=3, help='Max retries for translation API calls (default: 3)')
    # Performance/Process Parameters
    parser.add_argument('--chunk-size', type=int, default=100, help='Examples per chunk before saving progress (default: 100)')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing (ThreadPoolExecutor)')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads for parallel processing (default: 10)')
    parser.add_argument('--batch', action='store_true', help='Use batch translation for AWS (groups multiple short texts)')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of texts per AWS batch (default: 5)')
    parser.add_argument('--global-cache', action='store_true', default=True, help='Enable global caching for identical texts (default: True)')
    # Add option to disable cache if needed
    # parser.add_argument('--no-global-cache', dest='use_global_cache', action='store_false', help='Disable global caching')
    # parser.set_defaults(use_global_cache=True) # Keep default True behavior for now

    args = parser.parse_args()
    use_global_cache = args.global_cache

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
            if provider == 'openai': print(f"  OpenAI Model: {openai_model}")
            print(f"  Config: {progress.get('config', 'N/A')}")
            print(f"  Split: {progress.get('split', 'N/A')}")
            print(f"  Resuming from example row: {progress.get('current_row', 0)}")
            print(f"  Columns being translated: {progress.get('selected_cols', 'N/A')}")
            # Load other necessary info if resuming
            dataset_id = progress.get('dataset_id')
            selected_configs = progress.get('configs')
            selected_cols = progress.get('selected_cols')
            # Target repo is not saved in progress, will need to ask again or use arg
        else:
            print("Warning: Could not load progress file. Starting fresh setup.")
            args.resume = False # Force fresh setup

    # --- Initialize Translation Client ---
    client = None
    print(f"\nInitializing translation client for provider: {provider}")
    if provider == "openai":
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key and not args.resume: # Only ask if new run and not set
            openai_api_key = input("Enter OpenAI API key: ").strip()

        if openai_api_key:
            try:
                client = create_openai_client(openai_api_key)
                print(f"OpenAI client initialized (Model: {openai_model})")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}. Check API key/connectivity.")
                return
        else:
             print("Error: OpenAI provider selected, but no API key found or provided.")
             return
    else: # AWS
        try:
            client = create_aws_translate_client()
            print("AWS Translate client initialized.")
        except Exception as e:
            print(f"Error initializing AWS Translate client: {e}. Check credentials/region.")
            return

    # --- Get Dataset, Configs, Columns (if not resuming) ---
    if not args.resume:
        # 1. Get Dataset ID
        dataset_id = None
        if args.dataset:
            dataset_id = args.dataset.strip()
            print(f"\nUsing dataset ID from command line: {dataset_id}")
        else:
            dataset_id = input("\nEnter Hugging Face dataset ID (e.g., username/dataset_name): ").strip()
        if not dataset_id: print("Dataset ID required."); return

        # 2. Get Configs
        try:
            print("\nFetching dataset configurations...")
            available_configs = get_dataset_config_names(dataset_id, trust_remote_code=True)
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
                    print("Invalid config number format."); return
            if not selected_configs: print("No valid configs selected."); return
            print(f"\nSelected configs: {[cfg if cfg else 'default' for cfg in selected_configs]}")

        except Exception as e:
            print(f"Error fetching configs: {e}. Assuming default config.")
            selected_configs = [None]

        # 3. Get Columns
        print("\nAttempting to fetch column names...")
        available_columns = []
        selected_cols = []

        if args.columns:
           selected_cols = [c.strip() for c in args.columns.split(',') if c.strip()]
           print(f"Using columns from command line: {selected_cols}")
           if not selected_cols: print("Error: --columns specified but empty."); return
           # Skip auto-detection if columns are specified via args
        else:
            # --columns not provided, attempt auto-detection
            try:
                first_config = selected_configs[0]
                config_name_str = first_config if first_config else "default"
                print(f"Looking for splits in config '{config_name_str}'...")

                # Try getting splits first using config name string or None
                split_names = get_dataset_split_names(dataset_id, config_name_str if config_name_str != 'default' else None)

                if not split_names: # If that fails, try loading dataset object
                     print("Couldn't get splits directly, trying loading dataset object...")
                     ds_splits = load_dataset(dataset_id, first_config, download_mode='reuse_cache_if_exists', trust_remote_code=True)
                     if isinstance(ds_splits, dict):
                          split_names = list(ds_splits.keys())
                     else: # Maybe it's a single split Dataset object?
                          split_names = [ds_splits.split] if hasattr(ds_splits, 'split') and ds_splits.split else ['train'] # Guess 'train' as fallback


                if split_names:
                    first_split_name = split_names[0]
                    print(f"Found splits: {split_names}. Loading 1 example from '{first_split_name}'...")
                    ds_sample = load_dataset(dataset_id, first_config, split=f'{first_split_name}[:1]', trust_remote_code=True)
                    available_columns = ds_sample.column_names
                    if not available_columns: print(f"Warning: Loaded sample from '{first_split_name}' but found no columns.")
                else:
                    print(f"Warning: No splits found for config '{config_name_str}'. Cannot determine columns.")

            except Exception as e:
                 print(f"Could not automatically determine columns: {e}. Please specify columns manually.")
                 available_columns = [] # Ensure it's empty to trigger manual

            # Now prompt user based on auto-detection results (only if --columns wasn't used)
            if available_columns:
                print("\nAvailable columns:")
                for i, col in enumerate(available_columns): print(f"  [{i}] {col}")
                cols_input = input("Enter column numbers to translate (comma-separated) or 'all': ").strip()
                if cols_input.lower() == 'all': selected_cols = available_columns
                else:
                    try:
                        indices = [int(i.strip()) for i in cols_input.split(',')]
                        selected_cols = [available_columns[i] for i in indices if 0 <= i < len(available_columns)]
                    except ValueError: print("Invalid column number format."); return
            else: # Auto-detection failed AND --columns not provided
                 print("\nCould not automatically detect columns and --columns not specified.")
                 cols_input_manual = input("Enter column names to translate (comma-separated): ").strip()
                 if not cols_input_manual: print("Column names required."); return
                 selected_cols = [c.strip() for c in cols_input_manual.split(',')]

        # Final check and confirmation of selected columns
        if not selected_cols: print("No valid columns selected for translation."); return
        print(f"\nSelected columns for translation: {selected_cols}")

        # Save initial state before starting loops
        save_progress(work_dir, selected_configs[0], None, 0, selected_cols, dataset_id, selected_configs, provider, openai_model)


    # --- Get Target Repo Name ---
    repo_name = None
    if args.target_repo:
        repo_name = args.target_repo.strip()
        print(f"\nUsing target repository from command line: {repo_name}")
    else:
        repo_name_prompt = f"\nEnter HF repo name to upload translated dataset (e.g., your_username/{dataset_id.split('/')[-1]}-es): "
        repo_name = input(repo_name_prompt).strip()

    if not repo_name or '/' not in repo_name:
        print("Invalid repository name format (should be 'username/repo_name').")
        return

    # --- Get HF Token (only if not resuming, as it's not saved) ---
    token = None
    if not args.resume:
        token = input("Enter HF token with write access (or press Enter if logged in via CLI): ").strip() or None

    # --- Setup Translation Environment ---
    repo_dir = os.path.join(work_dir, "hf_repo") # Place repo clone inside work_dir
    translation_cache = {} if use_global_cache else None
    chunk_size = 5 if args.test else args.chunk_size # Smaller chunk for testing
    print(f"\nUsing chunk size: {chunk_size} examples per save.")
    if args.parallel: print(f"Using parallel processing with {args.workers} workers.")
    if args.batch and provider == 'aws': print(f"Using AWS batching with batch size {args.batch_size}.")


    # --- Main Translation Loop ---
    print("\nStarting translation process...")
    try:
        start_config_idx = 0
        start_split_idx = 0
        start_row = 0
        # Determine starting point from progress file if resuming
        if args.resume and progress:
            try:
                # Find starting config index
                start_config_idx = selected_configs.index(progress.get('config'))
                # Find starting split index (only if config matches)
                if progress.get('split'):
                     # Need to get splits for the target config to find index
                     config_to_resume = progress.get('config')
                     ds_splits_resume = load_dataset(dataset_id, config_to_resume, download_mode='reuse_cache_if_exists', trust_remote_code=True)
                     split_names_resume = list(ds_splits_resume.keys()) if isinstance(ds_splits_resume, dict) else ['train'] # Adjust if needed
                     try:
                         start_split_idx = split_names_resume.index(progress.get('split'))
                         # Get start row only if config and split match
                         start_row = progress.get('current_row', 0)
                     except ValueError:
                          print(f"Warning: Split '{progress.get('split')}' not found in resumed config '{config_to_resume}'. Starting config from first split.")
                else: # No specific split saved, start config from first split
                    start_split_idx = 0
                    start_row = 0

            except ValueError:
                print(f"Warning: Config '{progress.get('config')}' from progress file not found. Starting from first selected config.")
            except Exception as e_resume_load:
                 print(f"Warning: Error determining resume point from dataset structure: {e_resume_load}. Starting from beginning.")
                 start_config_idx = 0
                 start_split_idx = 0
                 start_row = 0


        # Process each config
        for config_idx in range(start_config_idx, len(selected_configs)):
            config = selected_configs[config_idx]
            config_name_str = config if config else "default"
            print(f"\n--- Processing Config: {config_name_str} ---")

            try:
                 ds = load_dataset(dataset_id, config, download_mode='reuse_cache_if_exists', trust_remote_code=True)
            except Exception as e:
                 print(f"Error loading dataset config '{config_name_str}': {e}. Skipping.")
                 continue

            # Handle splits within the config
            splits_to_process = {}
            split_names = []
            if isinstance(ds, dict):
                 splits_to_process = ds
                 split_names = list(ds.keys())
            else:
                 split_name_guess = ds.split if hasattr(ds, 'split') and ds.split else 'train'
                 splits_to_process[split_name_guess] = ds
                 split_names = [split_name_guess]


            current_start_split_idx = start_split_idx if config_idx == start_config_idx else 0

            for split_idx in range(current_start_split_idx, len(split_names)):
                split_name = split_names[split_idx]
                split_ds = splits_to_process[split_name]
                print(f"\nTranslating Split: {split_name} ({len(split_ds)} examples)")

                # Apply test limit if needed
                if args.test:
                    split_ds = split_ds.select(range(min(10, len(split_ds))))
                    print(f"Test mode: processing only {len(split_ds)} examples.")

                total_examples = len(split_ds)
                # Determine start row for this specific split
                current_start_row = start_row if config_idx == start_config_idx and split_idx == current_start_split_idx else 0

                if current_start_row >= total_examples:
                     print(f"Skipping split {split_name} as resume point ({current_start_row}) is beyond total examples ({total_examples}).")
                     continue

                # Define save path using HF standard (data/<config>/<split>.parquet)
                config_output_dir = os.path.join(work_dir, "data", config_name_str)
                os.makedirs(config_output_dir, exist_ok=True)
                final_split_path = os.path.join(config_output_dir, f"{split_name}.parquet")
                temp_chunk_dir = os.path.join(config_output_dir, f"chunks_{split_name}")
                os.makedirs(temp_chunk_dir, exist_ok=True)

                # Process in chunks
                for chunk_start in range(current_start_row, total_examples, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_examples)
                    print(f"\nProcessing rows {chunk_start} to {chunk_end-1}...")

                    # Get the chunk
                    try:
                         chunk_ds = split_ds.select(range(chunk_start, chunk_end))
                    except Exception as e_select:
                         print(f"Error selecting chunk {chunk_start}-{chunk_end-1}: {e_select}")
                         continue # Skip this chunk

                    # Translate the chunk
                    try:
                         # Clear cache hash warning check for each map call if possible?
                         # warnings.filterwarnings("ignore", category=UserWarning, message="Parameter 'fn_kwargs'") # Use cautiously

                         translated_chunk_ds = chunk_ds.map(
                              translate_batch,
                              batched=True,
                              batch_size=min(64, chunk_size), # Adjust map batch_size relative to chunk_size
                              fn_kwargs={
                                  "selected_cols": selected_cols,
                                  "client": client,
                                  "max_retries": args.retries,
                                  "translation_cache": translation_cache,
                                  "use_parallel": args.parallel,
                                  "workers": args.workers,
                                  "use_batching": args.batch,
                                  "batch_size": args.batch_size,
                                  "provider": provider,
                                  "openai_model": openai_model
                              },
                              desc=f"Translating {split_name} ({chunk_start}-{chunk_end-1})"
                         )

                         # Save translated chunk to a temporary file
                         chunk_file = os.path.join(temp_chunk_dir, f"chunk_{chunk_start}_{chunk_end}.parquet")
                         translated_chunk_ds.to_parquet(chunk_file)

                         # Update and save progress
                         save_progress(work_dir, config, split_name, chunk_end, selected_cols, dataset_id, selected_configs, provider, openai_model)

                    except KeyboardInterrupt:
                         print("\nProcess interrupted by user.")
                         save_progress(work_dir, config, split_name, chunk_start, selected_cols, dataset_id, selected_configs, provider, openai_model)
                         print(f"Progress saved at row {chunk_start}. To resume, use the same work directory:\n{work_dir}")
                         return # Exit cleanly
                    except Exception as e_map:
                         print(f"\nError during translation map operation for chunk {chunk_start}-{chunk_end-1}: {e_map}")
                         print("Skipping this chunk, progress saved before chunk.")
                         save_progress(work_dir, config, split_name, chunk_start, selected_cols, dataset_id, selected_configs, provider, openai_model)
                         continue # Try next chunk

                # --- Combine Chunks for the Split ---
                print(f"\nCombining translated chunks for split: {split_name}...")
                try:
                    chunk_files = [os.path.join(temp_chunk_dir, f) for f in os.listdir(temp_chunk_dir) if f.startswith("chunk_") and f.endswith(".parquet")]
                    if not chunk_files:
                         print(f"Warning: No translated chunk files found for split {split_name}. Skipping combination.")
                         continue

                    chunk_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1])) # Sort numerically

                    print(f"Found {len(chunk_files)} chunk files to combine.")
                    # Load all chunks and concatenate
                    all_chunk_datasets = [load_dataset("parquet", data_files=cf, split='train') for cf in chunk_files]
                    combined_ds = concatenate_datasets(all_chunk_datasets)

                    # Save the final combined split file
                    combined_ds.to_parquet(final_split_path)
                    print(f"Saved combined translated data to: {final_split_path}")

                    # Clean up temporary chunk files and directory
                    for cf in chunk_files: os.remove(cf)
                    os.rmdir(temp_chunk_dir)
                    print("Cleaned up temporary chunk files.")

                except Exception as e_combine:
                    print(f"Error combining chunks for split {split_name}: {e_combine}")
                    print(f"Temporary chunk files preserved in: {temp_chunk_dir}")

                # Reset start_row for next split within the same config/resume cycle
                start_row = 0

            # Reset start_split_idx for next config
            start_split_idx = 0
            # Save progress indicating config done (split=None)
            save_progress(work_dir, config, None, 0, selected_cols, dataset_id, selected_configs, provider, openai_model)


        # --- Final Steps after All Configs/Splits ---
        print("\nTranslation process completed for all selected configs.")

        # Create README and dataset_infos.yaml
        print("\nCreating metadata files (README.md, dataset_infos.yaml)...")
        # Pass repo_name obtained earlier
        create_readme_and_yaml(work_dir, selected_configs, dataset_id, provider, openai_model, repo_name)

        # --- Upload to Hugging Face Hub ---
        print(f"\nPreparing repository '{repo_name}' for upload...")
        # Use token obtained earlier if not resuming, otherwise prompt again if needed?
        # Current logic doesn't re-ask for token if resuming, relies on CLI login.
        if token and not args.resume: # Only auto-login if token provided on initial run
            try:
                login(token=token)
                print("Logged in to Hugging Face Hub using provided token.")
            except Exception as e_login:
                print(f"Warning: Failed to login using token: {e_login}. Upload might fail if not logged in via CLI.")

        if not prepare_repository(repo_dir, repo_name, token):
            print("Error: Failed to setup repository.")
            backup_dir = os.path.join(os.getcwd(), f"translated_{dataset_id.replace('/','_')}_backup")
            try:
                 if os.path.exists(backup_dir): shutil.rmtree(backup_dir)
                 shutil.copytree(work_dir, backup_dir, ignore=shutil.ignore_patterns('hf_repo'))
                 print(f"\nTranslation results saved locally (excluding repo clone) to:\n{backup_dir}")
            except Exception as e_backup: print(f"Error creating local backup: {e_backup}")
            return

        # Copy translated data ('data' dir) and metadata files into the repo dir
        print("Copying translated files to local repository clone...")
        try:
             data_src = os.path.join(work_dir, "data")
             data_dst = os.path.join(repo_dir, "data")
             if os.path.exists(data_dst): shutil.rmtree(data_dst)
             if os.path.exists(data_src): shutil.copytree(data_src, data_dst)

             readme_src = os.path.join(work_dir, "README.md")
             yaml_src = os.path.join(work_dir, "dataset_infos.yaml")
             if os.path.exists(readme_src): shutil.copy2(readme_src, repo_dir)
             if os.path.exists(yaml_src): shutil.copy2(yaml_src, repo_dir)
        except Exception as e_copy:
             print(f"Error copying files to repository directory: {e_copy}"); return

        # Push to Hugging Face
        print("\nAttempting to push translated dataset to Hugging Face Hub...")
        try:
            subprocess.run(['git', 'add', 'data/*', 'README.md', 'dataset_infos.yaml', '.gitattributes'], cwd=repo_dir, check=True)
            status_result = subprocess.run(['git', 'status', '--porcelain'], cwd=repo_dir, check=True, capture_output=True, text=True)
            if status_result.stdout:
                 commit_msg = f"Add/update translated dataset ({provider}, model: {openai_model if provider=='openai' else 'N/A'})"
                 subprocess.run(['git', 'commit', '-m', commit_msg], cwd=repo_dir, check=True)
                 print("Committing changes...")
                 # Determine current branch (usually main or master)
                 branch_result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=repo_dir, check=True, capture_output=True, text=True)
                 current_branch = branch_result.stdout.strip()
                 if not current_branch: current_branch = 'main' # Default assumption

                 print(f"Pushing to origin/{current_branch}...")
                 # Use --force-with-lease for potentially safer force pushes if needed, but start without
                 push_result = subprocess.run(['git', 'push', '-u', 'origin', current_branch], cwd=repo_dir, check=True, stderr=subprocess.PIPE, text=True)
                 print("Push successful!")
                 print(f"Dataset available at: https://huggingface.co/datasets/{repo_name}")
            else:
                 print("No changes detected to commit or push.")

        except subprocess.CalledProcessError as e_push:
            print(f"\nError pushing to Hugging Face repository: {e_push}")
            print(f"Stderr: {e_push.stderr if e_push.stderr else 'N/A'}")
            backup_dir = os.path.join(os.getcwd(), f"translated_{dataset_id.replace('/','_')}_backup")
            try:
                 if os.path.exists(backup_dir): shutil.rmtree(backup_dir)
                 shutil.copytree(work_dir, backup_dir, ignore=shutil.ignore_patterns('hf_repo'))
                 print(f"\nTranslation results saved locally (excluding repo clone) to:\n{backup_dir}")
            except Exception as e_backup: print(f"Error creating local backup: {e_backup}")
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