#!/usr/bin/env python3
import sys
import tempfile

import boto3
from datasets import (
    load_dataset,
    get_dataset_config_names,
    DatasetDict,
)
from huggingface_hub import login

# ------------------------------------------------------------------------------
# Top-level translation-related functions (to avoid pickling warnings)
# ------------------------------------------------------------------------------

def create_aws_translate_client(region_name=None):
    """
    Creates and returns an AWS Translate client.
    If region_name is None, it will rely on environment variables or AWS config.
    """
    if region_name:
        return boto3.client("translate", region_name=region_name)
    else:
        return boto3.client("translate")

def translate_text(text: str, client, source_lang="en", target_lang="es") -> str:
    """
    Uses AWS Translate to translate text from source_lang to target_lang.
    """
    if not text.strip():
        return text
    try:
        response = client.translate_text(
            Text=text,
            SourceLanguageCode=source_lang,
            TargetLanguageCode=target_lang
        )
        return response["TranslatedText"]
    except Exception as e:
        print(f"Error al traducir texto '{text[:50]}...': {e}")
        return text  # fallback

def translate_batch(batch, selected_cols, client):
    """
    Batch mapper function to translate selected_cols in a batch of data.
    """
    for col in selected_cols:
        # Skip if the column is not in this batch
        if col not in batch:
            continue
        batch[col] = [translate_text(txt, client) for txt in batch[col]]
    return batch

# ------------------------------------------------------------------------------

def main():
    """
    CLI script to handle multi-config datasets automatically:
      1) Ask for dataset ID (e.g., 'edinburgh-dawg/mmlu-redux-2.0').
      2) List available configs (subsets), let user pick a subset or "all".
      3) Ask which columns to translate.
      4) Ask if user wants to push to HF Hub or save locally.
      5) Load + translate ALL selected configs.
      6) Combine them into one DatasetDict and do a single push/save at the end.
    """

    print("=== Asistente de traducción de datasets de Hugging Face ===")

    # Step 1. Dataset ID
    dataset_identifier = input(
        "Introduce el identificador del dataset en Hugging Face (e.g. 'edinburgh-dawg/mmlu-redux-2.0'): "
    ).strip()
    if not dataset_identifier:
        print("Error: no se proporcionó un identificador de dataset.")
        sys.exit(1)

    # Step 2. Determine available configs
    try:
        config_names = get_dataset_config_names(dataset_identifier)
    except Exception as e:
        print(f"Error al obtener la lista de configs para {dataset_identifier}: {e}")
        sys.exit(1)

    if not config_names:
        # Dataset has no named configs => single config
        print(f"\nEl dataset '{dataset_identifier}' no tiene configs nombradas; se cargará directamente.")
        selected_configs = [None]
    else:
        print("\nEste dataset contiene múltiples subsets (configs):")
        for idx, cnf in enumerate(config_names):
            print(f"  [{idx}] {cnf}")

        print("Escribe el índice de la config que deseas cargar,")
        print("o 'all' para todas, o varios índices separados por coma (ej. '0,2,5').")
        cfg_selection = input("Selección de config: ").strip()

        if cfg_selection.lower() == "all":
            selected_configs = config_names
        else:
            try:
                cfg_indices = [int(x.strip()) for x in cfg_selection.split(",")]
                selected_configs = [config_names[i] for i in cfg_indices]
            except:
                print("Error al parsear la selección de configs. Saliendo.")
                sys.exit(1)

    print(f"\nConfigs seleccionadas: {selected_configs}\n")

    # Step 3. Load the first config to see columns
    first_config = selected_configs[0]
    try:
        if first_config:
            temp_dataset = load_dataset(dataset_identifier, first_config)
        else:
            temp_dataset = load_dataset(dataset_identifier)
    except Exception as e:
        print(f"Error al cargar el dataset para la config '{first_config}': {e}")
        sys.exit(1)

    # Retrieve columns
    if isinstance(temp_dataset, DatasetDict):
        ref_split = list(temp_dataset.keys())[0]
        columns = temp_dataset[ref_split].column_names
    else:
        columns = temp_dataset.column_names

    # Ask which columns to translate
    print("Columnas disponibles en la primera config seleccionada:")
    for idx, col in enumerate(columns):
        print(f"  [{idx}] {col}")
    print(
        "\nEscribe una lista de índices de columnas separados por comas (p.ej. '0,2,3') para traducir,\n"
        "o 'all' para traducir todas, o vacío para no traducir ninguna."
    )
    col_selection = input("Selección de columnas: ").strip()

    if col_selection.lower() == "all":
        selected_cols = columns
    elif col_selection == "":
        selected_cols = []
    else:
        try:
            selected_indices = [int(x.strip()) for x in col_selection.split(",")]
            selected_cols = [columns[i] for i in selected_indices]
        except:
            print("Error al analizar los índices de columnas. No se traducirá ninguna columna.")
            selected_cols = []

    print(f"\nColumnas seleccionadas para traducir: {selected_cols if selected_cols else 'Ninguna'}\n")

    # Step 4. Ask if user wants to push to HF or save locally
    print("¿Cómo deseas almacenar el dataset traducido (de TODAS las configs seleccionadas)?")
    print("1) Subir a Hugging Face Hub (requiere login/token).")
    print("2) Guardar localmente (formato Arrow).")
    store_choice = input("Selecciona una opción (1 o 2): ").strip()

    # If user chooses to push, ask for repo name + HF token
    repo_name = None
    hf_token = None
    out_dir = None

    if store_choice == "1":
        repo_name = input(
            "Introduce el nombre del repositorio (ej. 'usuario/dataset-traducido'): "
        ).strip()
        if not repo_name:
            print("No se proporcionó nombre de repositorio. Saliendo.")
            sys.exit(1)
        print("\nSi lo necesitas, pega tu token de HF aquí (o deja vacío si ya iniciaste sesión): ")
        hf_token = input("Token: ").strip()
    else:
        # Save locally
        out_dir = input(
            "Introduce la ruta donde guardar el dataset traducido (p.ej. './dataset_traducido'): "
        ).strip()
        if not out_dir:
            print("No se proporcionó ruta; se usará un directorio temporal.")
            out_dir = tempfile.mkdtemp(prefix="hf-traducido-")
        print(f"Se guardará en: {out_dir}")

    # Step 5. Create AWS Translate client (make sure to have region set up)
    print("Configurando cliente de AWS Translate...")
    try:
        translate_client = create_aws_translate_client()
    except Exception as e:
        print(f"Error al crear el cliente de AWS Translate: {e}")
        sys.exit(1)

    # Step 6. Load + Translate ALL configs, unify them into one big DatasetDict
    # We'll store each config's splits under keys like "abstract_algebra_train", etc.
    print("\nTraduciendo todas las configs seleccionadas. Esto puede tardar...\n")

    translated_all_configs = DatasetDict()  # We'll accumulate all splits here

    for cfg in selected_configs:
        cfg_name = cfg if cfg else "default_config"
        print("=" * 80)
        print(f"Traduciendo config: {cfg_name}")

        # Load dataset for this config
        try:
            if cfg:
                ds = load_dataset(dataset_identifier, cfg)
            else:
                ds = load_dataset(dataset_identifier)
        except Exception as e:
            print(f"  Error al cargar la config '{cfg_name}': {e}")
            continue

        # Translate
        def mapper_fn(batch):
            return translate_batch(batch, selected_cols, translate_client)

        if isinstance(ds, DatasetDict):
            # Each split can be train/validation/test, etc.
            for split_name, split_ds in ds.items():
                new_split = split_ds.map(mapper_fn, batched=True)
                new_key = f"{cfg_name}_{split_name}"
                translated_all_configs[new_key] = new_split
        else:
            # Single dataset
            new_split = ds.map(mapper_fn, batched=True)
            # We'll store it under <cfg_name>_data
            new_key = f"{cfg_name}_data"
            translated_all_configs[new_key] = new_split

    print("\n=== Traducción finalizada. Preparando para almacenar... ===\n")

    # Step 7. Store/push the entire translated DatasetDict
    if store_choice == "1":
        # Push to HF
        if hf_token:
            login(token=hf_token)

        print(f"Subiendo todo al repositorio '{repo_name}'...")
        try:
            translated_all_configs.push_to_hub(repo_id=repo_name)
            print("¡Dataset traducido subido con éxito a Hugging Face!")
        except Exception as e:
            print(f"Error al subir a Hugging Face: {e}")
    else:
        # Save locally
        try:
            print(f"Guardando dataset en {out_dir} ...")
            translated_all_configs.save_to_disk(out_dir)
            print("¡Guardado completado!")
        except Exception as e:
            print(f"Error al guardar localmente: {e}")

    print("\n=== Proceso completado. ¡Gracias por usar el asistente de traducción! ===")

if __name__ == "__main__":
    main()
