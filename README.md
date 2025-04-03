# TraductorDatasets: Dataset Translation Tool for Hugging Face

Este repositorio contiene scripts para traducir datasets de Hugging Face del inglés al español. Incluye dos herramientas:

1.  **main.py**: Descarga, traduce y publica datasets preservando su estructura completa.
2.  **calculate-translation-cost.py**: Analiza datasets y estima costos de traducción.

## Características Principales

-   Traducción de inglés a español usando **AWS Translate** o **OpenAI API** (e.g., GPT-4o mini).
-   Preservación completa de la estructura de datasets (configs, splits).
-   Traducción selectiva de columnas específicas.
-   Manejo especial para columnas de tipo conversación (`conversation` o `conversations`).
-   Caché global opcional para evitar retraducciones de contenido repetido (ej. system prompts idénticos).
-   Procesamiento en paralelo y traducción por lotes (batching) para eficiencia.
-   Progreso guardado automáticamente, permitiendo reanudar procesos interrumpidos.
-   Análisis de costos detallado para comparar AWS Translate vs OpenAI.

## Requisitos

-   **Python 3.8+**
-   **boto3**: Para AWS Translate.
-   **datasets**: Para manipulación de datasets de Hugging Face.
-   **huggingface_hub**: Para interacción con Hugging Face Hub.
-   **tqdm**: Para barras de progreso visuales.
-   **pyyaml**: Para leer/escribir archivos de configuración YAML.
-   **tiktoken**: Para análisis preciso de tokens (usado por `calculate-translation-cost.py`).
-   **openai**: Para traducción con OpenAI (requerido si se usa `--provider openai`).
-   **git** y **git-lfs**: Para preparar y publicar datasets en Hugging Face Hub.

Instala las dependencias con (se recomienda usar un entorno virtual):

```bash
pip install -r requirements.txt
# O para actualizar/reinstalar:
# pip install --upgrade -r requirements.txt
```

Asegúrate también de tener `git` y `git-lfs` instalados en tu sistema y accesibles desde la línea de comandos.

## Configuración

### Para AWS Translate:

Configura tus credenciales de AWS. La forma más común es mediante variables de entorno:

```bash
export AWS_ACCESS_KEY_ID=tu_access_key
export AWS_SECRET_ACCESS_KEY=tu_secret_key
export AWS_DEFAULT_REGION=tu_region_preferida # ej: us-east-1
```

Alternativamente, puedes usar el archivo `~/.aws/config` y `~/.aws/credentials`.

### Para OpenAI:

Configura tu API key de OpenAI, usualmente mediante una variable de entorno:

```bash
export OPENAI_API_KEY=tu_api_key_aqui
```

El script también te pedirá la clave si no la encuentra en el entorno y seleccionas `openai` como proveedor.

### Para Hugging Face:

Para poder subir el dataset traducido al Hub, necesitas autenticarte. Puedes hacerlo globalmente (recomendado):

```bash
huggingface-cli login
```

Opcionalmente, el script `main.py` te pedirá tu token de Hugging Face si no estás autenticado globalmente.

## Uso: Traducción de Datasets (`main.py`)

### Comandos Básicos

```bash
# Iniciar traducción en modo interactivo (te preguntará todo)
python main.py

# Modo de prueba rápido (procesa solo ~10 ejemplos por split)
python main.py --test

# Reanudar un proceso de traducción interrumpido (te pedirá el directorio de trabajo)
python main.py --resume
```

### Parámetros Avanzados (`main.py`)

Puedes combinar estos parámetros según necesites:

```bash
# Usar procesamiento paralelo con 10 hilos para acelerar
python main.py --parallel --workers 10

# Usar traducción por lotes para textos cortos (AWS) con tamaño de lote 5
python main.py --batch --batch-size 5

# Especificar el proveedor de traducción (aws o openai)
python main.py --provider openai

# Especificar el modelo de OpenAI a usar (si es diferente al default)
python main.py --provider openai --openai-model gpt-4o-mini

# Ajustar el tamaño del bloque (chunk) para guardar progreso (default: 100)
python main.py --chunk-size 200

# Cambiar el número de reintentos para llamadas API fallidas (default: 3)
python main.py --retries 5

# Habilitar caché global para contenido repetitivo (activado por defecto)
python main.py --global-cache
```

## Uso: Análisis de Costos (`calculate-translation-cost.py`)

Este script *no* realiza traducciones, solo analiza el dataset y estima los costos.

```bash
# Analizar costos completos para un dataset (puede tardar)
python calculate-translation-cost.py --dataset nombre/dataset

# Analizar solo ciertos configs (subsets) del dataset
python calculate-translation-cost.py --dataset nombre/dataset --configs subset1,subset2

# Analizar usando una muestra y extrapolar (RECOMENDADO para datasets grandes)
# Analiza los primeros 1000 ejemplos y estima el costo total
python calculate-translation-cost.py --dataset Org/DatasetName --sample 1000 --extrapolate

# Analizar solo columnas específicas que contienen texto a traducir
python calculate-translation-cost.py --dataset Org/DatasetName --columns texto_principal,descripcion --sample 500 --extrapolate
```

## Proceso de Traducción (`main.py`)

1.  **Selección del Dataset**: Proporciona el ID del dataset en Hugging Face (ej. `nombredelusuario/nombredeldataset`).
2.  **Selección de Configs**: Elige qué subconjuntos (configs) del dataset procesar (o todos).
3.  **Selección de Columnas**: Elige qué columnas contienen el texto a traducir.
4.  **Traducción Incremental**: El script procesa el dataset en bloques (`--chunk-size`), traduciendo el texto seleccionado usando el proveedor elegido (`--provider`). Guarda el progreso automáticamente después de cada bloque.
5.  **Combinación**: Una vez que todos los bloques de un split son traducidos, se combinan en un único archivo Parquet final para ese split.
6.  **Metadatos**: Se generan los archivos `README.md` y `dataset_infos.yaml` para el nuevo dataset traducido.
7.  **Publicación**: Intenta subir el dataset completo (archivos Parquet y metadatos) a un repositorio nuevo o existente en Hugging Face Hub. Si falla, guarda un backup localmente.

## Manejo de Casos Especiales

-   **System prompts**: Si se usa `--global-cache`, los prompts de sistema idénticos encontrados en la columna `system` se traducen una sola vez y se reutiliza la traducción.
-   **Conversaciones**: Para columnas llamadas `conversation` o `conversations`, el script intenta interpretar el contenido como una lista de diccionarios (ej. `[{'from': 'user', 'value': '...'}, ...]`) y traduce únicamente el texto dentro de los campos `value`.
-   **Textos Largos**: Los textos que exceden los límites de caracteres/tokens de la API de traducción (AWS o OpenAI) se dividen automáticamente en fragmentos más pequeños (intentando respetar frases/párrafos) antes de la traducción y se vuelven a unir después.
-   **Errores de API**: Se realizan reintentos automáticos (`--retries`) con un pequeño tiempo de espera si la API de traducción devuelve un error temporal. Si la traducción de un texto falla repetidamente, se mantiene el texto original.

## Estructura del Dataset Resultante en Hugging Face

El script organiza el dataset traducido siguiendo el estándar de Hugging Face Datasets.

```
nombreusuario/repo-traducido/
├── .gitattributes           # Asegura manejo LFS para Parquet
├── README.md                # Descripción autogenerada del dataset traducido
├── dataset_infos.yaml       # Configuración de carga (configs, splits, archivos)
└── data/
    ├── default/             # Config "default" (si no había nombres específicos)
    │   ├── train.parquet    # Archivo Parquet con el split "train" traducido (LFS)
    │   └── test.parquet     # Archivo Parquet con el split "test" traducido (LFS)
    └── config_nombre_original/ # Otros configs mantienen su nombre
        ├── validation.parquet # Archivo Parquet con split "validation" (LFS)
        └── ...
```

## Recuperación de Errores

-   **Interrupciones (Ctrl+C)**: El progreso se guarda al final del último bloque completado. Usa `python main.py --resume` y proporciona la ruta del directorio de trabajo para continuar.
-   **Errores de API**: Gestionados con reintentos; si un texto falla persistentemente, se mantiene el original.
-   **Fallos de Publicación en HF**: Si `git push` falla, el script guarda automáticamente los archivos traducidos y metadatos en un directorio local (`translated_dataset_backup` en la ubicación donde se ejecutó el script) para que puedas subirlos manualmente.

## Licencia

El código fuente de este proyecto está licenciado bajo la licencia MIT. Consulta el archivo [LICENSE](./LICENSE) para más detalles. La licencia de los *datasets traducidos* dependerá de la licencia del dataset original y de los términos de uso de los servicios de traducción (AWS/OpenAI).

## Colaboradores

Desarrollado por la Alianza Mexicana para la IA Soberana (AMIAS).