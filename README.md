# TraductorDatasets: Dataset Translation Tool for Hugging Face

Este repositorio contiene scripts para traducir datasets de Hugging Face del inglés al español. Incluye dos herramientas:

1. **main.py**: Descarga, traduce y publica datasets preservando su estructura completa
2. **calculate-translation-cost.py**: Analiza datasets y estima costos de traducción

## Características Principales

- Traducción de inglés a español usando **AWS Translate** o **OpenAI GPT**
- Preservación completa de la estructura de datasets (configs, splits)
- Traducción selectiva de columnas específicas
- Manejo especial para columnas de tipo conversación
- Caché inteligente para evitar retraducciones de contenido repetido
- Procesamiento en paralelo y traducción por lotes
- Progreso guardado automáticamente (reanudar procesos interrumpidos)
- Análisis de costos detallado para comparar AWS Translate vs OpenAI

## Requisitos

- **Python 3.8+**
- **boto3**: Para AWS Translate
- **datasets**: Para manipulación de datasets de Hugging Face
- **huggingface_hub**: Para interacción con Hugging Face
- **tqdm**: Para barras de progreso
- **pyyaml**: Para archivos de configuración
- **tiktoken**: Para análisis de tokens (herramienta de costos)
- **openai**: Para traducción con OpenAI (opcional)
- **git** y **git-lfs**: Para publicar datasets

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Configuración

### Para AWS Translate:

Configura tus credenciales de AWS mediante variables de entorno:

```bash
export AWS_ACCESS_KEY_ID=tu_access_key
export AWS_SECRET_ACCESS_KEY=tu_secret_key
export AWS_DEFAULT_REGION=tu_region_preferida  # ej: us-east-1
```

O en el archivo `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = tu_access_key
aws_secret_access_key = tu_secret_key
region = tu_region_preferida
```

### Para OpenAI:

Configura tu API key mediante:

```bash
export OPENAI_API_KEY=tu_api_key
```

O cuando el script te lo solicite.

### Para Hugging Face:

Inicia sesión globalmente:

```bash
huggingface-cli login
```

O proporciona tu token cuando el script te lo solicite.

## Uso: Traducción de Datasets

### Comandos Básicos

```bash
# Modo normal (interactivo)
python main.py

# Modo de prueba (procesa solo 10 ejemplos por split)
python main.py --test

# Reanudar un proceso interrumpido
python main.py --resume
```

### Parámetros Avanzados

```bash
# Traducción en paralelo (más rápido)
python main.py --parallel --workers 10

# Traducción por lotes (mejor para textos cortos)
python main.py --batch --batch-size 5

# Especificar proveedor de traducción
python main.py --provider openai

# Ajustar tamaño de bloque para guardar progreso
python main.py --chunk-size 200

# Reintentos para textos que fallan
python main.py --retries 5
```

## Uso: Análisis de Costos

```bash
# Analizar costos completos para un dataset
python calculate-translation-cost.py --dataset nombre/dataset

# Analizar solo ciertos subsets
python calculate-translation-cost.py --dataset nombre/dataset --configs subset1,subset2

# Analizar usando muestra y extrapolar
python calculate-translation-cost.py --dataset nombre/dataset --sample 100 --extrapolate

# Analizar solo ciertas columnas
python calculate-translation-cost.py --dataset nombre/dataset --columns texto,descripcion
```

## Proceso de Traducción

1. **Selección del dataset**: Proporciona el ID del dataset en Hugging Face
2. **Selección de configs**: Elige qué subsets procesar
3. **Selección de columnas**: Elige qué columnas traducir
4. **Traducción incremental**: Procesa en bloques con guardado automático
5. **Combinación**: Une los bloques traducidos en un dataset final
6. **Publicación**: Sube a Hugging Face o guarda localmente

## Manejo de Casos Especiales

- **System prompts**: Se identifican y cachean para evitar retraducciones
- **Conversaciones**: Extracción y traducción específica de campos relevantes
- **Textos largos**: División automática para cumplir límites de API
- **Errores**: Reintentos automáticos con espera incremental

## Estructura del Dataset Resultante

```
repo-traducido/
├── README.md               # Descripción del dataset traducido
├── dataset.yaml            # Configuración de splits y archivos
└── configs/
    ├── default/            # Config principal o "default"
    │   ├── train.parquet   # Split "train" traducido
    │   └── test.parquet    # Split "test" traducido
    └── config2/            # Otros configs manteniendo estructura
        ├── validation.parquet
        └── test.parquet
```

## Recuperación de Errores

- **Interrupciones**: Guarda progreso cada N ejemplos (--chunk-size)
- **Errores de API**: Reintentos automáticos (--retries)
- **Fallos de publicación**: Backup local automático

## Licencia

Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo [LICENSE](./LICENSE) para más detalles.

## Colaboradores

Desarrollado por la Alianza Mexicana para la IA Soberana (AMIAS)