# Script de traducción de datasets para Hugging Face

Este repositorio contiene un script de Python (`main.py`) que te ayudará a:

1. **Descargar** un dataset desde Hugging Face, preservando su estructura de subsets/configs.
2. **Seleccionar** columnas de texto para traducir.
3. **Traducir** esas columnas de inglés a español (usando AWS Translate).
4. **Subir** el dataset traducido a Hugging Face **o** guardarlo localmente.

## Requisitos

- **Python 3.12**
- **boto3** (`pip install boto3`)  
  > Requiere credenciales de AWS configuradas (variables de entorno o archivo `~/.aws/credentials`)
- **datasets** (`pip install datasets`)
- **huggingface_hub** (`pip install huggingface_hub`)
- **tqdm** (`pip install tqdm`)
- **pyyaml** (`pip install pyyaml`)
- **git** y **git-lfs** instalados en el sistema

**Opcional**: Si vas a subir el dataset traducido a Hugging Face, necesitas:
- Una cuenta en [Hugging Face](https://huggingface.co)
- Un [token de acceso](https://huggingface.co/settings/tokens) (si no iniciaste sesión globalmente)

## Configuración

1. Clona este repositorio o descarga el archivo `main.py`

2. Instala todas las dependencias de Python:
   ```bash
   pip install boto3 datasets huggingface_hub tqdm pyyaml
   ```

3. Instala git-lfs (necesario para subir datasets a Hugging Face):
   ```bash
   # En Ubuntu/Debian
   sudo apt-get install git-lfs

   # En macOS con Homebrew
   brew install git-lfs

   # En Windows con Chocolatey
   choco install git-lfs
   ```

4. Configura tus credenciales de AWS para la traducción:
   ```bash
   export AWS_ACCESS_KEY_ID=tu_access_key
   export AWS_SECRET_ACCESS_KEY=tu_secret_key
   export AWS_DEFAULT_REGION=tu_region_preferida
   ```
   O bien, usa el archivo `~/.aws/credentials`:
   ```ini
   [default]
   aws_access_key_id = tu_access_key
   aws_secret_access_key = tu_secret_key
   region = tu_region_preferida
   ```

5. (Opcional) Inicia sesión en Hugging Face:
   ```bash
   huggingface-cli login
   ```
   O proporciona tu token cuando el script te lo solicite.

## Uso

### Modos de ejecución
```bash
# Modo normal
python main.py

# Modo de prueba (2 subsets, 10 filas cada uno)
python main.py --test

# Especificar número de reintentos para AWS Translate
python main.py --retries 5

# Reanudar un proceso interrumpido
python main.py --resume
```

### Parámetros disponibles
- `--test`: Ejecuta en modo de prueba (2 subsets, 10 filas cada uno)
- `--retries N`: Número de reintentos para AWS Translate (por defecto: 3)
- `--resume`: Reanuda un proceso interrumpido previamente

### Funcionalidades nuevas

#### 1. Pausa y reanudación
- Puedes interrumpir el proceso en cualquier momento con Ctrl+C
- El progreso se guarda automáticamente
- Usa `--resume` para continuar desde donde lo dejaste
- Al reanudar, deberás proporcionar la ruta del directorio de trabajo anterior

#### 2. Reintentos de traducción
- Configura el número de reintentos para AWS Translate con `--retries`
- Por defecto, realizará 3 intentos antes de saltar un texto
- Útil para manejar errores temporales de red o límites de API

#### 3. Modo de prueba mejorado
- Procesa solo 2 subsets del dataset
- Limita cada subset a 10 filas
- Ideal para probar la configuración y el flujo de trabajo

### Pasos del asistente

1. **Identificador del dataset**  
   Introduce el identificador del dataset en Hugging Face.  
   Por ejemplo: `edinburgh-dawg/mmlu-redux-2.0`

2. **Selección de subsets (configs)**  
   - Si el dataset tiene múltiples subsets, podrás:
     - Seleccionar todos con `"all"`
     - Elegir específicos con índices (ej: `0,2,3`)
     - En modo prueba (`--test`), solo se procesarán dos

3. **Selección de columnas**  
   - Verás una lista de columnas disponibles
   - Puedes elegir:
     - Todas las columnas con `"all"`
     - Columnas específicas con índices (ej: `0,2,3`)
     - Ninguna dejando vacío

4. **Destino del dataset traducido**  
   Dos opciones:
   - **Opción 1**: Subir a Hugging Face
     - Necesitas proporcionar:
       - Nombre del repositorio (ej: `usuario/dataset-traducido`)
       - Token de HF (opcional si ya iniciaste sesión)
   - **Opción 2**: Guardar localmente
     - Especifica una ruta o se usará un directorio temporal

## Estructura del Dataset Resultante

El dataset traducido mantiene la estructura original:
```
repositorio/
├── README.md
├── dataset.yaml        # Configuración de subsets
└── subsets/
    ├── subset1/
    │   └── test/      # Archivos Arrow
    └── subset2/
        └── test/      # Archivos Arrow
```

## Manejo de errores

- Si falla la subida a Hugging Face, los archivos se guardan localmente como respaldo
- Cada error de traducción se registra pero no detiene el proceso
- El script incluye rate limiting para evitar límites de AWS
- Los errores de traducción se reintentan según el parámetro `--retries`
- Se mantiene un registro del progreso para poder reanudar en caso de interrupción

## Licencia

Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo [LICENSE](./LICENSE) para más detalles.