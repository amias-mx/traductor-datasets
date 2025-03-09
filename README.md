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

1. Clona este repositorio o descarga el archivo `main.py`.

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

- Si falla la subida a Hugging Face, los archivos se guardan localmente como respaldo.
- Cada error de traducción se registra pero no detiene el proceso.
- El script incluye manejo de rate limiting para evitar límites de AWS.
- Los errores de traducción se reintentan según el parámetro `--retries`.
- Se mantiene un registro del progreso para poder reanudar en caso de interrupción.

## Licencia

Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo [LICENSE](./LICENSE) para más detalles.