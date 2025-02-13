```markdown
# Script de traducción de datasets para Hugging Face

Este repositorio contiene un script de Python (`main.py`) que te ayudará a:

1. **Descargar** un dataset desde Hugging Face.
2. **Seleccionar** columnas de texto para traducir.
3. **Traducir** esas columnas de inglés a español (usando AWS Translate).
4. **Subir** el dataset traducido a Hugging Face **o** guardarlo localmente.

## Requisitos

- **Python 3.7+** (recomendado)
- **boto3** (`pip install boto3`)  
  > Requiere credenciales de AWS configuradas (variables de entorno o archivo `~/.aws/credentials`)
- **datasets** (`pip install datasets`)
- **huggingface_hub** (`pip install huggingface_hub`)

**Opcional**: Si vas a subir el dataset traducido a Hugging Face, necesitas:
- Una cuenta en [Hugging Face](https://huggingface.co).
- Un [token de acceso](https://huggingface.co/settings/tokens) (si no iniciaste sesión globalmente).

## Configuración

1. Clona este repositorio o descarga el archivo `main.py`.
2. Instala las dependencias de Python:
   ```bash
   pip install boto3 datasets huggingface_hub
   ```
3. Configura tus credenciales de AWS para que la traducción funcione:
   ```bash
   export AWS_ACCESS_KEY_ID=tu_access_key
   export AWS_SECRET_ACCESS_KEY=tu_secret_key
   export AWS_DEFAULT_REGION=tu_region_preferida
   ```
   O bien, usa el archivo `~/.aws/credentials`.

4. (Opcional) Inicia sesión en Hugging Face:
   ```bash
   huggingface-cli login
   ```
   O proporciona tu token cuando el script te lo solicite.

## Uso

Ejecuta el script:

```bash
python main.py
```

### Pasos del Asistente

1. **Identificador del dataset**  
   Por ejemplo: `edinburgh-dawg/mmlu-redux-2.0`.
2. **Selección de columnas**  
   Puedes escribir `"all"` para traducir todas las columnas, o una lista de índices (por ejemplo `0,2,3`).
3. **Traducción**  
   El script usa AWS Translate. Si no has configurado tus credenciales de AWS, dará error.
4. **Guardar/Subir**  
   - Opción 1: Sube el dataset resultante a un nuevo repositorio (o existente) en Hugging Face (por ejemplo `miusuario/mmlu-spanish`).
   - Opción 2: Guarda el dataset traducido localmente en formato Arrow.

## Licencia

Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo [LICENSE](./LICENSE) para más detalles.