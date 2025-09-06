## App: Carga de texto, chunking y procesamiento con Gemini 2.5 Pro

Aplicación web en Flask para cargar un archivo de texto, dividirlo en secciones (chunking) y procesar cada sección con la API de Google Gemini (modelo configurable). Los resultados se muestran en la página.

### Requisitos
- Python 3.10+
- Cuenta y API Key de Google AI Studio

### Instalación
1. Clonar o copiar este directorio.
2. Crear y activar un entorno virtual:
   - Linux/macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```bash
     py -3 -m venv .venv
     .venv\\Scripts\\Activate.ps1
     ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Configurar variables de entorno:
   ```bash
   cp .env.example .env
   # Edita .env y pon tu GEMINI_API_KEY
   ```

### Ejecución
```bash
python run.py
# Abre http://localhost:8000
```

### Configuración
- `GEMINI_API_KEY`: requerido
- `GEMINI_MODEL`: por defecto `gemini-2.5-pro` (puedes usar otro compatible)
- `MAX_CONTENT_LENGTH_MB`: tamaño máximo de subida (MB), por defecto `2`
- `ALLOWED_EXTENSIONS`: extensiones permitidas para archivos, por defecto `txt`

### Uso
1. Accede a la app en tu navegador.
2. Carga un archivo `.txt`.
3. Elige el método de chunking (párrafos o por caracteres) y tamaños si aplica.
4. Envía el formulario. Verás las secciones originales y su procesamiento por Gemini.

### Seguridad y buenas prácticas
- Las subidas se validan por extensión, tamaño y se procesan en memoria.
- No se guardan archivos en disco.
- Manejo de errores y mensajes claros al usuario.

### Notas
- El procesamiento por defecto realiza un resumen breve por sección en español.
- Puedes ajustar el prompt en `app/gemini_client.py`.
# llopenai