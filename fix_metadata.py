import os
import json
import uuid
from datetime import datetime

# Configuración
PROJECT_ID = "eb1adf9e-f41f-4475-8bd6-e6a8115e8622"
PROJECT_TITLE = "Criticas_Fotogramas_v3"
FILE_NAME = "criticas_fotogramas.txt"
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "files", PROJECT_TITLE)
METADATA_FILE = os.path.join(PROJECT_DIR, "metadata.json")
APP_PROJECT_DIR = os.path.join(os.path.dirname(__file__), "backend", "projects")

# Comprobar que el archivo existe
file_path = os.path.join(PROJECT_DIR, FILE_NAME)
if not os.path.exists(file_path):
    print(f"El archivo {FILE_NAME} no existe en {PROJECT_DIR}")
    exit(1)

print(f"Archivo encontrado: {file_path}")

# Leer los metadatos actuales
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    print(f"Metadatos actuales cargados: {metadata}")
else:
    print(f"Archivo de metadatos no encontrado: {METADATA_FILE}")
    exit(1)

# Leer los metadatos de la aplicación si existen
app_metadata_file = os.path.join(APP_PROJECT_DIR, f"{PROJECT_ID}.json")
app_metadata = None
if os.path.exists(app_metadata_file):
    with open(app_metadata_file, 'r') as f:
        app_metadata = json.load(f)
    print(f"Metadatos de la aplicación cargados: {app_metadata}")
else:
    print(f"Archivo de metadatos de la aplicación no encontrado: {app_metadata_file}")
    # Creamos la carpeta si no existe
    os.makedirs(APP_PROJECT_DIR, exist_ok=True)
    # Usamos los metadatos actuales como base
    app_metadata = metadata.copy()
    print(f"Se usarán los metadatos actuales como base para los metadatos de la aplicación")

# Crear un registro para el archivo si no existe
file_exists = False
for file in metadata.get('files', []):
    if file.get('original_name') == FILE_NAME:
        file_exists = True
        print(f"El archivo ya está registrado en los metadatos: {file}")
        break

if not file_exists:
    # Generar un ID único para el archivo
    unique_id = str(uuid.uuid4())
    unique_filename = f"{unique_id}_{FILE_NAME}"
    
    # Obtener el tamaño del archivo
    file_size = os.path.getsize(file_path)
    
    # Crear un nuevo registro de archivo
    file_record = {
        "id": unique_id,
        "name": unique_filename,
        "original_name": FILE_NAME,
        "size": file_size,
        "uploaded_at": datetime.now().isoformat(),
        "url": f"/api/projects/{PROJECT_ID}/files/{unique_filename}"
    }
    
    # Añadir el archivo a los metadatos
    if 'files' not in metadata:
        metadata['files'] = []
    metadata['files'].append(file_record)
    
    # Actualizar la marca de tiempo de última modificación
    metadata['last_updated'] = datetime.now().isoformat()
    
    # Guardar los metadatos actualizados
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)
    print(f"Metadatos actualizados y guardados en {METADATA_FILE}")
    
    # Crear una copia del archivo con el nombre único en el directorio de la aplicación
    UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "files")
    app_project_dir = os.path.join(UPLOAD_DIR, PROJECT_ID)
    os.makedirs(app_project_dir, exist_ok=True)
    unique_file_path = os.path.join(app_project_dir, unique_filename)
    
    # Copiar el archivo
    with open(file_path, 'rb') as src_file:
        with open(unique_file_path, 'wb') as dst_file:
            dst_file.write(src_file.read())
    print(f"Archivo copiado a {unique_file_path}")
    
    # Actualizar también los metadatos de la aplicación
    if 'files' not in app_metadata:
        app_metadata['files'] = []
    app_metadata['files'].append(file_record)
    app_metadata['last_updated'] = datetime.now().isoformat()
    
    with open(app_metadata_file, 'w') as f:
        json.dump(app_metadata, f)
    print(f"Metadatos de la aplicación actualizados y guardados en {app_metadata_file}")
    
    print("¡Proceso completado con éxito!")
else:
    print("No es necesario actualizar los metadatos, el archivo ya está registrado.") 