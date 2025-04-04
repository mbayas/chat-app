# Chat App

Geppetto es una aplicación de chat desarrollada en React, Node.js, y Python.
Permite a los usuarios interactuar con diferentes proveedores de LLM (OpenAI, Google, Ollama) y realizar projectos donde el usuario sube sus archivos y puede realizar consultas sobre los mismos utilzando RAG.
Está orietado a personas que quieren interactuar con multiples modelos LLM y medir su comportanmiento en función de tiempo de respuesta, consumo de recursos y tokens utilizados.

## Requisitos

- Python 3.8+
- Node.js 14+
- npm o yarn

## Estructura del Proyecto

- `/frontend` - Aplicación React
- `/backend` - Servidor Python

Al instalar el proyecto, la estructura de directorios será la siguiente:

```
chat-app/
├── README.md
├── fix_metadata.py
├── cleanup_projects.html
├── faiss_explorer.py
├── backend/
│   ├── app.py
│   ├── config.py
│   └── llm_service.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── build/
│   └── node_modules/
└── db/

```

Al ejecutar el proyecto, la estructura de directorios será la siguiente:


```
chat-app/
├── README.md
├── .DS_Store
├── fix_metadata.py
├── cleanup_projects.html
├── faiss_explorer.py
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── llm_service.py
│   └── uploads/
├── frontend/
│   ├── src/
│   ├── public/
│   ├── build/
│   └── node_modules/
├── utils/
├── docs/
├── db/
│   └── [proyectos]/
│       ├── index.faiss
│       └── index.pkl
├── chats/
│   ├── [chat_id].json
│   └── [chat_id]_metadata.json
├── files/
└── uploads/
```


## Descripción de la Estructura

### Directorios Principales

- `/backend`: Servidor Python Flask
  - `app.py`: Aplicación principal
  - `config.py`: Configuraciones
  - `llm_service.py`: Servicio de LLM
  - `uploads/`: Archivos temporales y PDFs generados

- `/frontend`: Aplicación React
  - `src/`: Código fuente
  - `public/`: Archivos estáticos
  - `build/`: Compilación de producción
  - `node_modules/`: Dependencias de Node.js

- `/db`: Base de datos FAISS
  - Subdirectorios por proyecto con índices FAISS

- `/chats`: Historiales de chat
  - Archivos JSON de conversaciones
  - Archivos de metadatos

- `/docs`: Documentación del proyecto
- `/utils`: Utilidades y herramientas
- `/files` y `/uploads`: Almacenamiento de archivos

### Archivos en la Raíz

- `README.md`: Documentación principal
- `.gitignore`: Configuración de Git
- `fix_metadata.py`: Script de utilidad
- `faiss_explorer.py`: Herramienta para FAISS
- `cleanup_projects.html`: Utilidad de limpieza

## Requisitos

Explicación de los componentes principales:
1. Framework Web:
    - flask: El framework web principal
    - flask-cors: Para manejar CORS en las peticiones
2. Proveedores de LLM:
    - openai: Cliente oficial de OpenAI
    - google-generativeai: Cliente oficial de Google para Gemini
3. Procesamiento de Documentos y RAG:
    - langchain: Framework para trabajar con LLMs
    - langchain-community: Componentes comunitarios de LangChain
    - langchain-openai: Integraciones específicas de OpenAI
    - faiss-cpu: Para búsqueda de similitud vectorial
    - pypdf: Para procesar archivos PDF
4. Generación de PDFs y Markdown:
    - reportlab: Para generar PDFs
    - markdown2: Para convertir markdown a HTML
5. Utilidades:
    - python-magic: Para detección de tipos MIME
    - requests: Para hacer peticiones HTTP
    - typing-extensions: Para anotaciones de tipo



## Dependencias

### Paquetes Principales
```python
# Framework Web
flask==2.3.3
werkzeug==2.3.7
flask-cors==3.0.10
gunicorn==20.1.0

# Proveedores de LLM
openai==1.65.1
google-generativeai==0.3.2

# LangChain y RAG
langchain==0.3.21
langchain-community==0.3.20
langchain-openai==0.3.10
faiss-cpu

# Generación de PDFs y Procesamiento de Imágenes
reportlab==4.3.1
pillow==11.1.0
markdown2

# Utilidades
python-dotenv==1.0.0
python-magic==0.4.27
requests==2.31.0
rich==13.9.4
typing-extensions==4.12.2
```

## Ejecución

### Backend
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
cd backend
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```
## Configuración

1. Copia `.env.example` a `.env` en el directorio backend
2. Configura tus claves API para Google y OpenAI en el archivo .env

## Ejecución

### Backend
```bash
cd backend
python app.py
```

### Frontend
```bash
cd frontend
npm start
```
```