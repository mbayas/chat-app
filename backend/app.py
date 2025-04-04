from flask import Flask, request, jsonify, send_from_directory, send_file, Response, stream_with_context
from flask_cors import CORS
import os
import json
from datetime import datetime
import uuid
from llm_service import LLMService
from config import Config
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import markdown2
import re
import shutil

# Import additional libraries needed for RAG processing - Updated for newer LangChain versions
# Usar langchain_community en lugar de langchain para evitar advertencias de deprecación
try:
    # Primero intentar las nuevas versiones de las librerías
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS  # Cambiado de Chroma a FAISS
    print("Using latest LangChain libraries with FAISS vectorstore")
except ImportError:
    # Fallback a las versiones anteriores si no están disponibles las nuevas
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS  # Cambiado de Chroma a FAISS
    print("Using older LangChain libraries - consider upgrading with pip install langchain-openai langchain-community")

app = Flask(__name__)
CORS(app)

# Initialize configuration
config = Config()

# Initialize LLM service
llm_service = LLMService(
    provider="openai",  # Specify the provider
    model=config.MODEL,
    api_key=config.OPENAI_API_KEY
)

# Update settings after initialization
llm_service.update_settings(
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKENS
)

# Function to estimate token count (very rough estimation)
def estimate_tokens(text):
    """
    Estimates the number of tokens in a text. This is a rough estimation.
    Args:
        text (str): The text to estimate tokens for
    Returns:
        int: Estimated number of tokens
    """
    if not text:
        return 0
    
    # Approximate tokenization rules (GPT models):
    # 1. ~4 chars per token on average for English text
    # 2. But code, spaces, and non-English characters have different ratios
    
    # Count different content types
    code_blocks = re.findall(r'```[\s\S]*?```', text, re.DOTALL)
    code_blocks_text = ''.join(code_blocks)
    non_code_text = re.sub(r'```[\s\S]*?```', '', text, flags=re.DOTALL)
    
    # Different token ratios for different content
    code_tokens = len(code_blocks_text) // 3  # Code is slightly more efficient in tokens
    text_tokens = len(non_code_text) // 4     # Normal text is ~4 chars per token
    
    return code_tokens + text_tokens

# Utility function to ensure RAG directory structure
def initialize_rag_directory():
    """
    Initialize the RAG directory structure for FAISS.
    This function creates the base DB directory if it doesn't exist.
    """
    base_db_path = os.path.join(os.path.dirname(__file__), '..', 'db')
    try:
        if not os.path.exists(base_db_path):
            print(f"Creating main RAG directory at: {base_db_path}")
            os.makedirs(base_db_path, exist_ok=True)
        
        # Ensure the directory is writable
        test_file_path = os.path.join(base_db_path, 'test_write.tmp')
        try:
            with open(test_file_path, 'w') as f:
                f.write('test')
            os.remove(test_file_path)
            print("RAG directory is properly configured and writable")
        except Exception as e:
            print(f"WARNING: RAG directory exists but might not be writable: {str(e)}")
    except Exception as e:
        print(f"ERROR initializing RAG directory: {str(e)}")
        print("RAG functionality may not work correctly!")

# Run the initialization on startup
initialize_rag_directory()

# Directory to store chat histories
CHAT_DIR = os.path.join(os.path.dirname(__file__), 'chats')
os.makedirs(CHAT_DIR, exist_ok=True)

# Directory to store uploaded documents
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directory to store projects
PROJECT_DIR = os.path.join(os.path.dirname(__file__), 'projects')
os.makedirs(PROJECT_DIR, exist_ok=True)

# In-memory storage for chats, projects and settings
chats = {}
projects = {}
settings = {
    "provider": "openai",
    "model": config.MODEL,
    "temperature": config.TEMPERATURE,
    "max_tokens": config.MAX_TOKENS,
    "ollama_base_url": "http://localhost:11434",
    "maintain_context": True,  # Por defecto, mantener el contexto
    "rag_num_chunks": 10  # Number of chunks to retrieve for RAG top_k
}

# Agregar un registro global para las solicitudes activas para permitir su cancelación
streaming_active_requests = {}

# Función para guardar los metadatos de un chat en un archivo
def save_chat_metadata(chat_id):
    """Guarda los metadatos del chat en un archivo específico para persistencia."""
    if chat_id not in chats:
        return False
    
    # Asegurar que la estructura del chat es correcta
    ensure_chat_structure(chat_id)
    
    # Crear estructura de metadatos
    metadata = {
        "id": chats[chat_id]["id"],
        "title": chats[chat_id]["title"],
        "created_at": chats[chat_id]["created_at"],
        "last_message": chats[chat_id]["last_message"]
    }
    
    # Guardar metadatos en archivo
    metadata_file = os.path.join(CHAT_DIR, f"{chat_id}_metadata.json")
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        return True
    except Exception as e:
        print(f"Error guardando metadatos del chat {chat_id}: {str(e)}")
        return False

# Modificar la función load_chats_from_files para cargar también desde metadatos
def load_chats_from_files():
    """Carga todos los chats desde los archivos JSON almacenados en CHAT_DIR."""
    print("Cargando chats desde archivos...")
    try:
        # Obtener todos los archivos de chat (mensajes y metadatos)
        all_files = os.listdir(CHAT_DIR)
        chat_files = [f for f in all_files if f.endswith('.json') and not f.endswith('_metadata.json')]
        metadata_files = [f for f in all_files if f.endswith('_metadata.json')]
        
        # Diccionario para almacenar temporalmente los chats cargados
        loaded_chats = {}
        
        # Primero intentamos cargar desde archivos de metadatos
        for filename in metadata_files:
            chat_id = filename.replace('_metadata.json', '')
            metadata_file_path = os.path.join(CHAT_DIR, filename)
            chat_file_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
            
            try:
                # Cargar los metadatos
                with open(metadata_file_path, 'r') as f:
                    metadata = json.load(f)
                
                # Cargar mensajes si el archivo existe
                messages = []
                if os.path.exists(chat_file_path):
                    with open(chat_file_path, 'r') as f:
                        messages = json.load(f)
                
                # Crear estructura del chat combinando metadatos y mensajes
                loaded_chats[chat_id] = {
                    "id": metadata.get("id", chat_id),
                    "title": metadata.get("title", chat_id),
                    "created_at": metadata.get("created_at", datetime.now().isoformat()),
                    "last_message": metadata.get("last_message", datetime.now().isoformat()),
                    "messages": messages
                }
                
                print(f"Cargado chat {chat_id} desde metadatos con {len(messages)} mensajes")
                
            except Exception as e:
                print(f"Error cargando metadatos del chat {chat_id}: {str(e)}")
        
        # Luego procesamos los archivos de mensajes que no tienen metadatos
        for filename in chat_files:
            chat_id = filename.replace('.json', '')
            
            # Saltar si ya lo cargamos desde metadatos
            if chat_id in loaded_chats:
                continue
                
            chat_file_path = os.path.join(CHAT_DIR, filename)
            
            try:
                # Cargar los mensajes del chat
                with open(chat_file_path, 'r') as f:
                    messages = json.load(f)
                
                # Solo procesar si hay mensajes
                if not messages:
                    continue
                
                # Crear una entrada en el diccionario de chats
                first_message = next((msg for msg in messages if msg['role'] == 'user'), None)
                model_name = settings['model']
                print(f"Load chats from files() Model name: {model_name}")
                # Usar solo el contenido del mensaje como título, sin ninguna referencia al modelo
                title = model_name + ' - ' + first_message['content'][:30] + ('...' if len(first_message['content']) > 30 else '') if first_message else chat_id
                print(f"Load chats from files() Title: {title}")

                # Encontrar la última fecha de mensaje
                last_message_time = messages[-1]['timestamp'] if messages else datetime.now().isoformat()
                created_at = messages[0]['timestamp'] if messages else datetime.now().isoformat()
                
                # Crear estructura del chat
                loaded_chats[chat_id] = {
                    "id": chat_id,
                    "title": title,
                    "created_at": created_at,
                    "last_message": last_message_time,
                    "messages": messages
                }
                
                # Crear archivo de metadatos para este chat
                save_chat_metadata(chat_id)
                
                print(f"Cargado chat {chat_id} con {len(messages)} mensajes y creado metadatos")
                
            except Exception as e:
                print(f"Error cargando chat {chat_id}: {str(e)}")
        
        # Actualizar el diccionario global de chats
        return loaded_chats
    
    except Exception as e:
        print(f"Error general cargando chats: {str(e)}")
        return {}

# Cargar los chats al iniciar el servidor
chats = load_chats_from_files()
print(f"Se cargaron {len(chats)} chats desde archivos")

# Función para asegurar que un chat tiene la estructura correcta
def ensure_chat_structure(chat_id):
    """Asegura que un chat tenga la estructura correcta con todos los campos necesarios."""
    if chat_id not in chats:
        return False
    
    # Comprobar si el chat tiene la clave 'messages'
    if 'messages' not in chats[chat_id]:
        print(f"Chat {chat_id} no tiene la clave 'messages'. Añadiendo...")
        chats[chat_id]['messages'] = []
    
    # Asegurar que todas las claves necesarias existen
    required_keys = ['id', 'title', 'created_at', 'last_message', 'messages']
    for key in required_keys:
        if key not in chats[chat_id]:
            if key == 'id':
                chats[chat_id][key] = chat_id
            elif key in ['title', 'created_at', 'last_message']:
                chats[chat_id][key] = datetime.now().isoformat()
            elif key == 'messages':
                chats[chat_id][key] = []
    
    return True

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Chat API is running. Use /api/* endpoints to interact with the API v2."})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    print(f"def chat() Message: {message}")
    chat_id = data.get('chat_id')
    
    if not chat_id:
        # Create a new chat session
        chat_id = str(uuid.uuid4())
        
        # Use the model name from the assistant's response to generate the chat title
        current_model = settings['model']
        print(f"def chat() Current model: {current_model}")
        # Usar solo el contenido del mensaje como título, sin ninguna referencia al modelo
        title = current_model + ' - ' + message[:30] + ('...' if len(message) > 30 else '') if message else chat_id
        print(f"def chat() Title: {title}")
        timestamp = datetime.now().isoformat()
        
        # Save chat in memory with proper structure
        chats[chat_id] = {
            "id": chat_id,
            "title": title,
            "created_at": timestamp,
            "last_message": timestamp,
            "messages": []  # Initialize with empty messages list
        }
        
        # Guardar metadatos para persistencia
        save_chat_metadata(chat_id)
    else:
        # Ensure chat has correct structure if it exists
        if chat_id in chats:
            ensure_chat_structure(chat_id)
        else:
            # If chat_id was provided but doesn't exist, create it
            timestamp = datetime.now().isoformat()
            current_model = settings['model']
            print(f"def chat2() Current model: {current_model}")
            # Usar solo el contenido del mensaje como título, sin ninguna referencia al modelo
            title = current_model + ' - ' + message[:30] + ('...' if len(message) > 30 else '') if message else chat_id
            print(f"def chat2() Title: {title}")
            chats[chat_id] = {
                "id": chat_id,
                "title": title,
                "created_at": timestamp,
                "last_message": timestamp,
                "messages": []
            }
            
            # Guardar metadatos para persistencia
            save_chat_metadata(chat_id)
    
    # Load existing chat history from file if available
    chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if os.path.exists(chat_file):
        try:
            with open(chat_file, 'r') as f:
                chat_history = json.load(f)
                # Ensure all messages have model and provider information
                for msg in chat_history:
                    if 'model' not in msg:
                        msg['model'] = settings.get('model', 'unknown')
                    if 'provider' not in msg:
                        msg['provider'] = settings.get('provider', 'unknown')
                # Update in-memory representation
                chats[chat_id]["messages"] = chat_history
        except Exception as e:
            print(f"Error loading chat history: {str(e)}")
            chat_history = []
    else:
        chat_history = chats[chat_id]["messages"]
    
    # Update last_message timestamp
    chats[chat_id]["last_message"] = datetime.now().isoformat()
    
    # Use current settings
    current_provider = settings["provider"]
    current_model = settings["model"]
    maintain_context = settings.get("maintain_context", True)  # Por defecto True
    
    print(f"Using provider: {current_provider}, model: {current_model} in /api/chat endpoint")
    print(f"Maintain context: {maintain_context}")
    
    # Add user message to history with current model and provider
    chat_history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat(),
        "model": current_model,
        "provider": current_provider
    })
    
    # Update LLM service with current settings
    if current_provider == "openai":
        llm_service.update_settings(
            provider="openai",  # Forzar explícitamente el proveedor
            model=current_model
        )
    elif current_provider == "ollama":
        ollama_base_url = settings.get("ollama_base_url", "http://localhost:11434")
        llm_service.update_settings(
            provider="ollama",  # Forzar explícitamente el proveedor
            model=current_model,
            ollama_base_url=ollama_base_url
        )
    elif current_provider == "google":
        google_api_key = settings.get("google_api_key", os.environ.get("GOOGLE_API_KEY"))
        llm_service.update_settings(
            provider="google",
            model=current_model,
            google_api_key=google_api_key
        )
    
    # Get response from LLM, pasando explícitamente el proveedor
    if maintain_context:
        # Usar todo el historial si maintain_context está activado
        print("Using full conversation history")
        start_time = datetime.now()
        response = llm_service.generate_response(message, chat_history, provider=current_provider)
        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
    else:
        # Usar sólo el último mensaje si maintain_context está desactivado
        print("Using only the last message (no context)")
        start_time = datetime.now()
        response = llm_service.generate_response(message, provider=current_provider)
        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
    
    # Add assistant response to history with response time and model info
    chat_history.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat(),
        "responseTimeMs": response_time_ms,
        "model": current_model,
        "provider": current_provider
    })
    
    # Save updated chat history
    with open(os.path.join(CHAT_DIR, f"{chat_id}.json"), 'w') as f:
        json.dump(chat_history, f)
    
    return jsonify({
        "chat_id": chat_id,
        "response": response,
        "history": chat_history
    })

@app.route('/api/chats', methods=['GET'])
def get_chats():
    # Obtener todos los chats como una lista
    chats_list = list(chats.values())
    
    # Ordenar los chats por fecha de último mensaje (del más reciente al más antiguo)
    sorted_chats = sorted(
        chats_list, 
        key=lambda chat: chat.get('last_message', chat.get('created_at', '')), 
        reverse=True  # Orden descendente (más reciente primero)
    )
    
    return jsonify(sorted_chats), 200

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    return jsonify(chats[chat_id]), 200

@app.route('/api/chats/<chat_id>', methods=['PUT'])
def update_chat(chat_id):
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    data = request.json
    
    if 'title' in data:
        chats[chat_id]['title'] = data['title']
        
    # Guardar los metadatos actualizados
    save_chat_metadata(chat_id)
    
    return jsonify(chats[chat_id]), 200

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    # Remove from memory
    del chats[chat_id]
    
    # Also remove the chat history file if it exists
    chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if os.path.exists(chat_file):
        try:
            os.remove(chat_file)
        except Exception as e:
            print(f"Error removing chat file: {e}")
    
    # Remove metadata file if it exists
    metadata_file = os.path.join(CHAT_DIR, f"{chat_id}_metadata.json")
    if os.path.exists(metadata_file):
        try:
            os.remove(metadata_file)
        except Exception as e:
            print(f"Error removing metadata file: {e}")
    
    return jsonify({"status": "success"}), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    chat_id = request.form.get('chat_id')
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    # Create a directory for this chat's uploads if it doesn't exist
    chat_upload_dir = os.path.join(UPLOAD_DIR, chat_id)
    os.makedirs(chat_upload_dir, exist_ok=True)
    
    # Save the file
    filename = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(chat_upload_dir, filename)
    file.save(file_path)
    
    # Process the file content (this would depend on the file type)
    # For now, we'll just acknowledge the upload
    return jsonify({
        "status": "success",
        "filename": filename,
        "message": f"File {file.filename} uploaded successfully"
    })

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(settings), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        if settings['provider'] == 'openai':
            models = llm_service.get_available_models()
        elif settings['provider'] == 'ollama':
            models = llm_service.get_available_models(provider='ollama')
        elif settings['provider'] == 'google':
            models = llm_service.get_available_models(provider='google')
        else:
            models = []
            
        return jsonify({"models": models}), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "models": []
        }), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    
    if 'provider' in data:
        settings['provider'] = data['provider']
        
        # Actualizar el modelo según el proveedor
        if 'model' in data:
            settings['model'] = data['model']
    
    if 'temperature' in data:
        settings['temperature'] = data['temperature']
    
    if 'max_tokens' in data:
        settings['max_tokens'] = data['max_tokens']
        
    if 'ollama_base_url' in data:
        settings['ollama_base_url'] = data['ollama_base_url']
    
    if 'google_api_key' in data:
        settings['google_api_key'] = data['google_api_key']
    
    # Nueva opción para mantener el contexto
    if 'maintain_context' in data:
        settings['maintain_context'] = data['maintain_context']
    
    # Nueva opción para número de chunks RAG
    if 'rag_num_chunks' in data:
        settings['rag_num_chunks'] = data['rag_num_chunks']
    
    # Update LLM service
    try:
        if settings['provider'] == 'openai':
            llm_service.update_settings(
                provider="openai",
                model=settings['model'],
                temperature=settings['temperature'],
                max_tokens=settings['max_tokens']
            )
        elif settings['provider'] == 'ollama':
            llm_service.update_settings(
                provider="ollama",
                model=settings['model'],
                temperature=settings['temperature'],
                max_tokens=settings['max_tokens'],
                ollama_base_url=settings['ollama_base_url']
            )
        elif settings['provider'] == 'google':
            google_api_key = settings.get('google_api_key', os.environ.get('GOOGLE_API_KEY'))
            llm_service.update_settings(
                provider="google",
                model=settings['model'],
                temperature=settings['temperature'],
                max_tokens=settings['max_tokens'],
                google_api_key=google_api_key
            )
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    
    return jsonify({
        "status": "success",
        "settings": settings
    }), 200

@app.route('/api/settings/api-key', methods=['POST'])
def update_api_key():
    data = request.json
    
    if 'api_key' not in data:
        return jsonify({"error": "API key is required"}), 400
    
    provider = data.get('provider', settings.get('provider', 'openai'))
    
    # Update API key in LLM service based on provider
    try:
        if provider == 'google':
            # Update Google API key
            settings['google_api_key'] = data['api_key']
            llm_service.update_api_key(data['api_key'], provider='google')
        else:
            # Default to OpenAI
            settings['api_key'] = data['api_key']
            llm_service.update_api_key(data['api_key'], provider='openai')
            
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def add_message(chat_id):
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    data = request.json
    
    if 'content' not in data:
        return jsonify({"error": "Message content is required"}), 400
    
    # Use current settings
    current_provider = settings["provider"]
    current_model = settings["model"]
    current_temperature = settings["temperature"]
    current_max_tokens = settings["max_tokens"]
    ollama_base_url = settings.get("ollama_base_url", "http://localhost:11434")
    maintain_context = settings.get("maintain_context", True)  # Por defecto True
    
    print(f"Current provider: {current_provider}, Current model: {current_model}")
    print(f"Maintain context: {maintain_context}")
    
    timestamp = datetime.now().isoformat()
    
    # User message
    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": data['content'],
        "timestamp": timestamp,
        "model": current_model,
        "provider": current_provider
    }
    
    chats[chat_id]['messages'].append(user_message)
    chats[chat_id]['last_message'] = timestamp
    
    # Actualizar metadatos después de añadir el mensaje del usuario
    save_chat_metadata(chat_id)
    
    # If this is the first message, set the chat title
    if len(chats[chat_id]['messages']) == 1:
        model_name = current_model
        print(f"def add_message() Model name: {model_name}")
        # Usar solo el contenido del mensaje como título, sin ninguna referencia al modelo
        title = model_name + ' - ' + data['content'][:30] + ('...' if len(data['content']) > 30 else '') if len(data['content']) > 0 else chat_id
        print(f"def add_message() Title: {title}")
        chats[chat_id]['title'] = title
        
        # Actualizar metadatos después de cambiar el título
        save_chat_metadata(chat_id)
    
    # Generate response using the LLM
    try:
        # Si estamos usando Ollama, verificar que el modelo existe
        if current_provider == "ollama":
            # Obtener modelos disponibles en Ollama
            print(f"Checking available Ollama models before generating response...")
            available_models = llm_service.get_available_models(provider="ollama")
            print(f"Available Ollama models: {available_models}")
            
            # Si el modelo seleccionado no está disponible, intentar usar un modelo alternativo
            if current_model not in available_models:
                print(f"Model {current_model} not found in available models")
                if available_models:
                    # Usar el primer modelo disponible
                    current_model = available_models[0]
                    settings["model"] = current_model
                    print(f"Switching to available model: {current_model}")
                else:
                    return jsonify({
                        "status": "error",
                        "message": {
                            "role": "system",
                            "content": f"No Ollama models available. Please make sure Ollama is running and you have downloaded models."
                        }
                    }), 400
        
        # Update LLM service with current settings
        if current_provider == "openai":
            llm_service.update_settings(
                provider="openai",  # Forzar explícitamente el proveedor
                model=current_model,
                temperature=current_temperature,
                max_tokens=current_max_tokens
            )
        elif current_provider == "ollama":
            llm_service.update_settings(
                provider="ollama",  # Forzar explícitamente el proveedor
                model=current_model,
                temperature=current_temperature,
                max_tokens=current_max_tokens,
                ollama_base_url=ollama_base_url
            )
        elif current_provider == "google":
            google_api_key = settings.get("google_api_key", os.environ.get("GOOGLE_API_KEY"))
            llm_service.update_settings(
                provider="google",
                model=current_model,
                google_api_key=google_api_key
            )
        
        # Get chat history for context
        history = [{"role": msg["role"], "content": msg["content"]} 
                   for msg in chats[chat_id]['messages'] 
                   if msg["role"] in ["user", "assistant"]]
        
        print(f"Generating response with provider: {current_provider}, model: {current_model}")
        
        # Generate response - considerar la opción maintain_context
        if maintain_context:
            # Usar todo el historial si maintain_context está activado
            print("Using full conversation history")
            start_time = datetime.now()
            response_content = llm_service.generate_response(
                history=history,
                provider=current_provider  # Pasar explícitamente el proveedor actual
            )
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        else:
            # Usar sólo el último mensaje si maintain_context está desactivado
            print("Using only the last message (no context)")
            start_time = datetime.now()
            last_message = {"role": "user", "content": data['content']}
            response_content = llm_service.generate_response(
                history=[last_message],
                provider=current_provider  # Pasar explícitamente el proveedor actual
            )
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Assistant message
        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": response_content,
            "timestamp": datetime.now().isoformat(),
            "responseTimeMs": response_time_ms,
            "model": current_model,
            "provider": current_provider,
            "tokenCount": token_count,  # Guardar siempre el conteo de tokens
            "endTimestamp": datetime.now().isoformat(),  # Guardar timestamp final
            "totalElapsedTime": (datetime.now() - start_time).total_seconds() * 1000,  # Guardar tiempo total en ms
            "timePerToken": ((datetime.now() - start_time).total_seconds() * 1000 - (response_time_ms or 0)) / token_count if token_count > 0 else 50,  # Calcular tiempo por token
            "tokensPerSecond": (token_count / (response_time_ms / 1000)) if response_time_ms > 0 else 0  # Nueva métrica: tokens por segundo
        }
        
        chats[chat_id]['messages'].append(assistant_message)
        chats[chat_id]['last_message'] = assistant_message['timestamp']
        
        # Actualizar metadatos después de añadir el mensaje del asistente
        save_chat_metadata(chat_id)
        
        # Guardar los mensajes en el archivo del chat
        chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
        try:
            with open(chat_file, 'w') as f:
                json.dump(chats[chat_id]["messages"], f)
            print(f"Chat history saved to file: {chat_file}")
        except Exception as e:
            print(f"Error saving chat history to file: {str(e)}")
        
        # Send completion message con todas las métricas calculadas
        completion_data = {
            "type": "done",
            "message_id": assistant_message["id"],
            "token_count": token_count,
            "time_per_token": assistant_message["timePerToken"],
            "total_elapsed_time": assistant_message["totalElapsedTime"],
            "response_time_ms": assistant_message["responseTimeMs"],
            "generation_time_ms": assistant_message["generationTimeMs"],
            "tokens_per_second": assistant_message["tokensPerSecond"]
        }
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        print(f"Error generating response: {e}")
        
        # Add error message to chat
        error_message = {
            "id": str(uuid.uuid4()),
            "role": "system",
            "content": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "model": current_model,
            "provider": current_provider
        }
        chats[chat_id]['messages'].append(error_message)
        chats[chat_id]['last_message'] = error_message["timestamp"]
        
        # Actualizar metadatos después de añadir el mensaje de error
        save_chat_metadata(chat_id)
        
        # Guardar los mensajes en el archivo del chat
        chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
        try:
            with open(chat_file, 'w') as f:
                json.dump(chats[chat_id]["messages"], f)
            print(f"Error message saved to chat file: {chat_file}")
        except Exception as e:
            print(f"Error saving chat history to file: {str(e)}")
        
        return jsonify({
            "status": "error",
            "message": error_message
        }), 500

@app.route('/api/chats', methods=['POST'])
def create_chat():
    data = request.json
    title = data.get('title', 'New Chat')
    
    chat_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    chat = {
        "id": chat_id,
        "title": title,
        "created_at": timestamp,
        "last_message": timestamp,
        "messages": []  # Añadimos la clave messages como una lista vacía
    }
    
    chats[chat_id] = chat
    
    # Guardar metadatos para persistencia
    save_chat_metadata(chat_id)
    
    return jsonify(chat), 201

@app.route('/api/chat/<chat_id>', methods=['GET'])
def get_chat_messages(chat_id):
    """Get all messages for a specific chat"""
    # Check if chat exists in memory
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    # Ensure the chat has the correct structure
    ensure_chat_structure(chat_id)
    
    # Path to the chat history file
    chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
    
    # Get chat history
    if os.path.exists(chat_file):
        with open(chat_file, 'r') as f:
            try:
                chat_history = json.load(f)
                
                # Ensure all assistant messages have responseTimeMs
                for msg in chat_history:
                    if msg['role'] == 'assistant' and 'responseTimeMs' not in msg:
                        msg['responseTimeMs'] = 0  # Add a default value
                
                # También actualizamos la estructura en memoria para mantener consistencia
                chats[chat_id]['messages'] = chat_history
                
                return jsonify(chat_history)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from chat file: {chat_file}")
                return jsonify([]), 200
    else:
        # If the file doesn't exist yet, return the messages from memory
        return jsonify(chats[chat_id]['messages']), 200

@app.route('/api/chat/<chat_id>', methods=['DELETE'])
def delete_chat_messages(chat_id):
    """Delete all messages for a specific chat"""
    chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
    
    # Check if chat exists
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    # Ensure chat has correct structure
    ensure_chat_structure(chat_id)
    
    # Delete chat history file if it exists
    if os.path.exists(chat_file):
        try:
            os.remove(chat_file)
            print(f"Deleted chat file: {chat_file}")
        except Exception as e:
            print(f"Error removing chat file: {e}")
    
    # Reset chat history in memory
    chats[chat_id]['messages'] = []
    chats[chat_id]['last_message'] = datetime.now().isoformat()
    
    return jsonify({"status": "success"}), 200

@app.route('/api/chat/stream/cancel/<chat_id>', methods=['POST'])
def cancel_streaming(chat_id):
    """
    Endpoint para cancelar un streaming en curso para un chat específico.
    """
    if chat_id in streaming_active_requests:
        print(f"Request to cancel streaming for chat {chat_id}")
        streaming_active_requests[chat_id] = False
        return jsonify({"message": "Streaming cancellation requested"}), 200
    else:
        return jsonify({"message": "No active streaming found for the specified chat"}), 404

@app.route('/api/chat/stream', methods=['POST'])
def stream_chat():
    """
    Stream response from the LLM using server-sent events (SSE).
    """
    # Configurar para evitar cualquier tipo de almacenamiento en búfer
    from flask import Response, stream_with_context
    
    data = request.json
    message = data.get('message', '')
    chat_id = data.get('chat_id')
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    # Check if this is a RAG query with **ProjectName** format
    processed_message, rag_context, project_id = process_rag_prompt(message)
    
    # If RAG context was found, we'll use the processed message
    is_rag_query = rag_context is not None
    project_system_prompt = None
    
    if is_rag_query:
        # Obtener el system_prompt del proyecto usando el project_id
        if project_id and project_id in projects:
            project_system_prompt = projects[project_id].get('system_prompt', '')
                
        message = processed_message
        print(f"RAG query detected. Using project context. Original message length: {len(message)}")
        print(f"RAG context length: {len(rag_context)}")
        if project_system_prompt:
            print(f"Found project system prompt, length: {len(project_system_prompt)}")
        else:
            print("No project system prompt found or empty system prompt")
    
    # Use current settings
    current_provider = settings.get("provider", "openai")
    current_model = settings.get("model", "gpt-3.5-turbo")
    maintain_context = settings.get("maintain_context", True)
    
    print(f"Using provider: {current_provider}, model: {current_model} in streaming endpoint")
    print(f"Maintain context: {maintain_context}")
    
    if not chat_id:
        # Create a new chat session
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        model_name = settings['model']
        print(f"def stream_chat() Model name: {model_name}")
        # Usar solo el contenido del mensaje como título, sin ninguna referencia al modelo
        title = model_name + ' - ' + message[:30] + ('...' if len(message) > 30 else '')
        print(f"def stream_chat() Title: {title}")
        # Save chat in memory
        chats[chat_id] = {
            "id": chat_id,
            "title": title,
            "created_at": timestamp,
            "last_message": timestamp,
            "messages": []
        }
        
        # Guardar metadatos para persistencia
        save_chat_metadata(chat_id)
    else:
        # Ensure the chat structure is correct
        if chat_id in chats:
            ensure_chat_structure(chat_id)
        else:
            # Create the chat if it doesn't exist
            timestamp = datetime.now().isoformat()
            model_name = settings['model'] 
            print(f"def stream_chat() Model name: {model_name}")
            # Usar solo el contenido del mensaje como título, sin ninguna referencia al modelo
            title = model_name + ' - ' + message[:30] + ('...' if len(message) > 30 else '')
            print(f"def stream_chat() Title: {title}")
            chats[chat_id] = {
                "id": chat_id,
                "title": title,
                "created_at": timestamp,
                "last_message": timestamp,
                "messages": []
            }
            
            # Guardar metadatos para persistencia
            save_chat_metadata(chat_id)
        
    # Add user message to the chat - we store the original message with **ProjectName** if it was a RAG query
    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": data.get('message', ''),  # Store original message
        "timestamp": datetime.now().isoformat(),
        "model": current_model,
        "provider": current_provider
    }
    
    # Add user message to chat history
    chats[chat_id]["messages"].append(user_message)
    chats[chat_id]["last_message"] = user_message["timestamp"]
    
    # Actualizar metadatos después de añadir el mensaje del usuario
    save_chat_metadata(chat_id)
    
    # Update LLM service with current settings
    if current_provider == "openai":
        llm_service.update_settings(
            provider="openai",
            model=current_model
        )
    elif current_provider == "ollama":
        ollama_base_url = settings.get("ollama_base_url", "http://localhost:11434")
        llm_service.update_settings(
            provider="ollama",
            model=current_model,
            ollama_base_url=ollama_base_url
        )
    elif current_provider == "google":
        google_api_key = settings.get("google_api_key", os.environ.get("GOOGLE_API_KEY"))
        llm_service.update_settings(
            provider="google",
            model=current_model,
            google_api_key=google_api_key
        )
    
    # Record start time for response time calculation
    start_time = datetime.now()
    
    # Get history based on maintain_context setting
    history = []
    if maintain_context:
        # Use full conversation history
        history = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in chats[chat_id]["messages"] 
            if msg["role"] in ["user", "assistant"]
        ]
        
        # If this is a RAG query, we replace the last user message with the processed message
        # and add the RAG context as a system message
        if is_rag_query and history:
            # Replace the last user message with the processed message (without **ProjectName**)
            for i in range(len(history)-1, -1, -1):
                if history[i]["role"] == "user":
                    history[i]["content"] = message
                    break
            
            # Add project system_prompt as the first system message if available
            if project_system_prompt:
                # Insertar al inicio del historial
                history.insert(0, {
                    "role": "system", 
                    "content": project_system_prompt
                })
            
            # Add RAG context as a system message before the user's query
            if rag_context:
                history.insert(len(history)-1, {
                    "role": "system", 
                    "content": f"The following is relevant information for answering the user's query:\n\n{rag_context}"
                })
    else:
        # Use only the last user message (for non-context mode)
        # If it's a RAG query, use the processed message + context as system message
        if is_rag_query:
            history = []
            
            # Add project system_prompt as the first system message if available
            if project_system_prompt:
                history.append({"role": "system", "content": project_system_prompt})
            
            # Add RAG context as system message
            history.append({"role": "system", "content": f"The following is relevant information for answering the user's query:\n\n{rag_context}"})
            
            # Add user message
            history.append({"role": "user", "content": message})
        else:
            history = [{"role": "user", "content": message}]
    
    # Registramos esta solicitud de streaming como activa
    streaming_active_requests[chat_id] = True
    
    # Function to estimate token count (very rough estimation)
    def estimate_tokens(text):
        # Average English word length is ~5 characters, and tokens are ~4 characters on average
        # So we'll estimate 1 token per 4 characters
        return len(text) // 4
    
    # Log the complete prompt being sent to the LLM
    print("\n" + "*"*100)
    print("PROMPT COMPLETO ENVIADO AL LLM:")
    print("*"*100)
    
    total_tokens = 0
    for msg in history:
        token_estimate = estimate_tokens(msg['content'])
        total_tokens += token_estimate
        print(f"\n[{msg['role'].upper()}] - Tokens aproximados: {token_estimate}")
        print("-"*80)
        # Imprimir el texto completo para mensajes normales
        print(msg['content'])
        # Si es un mensaje del sistema, destacarlo como System Prompt
        if msg['role'] == 'system':
            print("\nSYSTEM PROMPT:")
            print(msg['content'])
        print("-"*80)
    
    print(f"\nTOTAL DE TOKENS APROXIMADOS: {total_tokens}")
    print("*"*100 + "\n")
    
    def generate():
        """Generate Server-Sent Events (SSE) for streaming response."""
        try:
            # Start with response metadata
            metadata = {
                "chat_id": chat_id,
                "type": "metadata",
                "provider": current_provider,
                "model": current_model
            }
            print(f"Sending initial metadata: {json.dumps(metadata)}")
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # Initialize empty response
            assistant_content = ""
            first_token = True
            response_time_ms = None
            token_count = 0
            
            # Stream tokens
            try:
                print(f"Starting streaming with provider: {current_provider}, model: {current_model}")
                print(f"History length: {len(history)} messages")
                
                stream_iterator = llm_service.generate_streaming_response(history=history, provider=current_provider)
                
                for token in stream_iterator:
                    # Verificar si el chat ha solicitado cancelación
                    if not streaming_active_requests.get(chat_id, True):
                        print(f"Streaming for chat {chat_id} cancelled via API request")
                        break
                    
                    # Calculate response time for first token
                    if first_token:
                        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                        print(f"First token received after {response_time_ms:.2f}ms: '{token}'")
                        first_token = False
                    
                    # Add token to accumulated content
                    assistant_content += token
                    token_count += 1
                    
                    # Create token data
                    token_data = {
                        "type": "token",
                        "token": token,
                        "response_time_ms": response_time_ms if response_time_ms is not None else 0
                    }
                    
                    # Imprimir cada 20 tokens para no llenar los logs
                    if token_count % 20 == 0:
                        print(f"Streaming token #{token_count}: '{token}'")
                    
                    # Send token as SSE event
                    yield f"data: {json.dumps(token_data)}\n\n"
                
                # Calcular métricas finales
                end_time = datetime.now()
                end_timestamp = end_time.isoformat()
                total_elapsed_time_ms = (end_time - start_time).total_seconds() * 1000
                generation_time_ms = total_elapsed_time_ms - (response_time_ms or 0)
                time_per_token = (generation_time_ms / token_count) if token_count > 0 else 0
                tokens_per_second = (token_count / (generation_time_ms / 1000)) if generation_time_ms > 0 else 0
                word_count = len(assistant_content.split())
                words_per_second = (word_count / (generation_time_ms / 1000)) if generation_time_ms > 0 else 0
                
                # Send completion event
                completion_data = {
                    "type": "done",
                    "content": assistant_content,
                    "token_count": token_count,
                    "total_elapsed_time": total_elapsed_time_ms,
                    "response_time_ms": response_time_ms,
                    "generation_time_ms": generation_time_ms,
                    "time_per_token": time_per_token,
                    "tokens_per_second": tokens_per_second,
                    "word_count": word_count,
                    "words_per_second": words_per_second
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
                # Guardar el mensaje completo en la base de datos
                assistant_message = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": assistant_content,
                    "timestamp": datetime.now().isoformat(),
                    "responseTimeMs": response_time_ms if response_time_ms is not None else 0,
                    "model": current_model,
                    "provider": current_provider,
                    "tokenCount": token_count,
                    "endTimestamp": end_timestamp,
                    "totalElapsedTime": total_elapsed_time_ms,
                    "timePerToken": time_per_token,
                    "generationTimeMs": generation_time_ms,
                    "tokensPerSecond": tokens_per_second,
                    "wordCount": word_count,
                    "wordsPerSecond": words_per_second
                }
                
                # Add to chat history
                chats[chat_id]["messages"].append(assistant_message)
                chats[chat_id]["last_message"] = assistant_message["timestamp"]
                
                # Actualizar metadatos después de añadir el mensaje del asistente
                save_chat_metadata(chat_id)
                
                # Guardar los mensajes en el archivo del chat
                chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
                try:
                    with open(chat_file, 'w') as f:
                        json.dump(chats[chat_id]["messages"], f)
                    print(f"Chat history saved to file: {chat_file}")
                except Exception as e:
                    print(f"Error saving chat history to file: {str(e)}")
                
            except Exception as e:
                error_data = {
                    "type": "error",
                    "error": str(e)
                }
                print(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps(error_data)}\n\n"
                
        except Exception as e:
            error_data = {
                "type": "error",
                "error": str(e)
            }
            print(f"Error in generate function: {str(e)}")
            yield f"data: {json.dumps(error_data)}\n\n"
            
        # Al final del proceso de streaming, eliminamos este chat del registro de activos
        if chat_id in streaming_active_requests:
            del streaming_active_requests[chat_id]
    
    # Create a response with the generator function
    response = Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
            'Content-Encoding': 'none'
        }
    )
    
    return response

@app.route('/api/chats/<chat_id>/export-pdf', methods=['GET'])
def export_chat_to_pdf(chat_id):
    """
    Exporta un chat específico a un archivo PDF que refleja la apariencia visual de la interfaz.
    """
    if chat_id not in chats:
        return jsonify({"error": "Chat no encontrado"}), 404
    
    # Asegurar que la estructura del chat es correcta
    ensure_chat_structure(chat_id)
    
    # Cargar el historial de mensajes
    chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if os.path.exists(chat_file):
        try:
            with open(chat_file, 'r') as f:
                messages = json.load(f)
                
                # Procesar los mensajes para asegurar que tengan conteo de tokens
                for msg in messages:
                    # Si no tiene tokenCount, intentar calcularlos
                    if 'tokenCount' not in msg and 'content' in msg:
                        msg['tokenCount'] = estimate_tokens(msg['content'])
                        print(f"Added estimated token count for message: {msg['tokenCount']}")
                    
                    # Imprimir los mensajes para depuración
                    print(f"Message role: {msg.get('role')}, token count: {msg.get('tokenCount')}, has content: {'Yes' if 'content' in msg else 'No'}")
        except Exception as e:
            print(f"Error cargando historial del chat: {str(e)}")
            messages = chats[chat_id].get('messages', [])
    else:
        messages = chats[chat_id].get('messages', [])
        
        # Procesar los mensajes para asegurar que tengan conteo de tokens
        for msg in messages:
            # Si no tiene tokenCount, intentar calcularlos
            if 'tokenCount' not in msg and 'content' in msg:
                msg['tokenCount'] = estimate_tokens(msg['content'])
                print(f"Added estimated token count for message: {msg['tokenCount']}")
            
            # Imprimir los mensajes para depuración
            print(f"Memory message role: {msg.get('role')}, token count: {msg.get('tokenCount')}, has content: {'Yes' if 'content' in msg else 'No'}")
    
    # Crear un buffer para el PDF
    buffer = io.BytesIO()
    
    # Configurar el documento PDF
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
        allowSplitting=1,  # Permitir división de contenido entre páginas
        title=f"Chat: {chats[chat_id].get('title', f'Chat {chat_id}')}"
    )
    styles = getSampleStyleSheet()
    
    # Definir colores que coincidan con la interfaz de la aplicación
    background_color = colors.HexColor('#0B0F18')  # Fondo oscuro azulado
    text_color = colors.HexColor('#e0e0e0')        # Texto claro
    user_bubble_color = colors.HexColor('#2a2e42')  # Color oscuro para mensajes de usuario (en lugar de celeste)
    assistant_bubble_color = colors.HexColor('#1F2937')  # Gris oscuro para mensajes del asistente
    accent_color = colors.HexColor('#BB4333')      # Color rojo de acento usado en links
    code_bg_color = colors.HexColor('#282C34')     # Fondo para bloques de código
    
    # Crear estilos personalizados
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=text_color,
        spaceAfter=10,
        alignment=1  # Centrado
    )
    
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontName='Helvetica',
        fontSize=14,
        textColor=text_color,
        spaceBefore=10,
        spaceAfter=12,
        alignment=1  # Centrado
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        spaceBefore=2,
        spaceAfter=2
    )
    
    # Estilo para mensajes del usuario
    user_style = ParagraphStyle(
        'UserStyle',
        parent=normal_style,
        fontName='Helvetica-Bold',
        fontSize=8,
        textColor=colors.HexColor('#a0a0a0'),
        spaceBefore=4,
        spaceAfter=2,
        alignment=2  # Alineación a la derecha para el encabezado
    )
    
    user_content_style = ParagraphStyle(
        'UserContentStyle',
        parent=normal_style,
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,  # Use the brighter text color for content
        spaceAfter=8,
        leftIndent=10,
        rightIndent=10,
        alignment=2  # Alineación a la derecha (2=right, 1=center, 0=left)
    )
    
    # Estilo para mensajes del asistente
    assistant_style = ParagraphStyle(
        'AssistantStyle',
        parent=normal_style,
        fontName='Helvetica-Bold',
        fontSize=8,
        textColor=colors.HexColor('#a0a0a0'),
        spaceBefore=4,
        spaceAfter=2
    )
    
    assistant_content_style = ParagraphStyle(
        'AssistantContentStyle',
        parent=normal_style,
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,  # Use the brighter text color for content
        spaceAfter=8,
        leftIndent=10,
        rightIndent=10
    )
    
    # Estilo para encabezados de bloques de código
    code_header_style = ParagraphStyle(
        'CodeHeaderStyle',
        parent=normal_style,
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor('#90CAF9'),
        spaceBefore=4,
        spaceAfter=2
    )
    
    # Estilo para bloques de código
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=normal_style,
        fontName='Courier',
        fontSize=9,
        textColor=colors.white,
        backColor=code_bg_color,
        spaceBefore=0,
        spaceAfter=0,
        leftIndent=20,
        rightIndent=20,
        firstLineIndent=0,
        borderWidth=1,
        borderColor=colors.grey,
        borderPadding=6,
        borderRadius=4
    )
    
    # Estilo para mensajes del sistema
    system_style = ParagraphStyle(
        'SystemStyle',
        parent=normal_style,
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=6,
        alignment=1  # Centrado
    )
    
    # Estilo para links
    link_style = ParagraphStyle(
        'LinkStyle',
        parent=normal_style,
        textColor=accent_color,
        underline=True
    )
    
    # Definir contenido del PDF
    elements = []
    
    # Agregar una página de título
    chat_title = chats[chat_id].get('title', f"Chat {chat_id}")
    elements.append(Paragraph(f"Conversation: {chat_title}", title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Información del chat
    created_at = chats[chat_id].get('created_at', '')
    if created_at:
        try:
            created_date = datetime.fromisoformat(created_at)
            date_str = created_date.strftime("%d/%m/%Y %H:%M:%S")
            elements.append(Paragraph(f"Created on: {date_str}", subtitle_style))
        except:
            elements.append(Paragraph(f"Created on: {created_at}", subtitle_style))
    
    elements.append(Spacer(1, 0.5*inch))
    
    # Función para procesar markdown en formato más fiel a la pantalla
    def process_markdown_for_pdf(text, is_user_message=False):
        result_elements = []
        style_to_use = user_content_style if is_user_message else assistant_content_style
        
        # Detectar bloques de código
        code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        
        # Si hay bloques de código, dividir el texto y procesar cada parte
        if code_blocks:
            parts = re.split(r'```(.*?)```', text, flags=re.DOTALL)
            
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Texto normal (fuera de bloques de código)
                    if part.strip():
                        # Procesar el texto normal
                        process_text_part(part, style_to_use, result_elements)
                else:  # Bloque de código
                    if part.strip():
                        # Eliminar la declaración de lenguaje si existe
                        code_content = re.sub(r'^.*\n', '', part, count=1) if '\n' in part else part
                        
                        # Extraer el lenguaje de programación si está disponible
                        lang_match = re.match(r'^([a-zA-Z0-9+#]+)', part)
                        lang_name = lang_match.group(1) if lang_match else ""
                        if lang_name:
                            result_elements.append(Paragraph(lang_name, code_header_style))
                        
                        # Reemplazar caracteres especiales para evitar problemas
                        code_content = code_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        
                        # Separar por líneas para conservar espacios y saltos
                        code_lines = code_content.split('\n')
                        
                        # Crear una tabla para el bloque de código con fondo unificado
                        rows = []
                        for line in code_lines:
                            rows.append([Paragraph(line, code_style)])
                        
                        if rows:
                            code_table = Table(rows, colWidths=[400])
                            code_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (0, -1), code_bg_color),
                                ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
                                ('FONT', (0, 0), (0, -1), 'Courier'),
                                ('FONTSIZE', (0, 0), (0, -1), 9),
                                ('BOTTOMPADDING', (0, 0), (0, -1), 3),
                                ('TOPPADDING', (0, 0), (0, -1), 3),
                                ('LEFTPADDING', (0, 0), (0, -1), 10),
                                ('RIGHTPADDING', (0, 0), (0, -1), 10),
                                ('BOX', (0, 0), (0, -1), 1, colors.HexColor('#444444')),
                                ('ROUNDEDCORNERS', [6, 6, 6, 6]),
                            ]))
                            # Permitir división entre páginas
                            code_table._splitByRow = True
                            result_elements.append(code_table)
        else:
            # No hay bloques de código, procesar el texto completo
            process_text_part(text, style_to_use, result_elements)
        
        return result_elements
    
    # Función auxiliar para procesar partes de texto (incluidas listas)
    def process_text_part(text, style_to_use, result_elements):
        # Procesar enlaces
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" color="%s"><u>\1</u></a>' % accent_color.hexval(), text)
        
        # Procesar formato de markdown básico
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Negrita
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Cursiva
        text = re.sub(r'`(.*?)`', r'<font face="Courier" color="#F0F0F0" backColor="%s" borderPadding="2">\1</font>' % code_bg_color.hexval(), text)  # Código inline
        
        # Dividir el texto en párrafos
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # Verificar si el párrafo es una lista numerada
            if re.match(r'\s*\d+\.\s', paragraph):
                lines = paragraph.split('\n')
                in_list = False
                list_items = []
                
                for line in lines:
                    # Comprobar si la línea es un elemento de lista numerada
                    if re.match(r'\s*\d+\.\s', line):
                        in_list = True
                        # Extraer el número y el contenido
                        match = re.match(r'\s*(\d+)\.\ (.*)', line)
                        if match:
                            num, content = match.groups()
                            # Formatear como elemento de lista numerada con el mismo número
                            formatted_line = f"{num}. {content}"
                            list_items.append(formatted_line)
                    else:
                        # Si no es un elemento de lista pero estamos en una lista,
                        # lo tratamos como continuación del último elemento
                        if in_list and list_items:
                            list_items[-1] += "<br/>" + line
                        else:
                            # No es parte de una lista, lo procesamos como párrafo normal
                            if line.strip():
                                result_elements.append(Paragraph(line, style_to_use))
                
                # Si encontramos elementos de lista, los añadimos
                if list_items:
                    for item in list_items:
                        # Crear un estilo para los elementos de lista con indentación adecuada
                        list_style = ParagraphStyle(
                            'ListItemStyle',
                            parent=style_to_use,
                            leftIndent=20,
                            firstLineIndent=0
                        )
                        result_elements.append(Paragraph(item, list_style))
            
            # Verificar si el párrafo es una lista con viñetas
            elif re.match(r'\s*[\*\-\+]\s', paragraph):
                lines = paragraph.split('\n')
                list_items = []
                
                for line in lines:
                    # Comprobar si la línea es un elemento de lista con viñetas
                    if re.match(r'\s*[\*\-\+]\s', line):
                        # Extraer el contenido y formatearlo como viñeta
                        match = re.match(r'\s*[\*\-\+]\ (.*)', line)
                        if match:
                            content = match.group(1)
                            formatted_line = f"• {content}"
                            list_items.append(formatted_line)
                    else:
                        # Si no es un elemento de lista pero estamos procesando una lista,
                        # lo tratamos como continuación del último elemento
                        if list_items:
                            list_items[-1] += "<br/>" + line
                        else:
                            # No es parte de una lista, lo procesamos como párrafo normal
                            if line.strip():
                                result_elements.append(Paragraph(line, style_to_use))
                
                # Si encontramos elementos de lista, los añadimos
                if list_items:
                    for item in list_items:
                        # Crear un estilo para los elementos de lista con indentación adecuada
                        list_style = ParagraphStyle(
                            'ListItemStyle',
                            parent=style_to_use,
                            leftIndent=20,
                            firstLineIndent=0
                        )
                        result_elements.append(Paragraph(item, list_style))
            else:
                # No es una lista, lo procesamos como párrafo normal
                if paragraph.strip():
                    result_elements.append(Paragraph(paragraph, style_to_use))

    # Procesar cada mensaje y agregarlos al PDF
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        timestamp = msg.get('timestamp', '')
        
        # Formatear fecha si existe
        time_str = ""
        if timestamp:
            try:
                msg_time = datetime.fromisoformat(timestamp)
                time_str = msg_time.strftime("%d/%m/%Y %H:%M:%S")
            except:
                time_str = timestamp
                
        # Procesar según el rol del mensaje
        if role == 'user':
            # Crear una tabla con fondo azul para el mensaje del usuario (similar a la UI)
            # Añadir el conteo de tokens para el mensaje del usuario
            token_count = msg.get('tokenCount', 0)
            
            # Intentar otras posibles propiedades si tokenCount no está disponible
            if token_count == 0:
                token_count = msg.get('tokens', 0)
            if token_count == 0:
                token_count = msg.get('token_count', 0)
            if token_count == 0 and content:
                # Estimar tokens basados en el contenido si no hay datos de tokens
                token_count = estimate_tokens(content)
                # Guardar el valor calculado en el mensaje para futuras referencias
                msg['tokenCount'] = token_count
                print(f"Estimating tokens for user message: {token_count}")
            
            # Imprimir información para depuración
            print(f"User message token count: {token_count}, from data: {msg.get('tokenCount')}")
            
            header = f"User • {time_str}"
            if token_count > 0:
                header = f"User • {time_str} • {token_count} Tokens"
            
            # Imprimir el header final para verificar
            print(f"Final user message header: '{header}'")
            
            header_para = Paragraph(header, user_style)
            
            # Procesar el contenido del mensaje
            content_elements = process_markdown_for_pdf(content, is_user_message=True)
            
            # Crear los elementos para la tabla
            message_elements = [header_para] + content_elements
            
            # Crear una tabla para el mensaje con fondo de color
            data = [[e] for e in message_elements]
            message_table = Table(data, colWidths=[450], repeatRows=1)
            message_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), user_bubble_color),
                ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor('#a0a0a0')),  # Header color
                ('TEXTCOLOR', (0, 1), (0, -1), text_color),  # Content color
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),  # Todo alineado a la derecha, incluido el encabezado
                ('VALIGN', (0, 0), (0, -1), 'TOP'),
                ('INNERGRID', (0, 0), (0, -1), 0, colors.transparent),
                ('BOX', (0, 0), (0, -1), 1, colors.HexColor('#3a3e52')),  # Borde más claro para destacar
                ('BOTTOMPADDING', (0, 0), (0, -1), 8),
                ('TOPPADDING', (0, 0), (0, -1), 8),
                ('LEFTPADDING', (0, 0), (0, -1), 12),
                ('RIGHTPADDING', (0, 0), (0, -1), 12),
                ('ROUNDEDCORNERS', [10, 10, 10, 10]),
            ]))
            # Habilitar división de tablas
            message_table._splitByRow = True
            
            # Alinear a la derecha (igual que en la UI)
            right_aligned = Table([[message_table]], colWidths=[500])
            right_aligned.setStyle(TableStyle([
                ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
                ('LEFTPADDING', (0, 0), (0, 0), 0),
                ('RIGHTPADDING', (0, 0), (0, 0), 0),
                ('BACKGROUND', (0, 0), (0, 0), colors.transparent),
                ('GRID', (0, 0), (0, 0), 0, colors.transparent),
            ]))
            # Permitir división entre páginas
            right_aligned._splitByRow = True
            
            elements.append(right_aligned)
            
        elif role == 'assistant':
            # Crear una tabla con fondo gris oscuro para el mensaje del asistente
            model = msg.get('model', 'AI')
            provider = msg.get('provider', '')
            response_time = msg.get('responseTimeMs', 0)
            response_time_str = f"{response_time/1000:.2f}s" if response_time else ""
            
            # Incluir más contexto: modelo, proveedor, tiempo de respuesta
            context_info = f"Context: ON • {model}" if settings.get("maintain_context", True) else f"Context: OFF • {model}"
            
            # Formar el encabezado con toda la información de contexto
            header = f"Assistant • {time_str} • Response time {response_time_str} • {context_info}"
            header_para = Paragraph(header, assistant_style)
            
            # Procesar el contenido del mensaje
            content_elements = process_markdown_for_pdf(content)
            
            # Añadir la línea de estadísticas si tenemos los datos necesarios
            token_count = msg.get('tokenCount', 0)
            total_elapsed_time = msg.get('totalElapsedTime', 0)
            time_per_token = msg.get('timePerToken', 0)
            tokens_per_second = msg.get('tokensPerSecond', 0)
            words_per_second = msg.get('wordsPerSecond', 0)
            word_count = msg.get('wordCount', 0)
            
            if token_count > 0 and total_elapsed_time > 0:
                # Formatear tiempo en formato mm:ss.dd
                format_time = lambda ms: f"{int(ms/60000):02d}:{int((ms/1000)%60):02d}.{int((ms%1000)/10):02d}" if ms else "00:00.00"
                total_time_formatted = format_time(total_elapsed_time)
                time_per_token_formatted = format_time(time_per_token)
                
                # Formatear tokens por segundo y palabras por segundo con 2 decimales
                tokens_per_second_formatted = f"{tokens_per_second:.2f}" if tokens_per_second else "0.00"
                words_per_second_formatted = f"{words_per_second:.2f}" if words_per_second else "0.00"
                
                # Crear la línea de estadísticas
                stats_line = f"Total Response Time {total_time_formatted} • # of Tokens {token_count} • Time per Token {time_per_token_formatted} • Tokens/sec {tokens_per_second_formatted} • Words/sec {words_per_second_formatted} ({word_count} words)"
                
                # Crear un estilo para la línea de estadísticas
                stats_style = ParagraphStyle(
                    'StatsStyle',
                    parent=assistant_style,
                    fontSize=8,
                    textColor=colors.HexColor('#a0a0a0'),
                    alignment=0
                )
                
                # Añadir la línea de estadísticas después del contenido
                content_elements.append(Paragraph(stats_line, stats_style))
            
            # Crear los elementos para la tabla
            message_elements = [header_para] + content_elements
            
            # Crear una tabla para el mensaje con fondo de color
            data = [[e] for e in message_elements]
            message_table = Table(data, colWidths=[450], repeatRows=1)
            message_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), assistant_bubble_color),
                ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor('#a0a0a0')),  # Header color
                ('TEXTCOLOR', (0, 1), (0, -1), text_color),  # Content color
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('VALIGN', (0, 0), (0, -1), 'TOP'),
                ('INNERGRID', (0, 0), (0, -1), 0, colors.transparent),
                ('BOX', (0, 0), (0, -1), 1, colors.HexColor('#3a3e52')),  # Borde más claro para destacar
                ('BOTTOMPADDING', (0, 0), (0, -1), 8),
                ('TOPPADDING', (0, 0), (0, -1), 8),
                ('LEFTPADDING', (0, 0), (0, -1), 12),
                ('RIGHTPADDING', (0, 0), (0, -1), 12),
                ('ROUNDEDCORNERS', [10, 10, 10, 10]),
            ]))
            # Habilitar división de tablas
            message_table._splitByRow = True
            
            # Alinear a la izquierda (igual que en la UI)
            left_aligned = Table([[message_table]], colWidths=[500])
            left_aligned.setStyle(TableStyle([
                ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                ('LEFTPADDING', (0, 0), (0, 0), 0),
                ('RIGHTPADDING', (0, 0), (0, 0), 0),
                ('BACKGROUND', (0, 0), (0, 0), colors.transparent),
                ('GRID', (0, 0), (0, 0), 0, colors.transparent),
            ]))
            # Permitir división entre páginas
            left_aligned._splitByRow = True
            
            elements.append(left_aligned)
            
        else:
            # Mensaje del sistema centrado
            system_message = f"System ({time_str}): {content}"
            elements.append(Paragraph(system_message, system_style))
        
        # Añadir espacio entre mensajes - reducido para eliminar espacios innecesarios
        elements.append(Spacer(1, 8))  # Reducido de 12 a 8
    
    # Construir el PDF con un fondo de color similar a la UI
    # Crear una función para dibujar el fondo
    def add_page_background(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(background_color)
        canvas.rect(0, 0, letter[0], letter[1], fill=True, stroke=False)
        canvas.restoreState()
    
    # Agregar un manejador de excepciones para tratar con mensajes largos
    try:
        # Construir el PDF con el fondo personalizado
        doc.build(elements, onFirstPage=add_page_background, onLaterPages=add_page_background)
    except Exception as e:
        print(f"Error al construir el PDF: {str(e)}")
        # Intentar reconstruir con tablas más simples si hubo un error
        try:
            # Crear un nuevo buffer
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer, 
                pagesize=letter,
                leftMargin=36,
                rightMargin=36,
                topMargin=36,
                bottomMargin=36,
                allowSplitting=1
            )
            
            # Simplificar el contenido para evitar problemas de layout
            simplified_elements = []
            simplified_elements.append(Paragraph(f"Conversation: {chat_title}", title_style))
            simplified_elements.append(Spacer(1, 0.25*inch))
            
            # Agregar mensajes de forma más simple
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                timestamp = msg.get('timestamp', '')
                
                # Formatear fecha
                time_str = ""
                if timestamp:
                    try:
                        msg_time = datetime.fromisoformat(timestamp)
                        time_str = msg_time.strftime("%d/%m/%Y %H:%M:%S")
                    except:
                        time_str = timestamp
                
                if role == 'user':
                    # Añadir el conteo de tokens para el mensaje del usuario
                    token_count = msg.get('tokenCount', 0)
                    
                    # Intentar otras posibles propiedades si tokenCount no está disponible
                    if token_count == 0:
                        token_count = msg.get('tokens', 0)
                    if token_count == 0:
                        token_count = msg.get('token_count', 0)
                    if token_count == 0 and content:
                        # Estimar tokens basados en el contenido si no hay datos de tokens
                        token_count = estimate_tokens(content)
                        # Guardar el valor calculado en el mensaje para futuras referencias
                        msg['tokenCount'] = token_count
                    
                    # Imprimir información para depuración
                    print(f"User message token count (simplified): {token_count}, from data: {msg.get('tokenCount')}")
                    
                    header = f"User • {time_str}"
                    if token_count > 0:
                        header = f"User • {time_str} • {token_count} Tokens"
                    
                    # Imprimir el header final para verificar
                    print(f"Final user message header (simplified): '{header}'")
                    
                    simplified_elements.append(Paragraph(header, user_style))
                    simplified_elements.append(Paragraph(content, user_content_style))
                    simplified_elements.append(Spacer(1, 8))
                elif role == 'assistant':
                    model = msg.get('model', 'AI')
                    response_time = msg.get('responseTimeMs', 0)
                    response_time_str = f"{response_time/1000:.2f}s" if response_time else ""
                    context_info = f"Context: ON • {model}" if settings.get("maintain_context", True) else f"Context: OFF • {model}"
                    header = f"Assistant • {time_str} • Response time {response_time_str} • {context_info}"
                    simplified_elements.append(Paragraph(header, assistant_style))
                    simplified_elements.append(Paragraph(content, assistant_content_style))
                    
                    # Añadir la línea de estadísticas también en el formato simplificado
                    token_count = msg.get('tokenCount', 0)
                    total_elapsed_time = msg.get('totalElapsedTime', 0)
                    time_per_token = msg.get('timePerToken', 0)
                    tokens_per_second = msg.get('tokensPerSecond', 0)
                    words_per_second = msg.get('wordsPerSecond', 0)
                    word_count = msg.get('wordCount', 0)
                    
                    if token_count > 0 and total_elapsed_time > 0:
                        # Formatear tiempo en formato mm:ss.dd
                        format_time = lambda ms: f"{int(ms/60000):02d}:{int((ms/1000)%60):02d}.{int((ms%1000)/10):02d}" if ms else "00:00.00"
                        total_time_formatted = format_time(total_elapsed_time)
                        time_per_token_formatted = format_time(time_per_token)
                        
                        # Formatear tokens por segundo y palabras por segundo con 2 decimales
                        tokens_per_second_formatted = f"{tokens_per_second:.2f}" if tokens_per_second else "0.00"
                        words_per_second_formatted = f"{words_per_second:.2f}" if words_per_second else "0.00"
                        
                        # Crear la línea de estadísticas
                        stats_line = f"Total Response Time {total_time_formatted} • # of Tokens {token_count} • Time per Token {time_per_token_formatted} • Tokens/sec {tokens_per_second_formatted} • Words/sec {words_per_second_formatted} ({word_count} words)"
                        
                        # Crear un estilo para la línea de estadísticas
                        stats_style = ParagraphStyle(
                            'StatsStyle',
                            parent=assistant_style,
                            fontSize=8,
                            textColor=colors.HexColor('#a0a0a0'),
                            alignment=0
                        )
                        
                        simplified_elements.append(Paragraph(stats_line, stats_style))
                    
                    simplified_elements.append(Spacer(1, 8))
                else:
                    system_message = f"System ({time_str}): {content}"
                    simplified_elements.append(Paragraph(system_message, system_style))
                    simplified_elements.append(Spacer(1, 8))
            
            # Reconstruir con contenido simplificado
            doc.build(simplified_elements, onFirstPage=add_page_background, onLaterPages=add_page_background)
            
            # Mover el cursor al inicio del buffer
            buffer.seek(0)
        except Exception as e2:
            print(f"Error al construir PDF simplificado: {str(e2)}")
            # Si todo falla, devolver un mensaje de error
            return jsonify({"error": f"No se pudo generar el PDF: {str(e2)}"}), 500
    
    # Mover el cursor al inicio del buffer
    buffer.seek(0)
    
    # Generar un nombre de archivo basado en el título del chat
    safe_title = re.sub(r'[^\w\s-]', '', chat_title)
    safe_title = re.sub(r'[-\s]+', '-', safe_title).strip('-_')
    filename = f"chat_{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Devolver el archivo PDF como respuesta
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )

@app.route('/api/chat/<chat_id>/update-messages', methods=['POST'])
def update_chat_messages(chat_id):
    """
    Updates messages in a chat with enhanced metrics (token count, response times, etc.)
    This endpoint allows the frontend to persistently store calculated metrics
    """
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    # Get the updated messages from the request
    updated_messages = request.json
    
    if not updated_messages or not isinstance(updated_messages, list):
        return jsonify({"error": "Invalid message data format"}), 400
    
    # Update the messages in memory
    chats[chat_id]["messages"] = updated_messages
    
    # Save the updated messages to the chat file
    chat_file = os.path.join(CHAT_DIR, f"{chat_id}.json")
    try:
        with open(chat_file, 'w') as f:
            json.dump(updated_messages, f)
        print(f"Enhanced chat messages saved to file: {chat_file}")
        
        # Update the last_message timestamp
        if updated_messages:
            chats[chat_id]["last_message"] = updated_messages[-1].get("timestamp", datetime.now().isoformat())
            save_chat_metadata(chat_id)
            
        return jsonify({"status": "success", "message": "Messages updated with enhanced metrics"}), 200
    except Exception as e:
        print(f"Error saving enhanced chat messages: {str(e)}")
        return jsonify({"error": f"Failed to save messages: {str(e)}"}), 500

# Load projects from files
def load_projects_from_files():
    """Loads all projects from JSON files stored in PROJECT_DIR."""
    print("Loading projects from files...")
    try:
        loaded_projects = {}
        valid_projects = set()
        
        # Get all project files
        all_files = os.listdir(PROJECT_DIR)
        project_files = [f for f in all_files if f.endswith('.json')]
        
        print(f"Found {len(project_files)} project files in directory")
        
        # First pass: validate projects and check which are valid
        for filename in project_files:
            project_id = filename.replace('.json', '')
            project_file_path = os.path.join(PROJECT_DIR, filename)
            
            try:
                # Check if file is empty or corrupted
                if os.path.getsize(project_file_path) == 0:
                    print(f"Skipping empty project file: {filename}")
                    continue
                
                with open(project_file_path, 'r') as f:
                    try:
                        project_data = json.load(f)
                        
                        # Basic validation of project data
                        if not all(key in project_data for key in ["id", "title"]):
                            print(f"Skipping invalid project file (missing required fields): {filename}")
                            continue
                        
                        # Check if project directory exists (if not, project might be invalid)
                        project_title = project_data.get("title", "")
                        external_project_dir = os.path.join('/Users/mbayasp/Library/CloudStorage/OneDrive-office.bayteq.com/Documents/Cursos/Master Ejecutivo en IA/5. Proyecto de Fin de Master/Ejercicio Seguro Medico/Code/programas/chat-app/files', project_title)
                        
                        if not os.path.exists(external_project_dir) and project_title:
                            print(f"Project directory for '{project_title}' does not exist")
                        
                        # This project is valid, add to our valid set
                        valid_projects.add(project_id)
                    except json.JSONDecodeError:
                        print(f"Skipping corrupted project file (invalid JSON): {filename}")
                        continue
            except Exception as e:
                print(f"Error validating project {project_id}: {str(e)}")
        
        # Second pass: load only valid projects
        for filename in project_files:
            project_id = filename.replace('.json', '')
            
            # Skip if not in our valid projects set
            if project_id not in valid_projects:
                continue
                
            project_file_path = os.path.join(PROJECT_DIR, filename)
            
            try:
                with open(project_file_path, 'r') as f:
                    project_data = json.load(f)
                    
                # Ensure project data has all required fields
                loaded_projects[project_id] = {
                    "id": project_data.get("id", project_id),
                    "title": project_data.get("title", "Untitled Project"),
                    "created_at": project_data.get("created_at", datetime.now().isoformat()),
                    "last_updated": project_data.get("last_updated", datetime.now().isoformat()),
                    "system_prompt": project_data.get("system_prompt", ""),
                    "files": project_data.get("files", [])
                }
                
                print(f"Loaded project {project_id} with {len(project_data.get('files', []))} files")
                
            except Exception as e:
                print(f"Error loading project {project_id}: {str(e)}")
        
        print(f"Successfully loaded {len(loaded_projects)} valid projects")
        return loaded_projects
        
    except Exception as e:
        print(f"Error loading projects: {str(e)}")
        return {}

# Save project to file
def save_project(project_id):
    """Saves a project to its JSON file."""
    if project_id not in projects:
        return False
    
    project_file = os.path.join(PROJECT_DIR, f"{project_id}.json")
    try:
        with open(project_file, 'w') as f:
            json.dump(projects[project_id], f)
        return True
    except Exception as e:
        print(f"Error saving project {project_id}: {str(e)}")
        return False

# Load projects at startup
projects = load_projects_from_files()
print(f"Loaded {len(projects)} projects from files")

# PROJECT API ENDPOINTS

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """Get all projects."""
    try:
        # Get active projects that actually have a corresponding directory
        valid_projects = []
        
        # Create a list of projects with existing directories
        files_dir = '/Users/mbayasp/Library/CloudStorage/OneDrive-office.bayteq.com/Documents/Cursos/Master Ejecutivo en IA/5. Proyecto de Fin de Master/Ejercicio Seguro Medico/Code/programas/chat-app/files'
        existing_project_dirs = [d for d in os.listdir(files_dir) if os.path.isdir(os.path.join(files_dir, d))]
        
        # Filter projects to only include those with existing directories
        for project_id, project in projects.items():
            project_title = project.get('title', '')
            
            # Keep project if directory exists or if it's a new project without a title
            if project_title in existing_project_dirs or not project_title:
                valid_projects.append(project)
        
        # Sort projects by last_updated (most recent first)
        sorted_projects = sorted(
            valid_projects,
            key=lambda p: p.get('last_updated', p.get('created_at', '')),
            reverse=True
        )
        
        # Limit to most recent 5 projects to prevent UI overflow
        limited_projects = sorted_projects[:5]
        
        print(f"Returning {len(limited_projects)} projects out of {len(sorted_projects)} valid projects")
        
        return jsonify(limited_projects), 200
    except Exception as e:
        print(f"Error in get_projects: {str(e)}")
        return jsonify({"error": f"Failed to get projects: {str(e)}"}), 500

@app.route('/api/projects', methods=['POST'])
def create_project():
    """Create a new project."""
    data = request.json
    title = data.get('title', 'New Project')
    system_prompt = data.get('system_prompt', '')
    
    # Validar que el título del proyecto cumpla con los requisitos
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', title):
        return jsonify({"error": "El nombre del proyecto solo puede contener letras, números, guiones bajos (_) y guiones medios (-). No se permiten espacios ni caracteres especiales."}), 400
    
    project_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    project = {
        "id": project_id,
        "title": title,
        "created_at": timestamp,
        "last_updated": timestamp,
        "system_prompt": system_prompt,
        "files": []
    }
    
    projects[project_id] = project
    
    # Create a physical directory for the project
    project_dir_path = os.path.join('/Users/mbayasp/Library/CloudStorage/OneDrive-office.bayteq.com/Documents/Cursos/Master Ejecutivo en IA/5. Proyecto de Fin de Master/Ejercicio Seguro Medico/Code/programas/chat-app/files', title)
    os.makedirs(project_dir_path, exist_ok=True)
    
    # Save project metadata
    project_file_path = os.path.join(project_dir_path, 'metadata.json')
    with open(project_file_path, 'w') as f:
        json.dump(project, f)
    
    # Save project to file
    save_project(project_id)
    
    return jsonify(project), 201

@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get a specific project."""
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    
    return jsonify(projects[project_id]), 200

@app.route('/api/projects/<project_id>', methods=['PUT'])
def update_project(project_id):
    """Update a project."""
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    
    data = request.json
    
    # Update fields that are provided
    if 'title' in data:
        # Validar que el título del proyecto cumpla con los requisitos
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', data['title']):
            return jsonify({"error": "El nombre del proyecto solo puede contener letras, números, guiones bajos (_) y guiones medios (-). No se permiten espacios ni caracteres especiales."}), 400
        
        projects[project_id]['title'] = data['title']
    
    if 'system_prompt' in data:
        projects[project_id]['system_prompt'] = data['system_prompt']
    
    # Update last_updated timestamp
    projects[project_id]['last_updated'] = datetime.now().isoformat()
    
    # Save project to file
    save_project(project_id)
    
    return jsonify(projects[project_id]), 200

@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project."""
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    
    # Get project files to delete
    project_files = projects[project_id].get('files', [])
    project_title = projects[project_id].get('title', 'Untitled Project')
    
    # Delete project files from disk
    for file in project_files:
        file_path = os.path.join(UPLOAD_DIR, project_id, file.get('name', ''))
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")
    
    # Delete project directory if it exists
    project_dir = os.path.join(UPLOAD_DIR, project_id)
    if os.path.exists(project_dir) and os.path.isdir(project_dir):
        try:
            os.rmdir(project_dir)
        except Exception as e:
            print(f"Error deleting project directory {project_dir}: {str(e)}")
    
    # Delete the external project directory
    external_project_dir = os.path.join('/Users/mbayasp/Library/CloudStorage/OneDrive-office.bayteq.com/Documents/Cursos/Master Ejecutivo en IA/5. Proyecto de Fin de Master/Ejercicio Seguro Medico/Code/programas/chat-app/files', project_title)
    if os.path.exists(external_project_dir) and os.path.isdir(external_project_dir):
        try:
            shutil.rmtree(external_project_dir)
            print(f"Deleted external project directory: {external_project_dir}")
        except Exception as e:
            print(f"Error deleting external project directory {external_project_dir}: {str(e)}")
    
    # Delete project file
    project_file = os.path.join(PROJECT_DIR, f"{project_id}.json")
    if os.path.exists(project_file):
        try:
            os.remove(project_file)
        except Exception as e:
            print(f"Error deleting project file {project_file}: {str(e)}")
    
    # Remove from memory
    del projects[project_id]
    
    return jsonify({"status": "success"}), 200

@app.route('/api/projects/<project_id>/system-prompt', methods=['PUT'])
def update_system_prompt(project_id):
    """Update the system prompt for a project."""
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    
    data = request.json
    
    if 'prompt' not in data:
        return jsonify({"error": "System prompt is required"}), 400
    
    projects[project_id]['system_prompt'] = data['prompt']
    projects[project_id]['last_updated'] = datetime.now().isoformat()
    
    # Save project to file
    save_project(project_id)
    
    return jsonify(projects[project_id]), 200

@app.route('/api/projects/upload', methods=['POST'])
def upload_project_file():
    """Upload a file to a project."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    project_id = request.form.get('project_id')
    
    if not project_id:
        return jsonify({"status": "error", "message": "Project ID is required"}), 400
    
    if project_id not in projects:
        return jsonify({"status": "error", "message": "Project not found"}), 404
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    # Create a directory for this project's uploads if it doesn't exist
    project_title = projects[project_id].get('title', 'Untitled Project')
    
    # Standard directory for in-app file storage
    project_upload_dir = os.path.join(UPLOAD_DIR, project_id)
    os.makedirs(project_upload_dir, exist_ok=True)
    
    # Additional directory with project name in the specified location
    project_files_dir = os.path.join('/Users/mbayasp/Library/CloudStorage/OneDrive-office.bayteq.com/Documents/Cursos/Master Ejecutivo en IA/5. Proyecto de Fin de Master/Ejercicio Seguro Medico/Code/programas/chat-app/files', project_title)
    os.makedirs(project_files_dir, exist_ok=True)
    
    # Generate a unique filename
    unique_id = str(uuid.uuid4())
    filename = unique_id + "_" + file.filename
    
    # Save the file to both locations
    file_path = os.path.join(project_upload_dir, filename)
    external_file_path = os.path.join(project_files_dir, file.filename)
    
    # Save the file to the app's upload directory (with unique ID)
    file.save(file_path)
    
    # Create a copy in the external project directory (with original name)
    with open(file_path, 'rb') as src_file:
        with open(external_file_path, 'wb') as dst_file:
            dst_file.write(src_file.read())
    
    # Create file record
    file_record = {
        "id": unique_id,
        "name": filename,
        "original_name": file.filename,
        "size": os.path.getsize(file_path),
        "uploaded_at": datetime.now().isoformat(),
        "url": f"/api/projects/{project_id}/files/{filename}"
    }
    
    # Add file to project
    projects[project_id]['files'].append(file_record)
    projects[project_id]['last_updated'] = datetime.now().isoformat()
    
    # Save project to file
    save_project(project_id)
    
    return jsonify({
        "status": "success",
        "file": file_record,
        "message": f"File {file.filename} uploaded successfully"
    }), 200

@app.route('/api/projects/<project_id>/files/<file_id>', methods=['DELETE'])
def delete_project_file(project_id, file_id):
    """Delete a file from a project."""
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    
    # Find the file record
    file_record = None
    file_index = -1
    for i, file in enumerate(projects[project_id]['files']):
        if file.get('id') == file_id:
            file_record = file
            file_index = i
            break
    
    if not file_record:
        return jsonify({"error": "File not found"}), 404
    
    # Get file paths
    app_file_path = os.path.join(UPLOAD_DIR, project_id, file_record.get('name', ''))
    project_title = projects[project_id].get('title', 'Untitled Project')
    external_file_path = os.path.join('/Users/mbayasp/Library/CloudStorage/OneDrive-office.bayteq.com/Documents/Cursos/Master Ejecutivo en IA/5. Proyecto de Fin de Master/Ejercicio Seguro Medico/Code/programas/chat-app/files', project_title, file_record.get('original_name', ''))
    
    # Delete the file from both locations
    if os.path.exists(app_file_path):
        try:
            os.remove(app_file_path)
            print(f"Deleted file from app directory: {app_file_path}")
        except Exception as e:
            print(f"Error deleting file from app directory {app_file_path}: {str(e)}")
    
    if os.path.exists(external_file_path):
        try:
            os.remove(external_file_path)
            print(f"Deleted file from external directory: {external_file_path}")
        except Exception as e:
            print(f"Error deleting file from external directory {external_file_path}: {str(e)}")
    
    # Remove file from project
    if file_index >= 0:
        projects[project_id]['files'].pop(file_index)
        projects[project_id]['last_updated'] = datetime.now().isoformat()
    
    # Save project to file
    save_project(project_id)
    
    return jsonify({"status": "success"}), 200

@app.route('/api/projects/<project_id>/files/<file_name>', methods=['GET'])
def get_project_file(project_id, file_name):
    """Download a file from a project."""
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    
    # Check if file exists in project
    file_exists = False
    for file in projects[project_id]['files']:
        if file.get('name') == file_name:
            file_exists = True
            break
    
    if not file_exists:
        return jsonify({"error": "File not found in project"}), 404
    
    # Get file path
    file_path = os.path.join(UPLOAD_DIR, project_id, file_name)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found on disk"}), 404
    
    # Get original filename for better download experience
    original_filename = None
    for file in projects[project_id]['files']:
        if file.get('name') == file_name:
            original_filename = file.get('original_name', file_name)
            break
    
    return send_file(file_path, as_attachment=True, download_name=original_filename)

@app.route('/api/projects/<project_id>/process-rag', methods=['POST'])
def process_project_files_for_rag(project_id):
    """Process project files for RAG (Retrieval Augmented Generation)."""
    print(f"RAG processing requested for project_id: {project_id}")
    
    if project_id not in projects:
        print(f"Error: Project with ID {project_id} not found")
        return jsonify({"error": "Project not found"}), 404
    
    # Get chunking parameters from request or use defaults
    data = request.json or {}
    chunk_size = data.get('chunk_size', 1200)
    chunk_overlap = data.get('chunk_overlap', 200)
    
    # Get project info
    project = projects[project_id]
    project_title = project.get('title', 'Untitled Project')
    project_files = project.get('files', [])
    
    print(f"Processing project: {project_title} with {len(project_files)} files")
    print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    
    if not project_files:
        print("Error: No files to process in project")
        return jsonify({"error": "No files to process"}), 400
    
    try:
        # Define the FAISS directory path - both for the global DB location and the specific project
        # Base path for all DB files 
        base_db_path = os.path.join(os.path.dirname(__file__), '..', 'db')
        os.makedirs(base_db_path, exist_ok=True)
        
        # Project-specific DB directory using project name instead of ID
        project_db_dir = os.path.join(base_db_path, project_title)
        
        print(f"Using FAISS directory: {project_db_dir}")
        
        # Clean up the DB directory - completely remove it if it exists
        if os.path.exists(project_db_dir):
            print(f"Removing existing FAISS directory: {project_db_dir}")
            try:
                shutil.rmtree(project_db_dir)
                print(f"Successfully removed existing FAISS directory")
            except Exception as e:
                print(f"Error removing directory: {str(e)}")
                # Continue anyway - we'll try to create a fresh directory
        
        # Create the directory
        try:
            print(f"Creating fresh FAISS directory: {project_db_dir}")
            os.makedirs(project_db_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"Error creating FAISS directory: {str(e)}"
            }), 500
        
        # Initialize the text splitter for chunking
        print("Initializing text splitter for chunking")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],  # Natural text separators
            length_function=len
        )
        
        # Initialize OpenAI embeddings
        print("Initializing OpenAI embeddings")
        try:
            # Usar la versión importada de OpenAIEmbeddings (ya sea langchain_openai o langchain.embeddings.openai)
            embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
            print("OpenAI embeddings initialized successfully")
        except Exception as e:
            print(f"Error initializing OpenAI embeddings: {str(e)}")
            return jsonify({"status": "error", "message": f"Error initializing OpenAI API: {str(e)}"}), 500
        
        # Process each file
        processed_files = []
        skipped_files = []
        all_chunks = []
        
        print(f"Beginning to process {len(project_files)} files")
        
        for file in project_files:
            try:
                file_name = file.get('name', '')
                file_path = os.path.join(UPLOAD_DIR, project_id, file_name)
                original_name = file.get('original_name', file_name)
                
                print(f"Processing file: {original_name} (internal name: {file_name})")
                print(f"File path: {file_path}")
                
                # Check if file exists
                if not os.path.exists(file_path):
                    print(f"Error: File not found at path: {file_path}")
                    skipped_files.append({
                        "name": original_name,
                        "reason": "File not found on disk"
                    })
                    continue
                
                # Process file based on extension
                if original_name.lower().endswith('.pdf'):
                    print(f"Loading PDF: {original_name}")
                    try:
                        # Load the PDF
                        loader = PyPDFLoader(file_path)
                        documents = loader.load()
                        print(f"Loaded {len(documents)} pages from PDF")
                    except Exception as pdf_error:
                        print(f"Error loading PDF: {str(pdf_error)}")
                        skipped_files.append({
                            "name": original_name,
                            "reason": f"Error loading PDF: {str(pdf_error)}"
                        })
                        continue
                
                elif original_name.lower().endswith('.json'):
                    print(f"Loading JSON: {original_name}")
                    try:
                        # Load the JSON file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json_content = json.load(f)
                        
                        # Convert JSON to formatted text
                        formatted_text = ""
                        
                        # Check if it's a list of objects (common case)
                        if isinstance(json_content, list):
                            for item in json_content:
                                # Process each item in the list
                                if isinstance(item, dict):
                                    # Create a formatted text entry for each object
                                    item_text = []
                                    for key, value in item.items():
                                        if key.lower() == "nombre":
                                            # Name/title gets special treatment - no emoji
                                            item_text.insert(0, f"{value}")
                                        elif key.lower() == "dirección" or key.lower() == "direccion":
                                            item_text.append(f"📍 {key}: {value}")
                                        elif key.lower() == "teléfono" or key.lower() == "telefono":
                                            item_text.append(f"📞 {key}: {value}")
                                        else:
                                            item_text.append(f"{key}: {value}")
                                    
                                    # Join item text with double line breaks
                                    formatted_text += "\n".join(item_text) + "\n\n"
                        elif isinstance(json_content, dict):
                            # Process single dictionary
                            for key, value in json_content.items():
                                formatted_text += f"{key}: {value}\n"
                        else:
                            # Fallback for other formats - use regular JSON string
                            formatted_text = json.dumps(json_content, indent=2, ensure_ascii=False)
                            
                        print(f"Transformed JSON to formatted text: {len(formatted_text)} characters")
                        
                        # Use special chunking parameters for JSON files
                        json_text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=600,
                            chunk_overlap=20,
                            separators=["\n\n", "\n", ".", " "],
                            length_function=len
                        )
                        
                        # Create a document with the formatted text content
                        try:
                            # Intentar usar la versión más reciente
                            from langchain_core.documents import Document
                        except ImportError:
                            # Fallback a la versión anterior
                            from langchain.schema import Document
                        document = Document(
                            page_content=formatted_text,
                            metadata={"source": original_name, "page": 1, "file_type": "json"}
                        )
                        
                        # Split into chunks using the JSON-specific splitter
                        print(f"Splitting JSON document into chunks with custom parameters (size=600, overlap=50)")
                        chunks = json_text_splitter.split_documents([document])
                        print(f"Created {len(chunks)} chunks from JSON document")
                        
                        # Add metadata to identify source file for each chunk
                        for chunk in chunks:
                            chunk.metadata['source'] = original_name
                            chunk.metadata['project'] = project_title
                            chunk.metadata['file_type'] = 'json'
                        
                        # Add to collection of all chunks
                        all_chunks.extend(chunks)
                        
                        processed_files.append({
                            "name": original_name,
                            "chunks": len(chunks)
                        })
                        print(f"Successfully processed JSON file {original_name}")
                        
                        # Skip the rest of the processing for this file
                        continue
                        
                    except Exception as json_error:
                        print(f"Error loading JSON: {str(json_error)}")
                        skipped_files.append({
                            "name": original_name,
                            "reason": f"Error loading JSON: {str(json_error)}"
                        })
                        continue
                
                elif original_name.lower().endswith(('.txt', '.md')):
                    print(f"Loading text file: {original_name}")
                    text_content = None
                    
                    # Lista de codificaciones a probar
                    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                    
                    # Probar diferentes codificaciones
                    for encoding in encodings_to_try:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                text_content = f.read()
                            print(f"Successfully loaded text file with {encoding} encoding, {len(text_content)} characters")
                            break  # Si se lee correctamente, salir del bucle
                        except UnicodeDecodeError as e:
                            print(f"Failed to decode with {encoding}: {str(e)}")
                    
                    # Si no se pudo leer con ninguna codificación, usar opción binaria
                    if text_content is None:
                        print(f"Failed to load text file with standard encodings, trying binary mode")
                        try:
                            with open(file_path, 'rb') as f:
                                binary_content = f.read()
                                text_content = binary_content.decode('utf-8', errors='replace')
                            print(f"Loaded text file in binary mode with replacement, {len(text_content)} characters")
                        except Exception as binary_error:
                            print(f"Error loading text file in binary mode: {str(binary_error)}")
                            skipped_files.append({
                                "name": original_name,
                                "reason": f"Could not decode file with any encoding"
                            })
                            continue
                            
                    try:
                        # Usar un separador específico para archivos de texto
                        txt_text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=600,
                            chunk_overlap=50,
                            separators=["\n\n", "\n", ".", " "],
                            length_function=len
                        )
                        
                        # Create a document with the text content
                        try:
                            # Intentar usar la versión más reciente
                            from langchain_core.documents import Document
                        except ImportError:
                            # Fallback a la versión anterior
                            from langchain.schema import Document
                        
                        document = Document(
                            page_content=text_content,
                            metadata={"source": original_name, "page": 1, "file_type": "txt"}
                        )
                        
                        # Split document into chunks using text-specific splitter
                        chunks = txt_text_splitter.split_documents([document])
                        print(f"Created {len(chunks)} chunks from text document")
                        
                        # Add metadata to identify source file for each chunk
                        for chunk in chunks:
                            chunk.metadata['source'] = original_name
                            chunk.metadata['project'] = project_title
                            chunk.metadata['file_type'] = 'txt'
                        
                        # Add to collection of all chunks
                        all_chunks.extend(chunks)
                        
                        processed_files.append({
                            "name": original_name,
                            "chunks": len(chunks)
                        })
                        print(f"Successfully processed text file {original_name}")
                    
                    except Exception as text_error:
                        print(f"Error processing text file: {str(text_error)}")
                        import traceback
                        traceback.print_exc()
                        skipped_files.append({
                            "name": original_name,
                            "reason": f"Error processing text file: {str(text_error)}"
                        })
                    
                    # Skip the rest of the processing for this file
                    continue
                
                else:
                    # Skip unsupported file types
                    print(f"Skipping unsupported file type: {original_name}")
                    file_extension = os.path.splitext(original_name)[1].lower()
                    skipped_files.append({
                        "name": original_name,
                        "reason": f"Unsupported file type: {file_extension} (supported: .pdf, .json, .txt, .md)"
                    })
                    continue
                
                # Split into chunks
                print(f"Splitting document into chunks")
                
                # Use file-specific chunking if needed
                if original_name.lower().endswith('.json'):
                    # For JSON files, chunks were already created using custom parameters
                    print(f"Using pre-generated chunks for JSON file")
                    # No need to regenerate chunks here as they were already created
                else:
                    # For other file types, use the standard chunker
                    chunks = text_splitter.split_documents(documents)
                    print(f"Created {len(chunks)} chunks from document")
                
                # Add metadata to identify source file
                for chunk in chunks:
                    chunk.metadata['source'] = original_name
                    chunk.metadata['project'] = project_title
                
                # Add to collection of all chunks
                all_chunks.extend(chunks)
                
                processed_files.append({
                    "name": original_name,
                    "chunks": len(chunks)
                })
                print(f"Successfully processed {original_name}")
                
            except Exception as e:
                print(f"Error processing file {original_name}: {str(e)}")
                skipped_files.append({
                    "name": original_name,
                    "reason": str(e)
                })
        
        # Create the vector database if we have chunks to process
        if all_chunks:
            print(f"Creating vector database with {len(all_chunks)} chunks")
            try:
                # Define los directorios para FAISS
                base_db_path = os.path.join(os.path.dirname(__file__), '..', 'db')
                os.makedirs(base_db_path, exist_ok=True)
                project_db_dir = os.path.join(base_db_path, project_title)
                
                # Asegurarse de que exista el directorio del proyecto
                os.makedirs(project_db_dir, exist_ok=True)
                
                # Define la ruta específica para el índice FAISS de este proyecto
                faiss_index_path = project_db_dir
                
                print(f"Creating FAISS index with {len(all_chunks)} chunks")
                
                # Crear índice FAISS a partir de documentos
                db = FAISS.from_documents(
                    documents=all_chunks,
                    embedding=embeddings
                )
                
                # Guardar el índice FAISS en el directorio del proyecto
                # Anteriormente estaba guardando en: os.path.join(project_db_dir, f"project_{project_id}")
                # Ahora guardaremos directamente en el directorio del proyecto para simplificar
                # faiss_index_path = project_db_dir
                faiss_index_path = project_db_dir  # Simplificación de la estructura
                
                # Intentar guardar el índice
                try:
                    db.save_local(faiss_index_path)
                    print(f"FAISS index successfully created at {faiss_index_path}")
                except Exception as save_error:
                    print(f"Error guardando índice FAISS: {str(save_error)}")
                    import traceback
                    traceback.print_exc()
                    # Intentar crear el directorio si no existe
                    os.makedirs(faiss_index_path, exist_ok=True)
                    # Intentar guardar de nuevo
                    db.save_local(faiss_index_path)
                    print(f"FAISS index saved on second attempt at {faiss_index_path}")
                
                # Verificar que el índice se ha guardado correctamente
                if os.path.exists(faiss_index_path):
                    print(f"Verificación: FAISS index path existe en: {faiss_index_path}")
                    # Si es un directorio, listar su contenido
                    if os.path.isdir(faiss_index_path):
                        faiss_files = os.listdir(faiss_index_path)
                        print(f"Archivos en el directorio del índice FAISS: {faiss_files}")
                        
                        # Verificar que existan los archivos necesarios
                        if "index.faiss" in faiss_files and "index.pkl" in faiss_files:
                            print("Índice FAISS completo y válido encontrado")
                        else:
                            print(f"ADVERTENCIA: El índice FAISS podría estar incompleto. Archivos encontrados: {faiss_files}")
                    else:
                        print(f"ADVERTENCIA: faiss_index_path existe pero no es un directorio")
                else:
                    print(f"ERROR: No se pudo encontrar el índice FAISS después de guardarlo")
                
                # Store mapping of project_id to project_title for future reference
                mapping_file = os.path.join(base_db_path, "project_mapping.json")
                try:
                    # Load existing mapping if it exists
                    if os.path.exists(mapping_file):
                        with open(mapping_file, 'r') as f:
                            mapping = json.load(f)
                    else:
                        mapping = {}
                    
                    # Update mapping
                    mapping[project_id] = project_title
                    
                    # Save mapping back to file
                    with open(mapping_file, 'w') as f:
                        json.dump(mapping, f, indent=2)
                    
                    print(f"Updated project mapping: {project_id} -> {project_title}")
                except Exception as map_error:
                    print(f"Warning: Could not update project mapping: {str(map_error)}")
                
                return jsonify({
                    "status": "success",
                    "message": f"Generated RAG index with {len(all_chunks)} chunks from {len(processed_files)} files",
                    "processed_files": processed_files,
                    "skipped_files": skipped_files,
                    "total_chunks": len(all_chunks),
                    "db_location": faiss_index_path
                }), 200
            except Exception as e:
                print(f"Error creating FAISS vector database: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    "status": "error",
                    "message": f"Error creating vector database: {str(e)}"
                }), 500
        else:
            print("No chunks were generated from the files")
            return jsonify({
                "status": "warning",
                "message": "No chunks were generated from the files.\nPlease check that you have uploaded supported files (PDF, TXT, JSON, MD).",
                "processed_files": processed_files,
                "skipped_files": skipped_files
            }), 200
            
    except Exception as e:
        print(f"Error processing RAG for project {project_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Failed to process files: {str(e)}"
        }), 500

# Function to process user prompt with RAG
def process_rag_prompt(message):
    """
    Process a user prompt that starts with '**ProjectName**' to implement RAG.
    Returns tuple of (processed_message, context_to_add, project_id)
    """
    print("\n" + "="*50)
    print("PROCESANDO PROMPT RAG")
    print("="*50)
    print(f"Mensaje original recibido: '{message}'")
    
    # Check if message starts with '**'
    if not message.startswith('**'):
        print("El mensaje no comienza con '**', no es un prompt RAG")
        return message, None, None
    
    # Extract project name between ** and **
    second_marker = message.find('**', 2)
    if second_marker == -1:
        # Not properly formatted
        print("Formato incorrecto: no se encontró el segundo '**'")
        return message, None, None
    
    project_name = message[2:second_marker]
    print(f"Nombre del proyecto extraído: '{project_name}'")
    
    # Validate project exists
    project_id = None
    for pid, project in projects.items():
        if project.get('title') == project_name:
            project_id = pid
            break
    
    if not project_id:
        # No matching project found
        print(f"ERROR: No se encontró el proyecto '{project_name}' en la base de datos")
        return message, None, None
    
    print(f"Proyecto encontrado con ID: {project_id}")
    
    # Extract the actual query (remove **ProjectName**)
    actual_query = message[second_marker + 2:].strip()
    if not actual_query:
        # No query after project name
        print("ERROR: No hay consulta después del nombre del proyecto")
        return message, None, None
    
    print(f"Consulta RAG extraída: '{actual_query}'")
    
    # Locate the project's vector database using project name directly
    base_db_path = os.path.join(os.path.dirname(__file__), '..', 'db')
    project_db_dir = os.path.join(base_db_path, project_name)
    faiss_index_path = project_db_dir
    
    print(f"Directorio FAISS index: {faiss_index_path}")
    
    # Verificar que existe el directorio base de db
    if not os.path.exists(base_db_path):
        print(f"ERROR: No existe el directorio base de la base de datos: {base_db_path}")
        return message, None, None
    
    # Verificar que existe el directorio del proyecto
    if not os.path.exists(project_db_dir):
        print(f"ERROR: No existe el directorio del proyecto: {project_db_dir}")
        # Listar lo que hay en el directorio base
        print(f"Contenido del directorio base: {os.listdir(base_db_path)}")
        return message, None, None
    
    # Verificar que existe el directorio/archivo del índice FAISS
    if not os.path.exists(faiss_index_path):
        # Vector database doesn't exist
        print(f"ERROR: No existe el índice FAISS en {faiss_index_path}")
        # Listar lo que hay en el directorio del proyecto
        print(f"Contenido del directorio del proyecto: {os.listdir(project_db_dir)}")
        
        # Intentar ubicaciones alternativas (por si hay cambios en la estructura del directorio)
        print("Intentando encontrar índice FAISS en ubicaciones alternativas...")
        alternative_path = project_db_dir  # Podría ser que el índice se guardó directamente en el directorio del proyecto
        if os.path.exists(alternative_path):
            print(f"Probando ubicación alternativa: {alternative_path}")
            faiss_index_path = alternative_path
            print(f"Usando directorio FAISS alternativo: {faiss_index_path}")
        else:
            return message, None, None
    
    # Verificar que el índice FAISS tenga los archivos necesarios
    faiss_files = os.listdir(faiss_index_path) if os.path.isdir(faiss_index_path) else []
    print(f"Archivos en el índice FAISS: {faiss_files}")
    
    # Verificación adicional para validar la integridad del índice
    faiss_index_file = os.path.join(faiss_index_path, "index.faiss") 
    if os.path.isdir(faiss_index_path) and not os.path.exists(faiss_index_file):
        print(f"ADVERTENCIA: No se encontró el archivo index.faiss en {faiss_index_path}")
        
    try:
        print("Iniciando búsqueda de similitud en FAISS...")
        
        # Initialize OpenAI embeddings
        # Usar la versión importada de OpenAIEmbeddings (ya sea langchain_openai o langchain.embeddings.openai)
        embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
        
        # Load the FAISS index
        try:
            db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            print(f"FAISS index cargado correctamente desde {faiss_index_path}")
        except Exception as db_error:
            print(f"Error al cargar FAISS index: {str(db_error)}")
            import traceback
            traceback.print_exc()
            return message, None, None
        
        # Get number of chunks to retrieve from settings
        num_chunks = settings.get("rag_num_chunks", 3)
        print(f"Buscando los {num_chunks} chunks más similares")
        
        # Query the database for similar documents
        docs = db.similarity_search(actual_query, k=num_chunks)
        print(f"Se encontraron {len(docs)} documentos similares")
        
        # Extract text from retrieved documents
        context_texts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            file_type = doc.metadata.get('file_type', None)
            
            # Add a header that includes the source document information
            header = f"[Document {i+1} - {source}"
            if page != 'Unknown':
                header += f" (Page {page})"
            header += "]"
            
            # For JSON files, ensure the formatted content is preserved nicely
            if file_type == 'json':
                # JSON documents get special formatting
                context_texts.append(f"{header}\n{doc.page_content}\n")
                print(f"Documento JSON {i+1}: {source}, longitud: {len(doc.page_content)} caracteres")
            elif file_type == 'txt':
                # Text file documents
                context_texts.append(f"{header}\n{doc.page_content}\n")
                print(f"Documento TXT {i+1}: {source}, longitud: {len(doc.page_content)} caracteres")
            else:
                # Regular documents
                context_texts.append(f"{header}\n{doc.page_content}\n")
                print(f"Documento {i+1}: {source} (Página {page}), longitud: {len(doc.page_content)} caracteres")
        
        # Join the contexts
        context_to_add = "\n".join(context_texts)
        
        print(f"Contexto total: {len(context_to_add)} caracteres")
        print("Búsqueda RAG completada con éxito")
        print("="*50 + "\n")
        
        # Return the query without the **ProjectName** prefix and the context
        return actual_query, context_to_add, project_id
    
    except Exception as e:
        print(f"ERROR procesando RAG para el proyecto '{project_name}': {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n")
        return message, None, None

# Endpoint to clean all FAISS databases
@app.route('/api/rag/clean-all-databases', methods=['POST'])
def clean_all_rag_databases():
    """
    Clean all FAISS databases by completely removing and recreating the DB directory.
    This is useful when encountering persistent vector database errors.
    """
    try:
        # Get the path to the DB directory
        base_db_path = os.path.join(os.path.dirname(__file__), '..', 'db')
        
        print(f"Request to clean all RAG databases at: {base_db_path}")
        
        # Check if the directory exists
        if os.path.exists(base_db_path):
            print(f"Removing vector database directory: {base_db_path}")
            # Remove the entire directory
            shutil.rmtree(base_db_path)
            print("Vector database directory successfully removed")
        else:
            print("Vector database directory did not exist")
        
        # Recreate the directory
        print(f"Creating fresh vector database directory: {base_db_path}")
        os.makedirs(base_db_path, exist_ok=True)
        
        # Test that the directory is writable
        test_file_path = os.path.join(base_db_path, 'test_write.tmp')
        with open(test_file_path, 'w') as f:
            f.write('test')
        os.remove(test_file_path)
        print("RAG directory is properly configured and writable")
        
        return jsonify({
            "status": "success",
            "message": "All RAG databases successfully cleaned"
        }), 200
    except Exception as e:
        print(f"Error cleaning RAG databases: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Failed to clean RAG databases: {str(e)}"
        }), 500

@app.route('/api/projects/cleanup', methods=['POST'])
def cleanup_projects():
    """Clean up invalid or obsolete projects."""
    try:
        # Keep track of cleaned up projects
        removed_projects = []
        kept_projects = []
        
        # Create a list of projects with existing directories
        files_dir = '/Users/mbayasp/Library/CloudStorage/OneDrive-office.bayteq.com/Documents/Cursos/Master Ejecutivo en IA/5. Proyecto de Fin de Master/Ejercicio Seguro Medico/Code/programas/chat-app/files'
        existing_project_dirs = [d for d in os.listdir(files_dir) if os.path.isdir(os.path.join(files_dir, d))]
        
        # Identify projects to remove (those without corresponding directories)
        projects_to_remove = []
        for project_id, project in projects.items():
            project_title = project.get('title', '')
            
            # Skip projects without titles
            if not project_title:
                kept_projects.append(project)
                continue
                
            # If directory doesn't exist, mark for removal
            if project_title not in existing_project_dirs:
                projects_to_remove.append(project_id)
                removed_projects.append(project)
            else:
                kept_projects.append(project)
        
        # Remove the identified projects
        for project_id in projects_to_remove:
            # Delete project file
            project_file = os.path.join(PROJECT_DIR, f"{project_id}.json")
            if os.path.exists(project_file):
                os.remove(project_file)
            
            # Remove from memory
            if project_id in projects:
                del projects[project_id]
        
        print(f"Cleanup complete. Removed {len(removed_projects)} projects, kept {len(kept_projects)} projects.")
        
        return jsonify({
            "status": "success",
            "removed_count": len(removed_projects),
            "kept_count": len(kept_projects),
            "removed_projects": [{"id": p.get("id"), "title": p.get("title")} for p in removed_projects]
        }), 200
        
    except Exception as e:
        print(f"Error during project cleanup: {str(e)}")
        return jsonify({"error": f"Failed to clean up projects: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050) 