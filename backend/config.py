import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Config:
    def __init__(self):
        # OpenAI API configuration - Ahora siempre desde .env
        self.OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
        
        # Verificar y mostrar información sobre la API Key de OpenAI
        if self.OPENAI_API_KEY:
            print(f"API Key de OpenAI configurada. Primeros 4 caracteres: {self.OPENAI_API_KEY[:4]}...")
        else:
            print("ADVERTENCIA: No se configuró una API Key de OpenAI. El proveedor OpenAI no funcionará.")

        # LLM configuration
        self.MODEL = os.environ.get('LLM_MODEL', 'gemini-1.5-flash')
        self.TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', '0.7'))
        self.MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '800'))
        
        # Ollama configuration
        self.OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')
        
        # Google configuration
        self.GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
        
        # Verificar y mostrar información sobre la API Key de Google
        if self.GOOGLE_API_KEY:
            print(f"API Key de Google configurada. Primeros 4 caracteres: {self.GOOGLE_API_KEY[:4]}...")
        else:
            print("ADVERTENCIA: No se configuró una API Key de Google. El servicio de Gemini no funcionará.")
        
        # Configuración de modelo de Google - usar gemini-1.5-flash como predeterminado
        # por ser más estable que gemini 2.5 pro exp 03 25
        self.GOOGLE_MODEL = os.environ.get('GOOGLE_MODEL', 'gemini-1.5-flash')
        
        # Default provider
        self.DEFAULT_PROVIDER = os.environ.get('DEFAULT_PROVIDER', 'google')  # 'openai', 'ollama' o 'google' 