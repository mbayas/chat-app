import os
import requests
from openai import OpenAI
from typing import List, Dict, Any, Optional
import subprocess
import json
import sys
import time
import re
import google.generativeai as genai

class LLMService:
    """
    Service to handle interactions with different LLM providers.
    Currently supports OpenAI, Ollama, and Google Gemini, designed to be extended for other providers.
    """
    
    # Lista predeterminada de modelos de OpenAI
    OPENAI_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ]
    
    # Lista predeterminada de modelos populares de Ollama
    OLLAMA_MODELS = [
        "llama2",
        "mistral",
        "mixtral",
        "phi3",
        "gemma",
        "codellama",
        "llama2:13b",
        "llama2:70b",
        "neural-chat"
    ]
    
    # Lista predeterminada de modelos de Google Gemini
    GOOGLE_MODELS = [
        "gemini-1.0-pro",
        "gemini-1.0-pro-vision",
        "gemini-1.0-ultra",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.5-pro-001",
        "gemini-1.5-pro-002",
        "gemini-pro",
        "gemini-pro-vision"
    ]
    
    def __init__(
        self, 
        provider: str = "openai", 
        model: str = "gpt-3.5-turbo", 
        temperature: float = 0.7, 
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        google_api_key: Optional[str] = None
    ):
        """
        Initialize the LLM service with the specified parameters.
        
        Args:
            provider: The LLM provider to use (e.g., "openai", "ollama", "google" etc.)
            model: The specific model to use from the provider
            temperature: Controls randomness in responses (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate in the response
            api_key: Direct API key input for OpenAI (optional, will use environment variables if not provided)
            ollama_base_url: Base URL for Ollama API (default: http://localhost:11434)
            google_api_key: API key for Google Gemini (optional, will use environment variables if not provided)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_base_url = ollama_base_url
        
        # Use provided API keys or get from environment variables
        if api_key:
            self.openai_api_key = api_key
        else:
            # Set up API keys from environment variables
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
            
        if google_api_key:
            self.google_api_key = google_api_key
        else:
            # Set up Google API key from environment variables
            self.google_api_key = os.environ.get("GOOGLE_API_KEY")
            
        # Check for required API keys
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("API key is required when using OpenAI provider. Either provide it directly or set the OPENAI_API_KEY environment variable.")
        
        if self.provider == "google" and not self.google_api_key:
            raise ValueError("API key is required when using Google provider. Either provide it directly or set the GOOGLE_API_KEY environment variable.")
        
        # Initialize the appropriate client based on the provider
        if self.provider == "openai":
            self.client = OpenAI(api_key=self.openai_api_key)
            
        # Initialize Google Gemini client if needed
        if self.provider == "google":
            genai.configure(api_key=self.google_api_key)
    
    def get_available_models(self, provider=None) -> List[str]:
        """
        Get the list of available models from the provider.
        For OpenAI, we try to fetch from the API, but fall back to a predefined list if that fails.
        For Ollama, we try to fetch locally installed models.
        For Google, we return the available Gemini models.
        
        Args:
            provider: Override the provider to use (useful when provider hasn't been set yet)
            
        Returns:
            List of model IDs as strings
        """
        current_provider = provider or self.provider
        
        if current_provider == "openai":
            try:
                # Try to get models from the API
                response = self.client.models.list()
                models = [model.id for model in response.data if model.id.startswith("gpt")]
                # Filter to commonly used models if the list is too long
                if len(models) > 40:
                    return self.OPENAI_MODELS
                return sorted(models)
            except Exception as e:
                print(f"Error fetching OpenAI models: {str(e)}")
                # Fall back to predefined list
                return self.OPENAI_MODELS
        elif current_provider == "ollama":
            try:
                # Get models from Ollama CLI
                print("Trying to get models directly from Ollama CLI")
                try:
                    import subprocess
                    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:  # Skip header
                            models = []
                            for line in lines[1:]:  # Skip header row
                                parts = line.split()
                                if parts:
                                    models.append(parts[0])
                            if models:
                                print(f"Models from Ollama CLI: {models}")
                                return models
                except Exception as e:
                    print(f"Error getting models from Ollama CLI: {e}")
                    
                # Fallback to API if CLI fails
                print(f"Fetching Ollama models from: {self.ollama_base_url}/api/tags")
                response = requests.get(f"{self.ollama_base_url}/api/tags")
                
                if response.status_code != 200:
                    print(f"Error fetching Ollama models. Status code: {response.status_code}")
                    print(f"Response: {response.text}")
                    return self.OLLAMA_MODELS
                
                # Parse the response data
                try:
                    data = response.json()
                    print(f"Ollama API response: {data}")
                    
                    # The API returns a list of objects under 'models' key
                    if 'models' in data:
                        # Newer Ollama API format
                        models = [model["name"] for model in data["models"]]
                    else:
                        # Direct list format in older Ollama versions
                        models = [model["name"] for model in data]
                    
                    if models:
                        print(f"Detected Ollama models: {models}")
                        return sorted(models)
                    else:
                        print("No models found in Ollama response")
                except Exception as e:
                    print(f"Error parsing Ollama response: {str(e)}")
                
                # Fall back to predefined list if we couldn't get valid models
                print("Falling back to predefined Ollama models")
                return ["llama2"]
            except Exception as e:
                print(f"Exception when fetching Ollama models: {str(e)}")
                return ["llama2"]
        elif current_provider == "google":
            try:
                # Para Google Gemini, intentar obtener modelos disponibles usando la nueva API
                try:
                    # Listar modelos disponibles
                    available_models = genai.list_models()
                    # Filtrar solo los modelos de Gemini
                    gemini_models = [model.name.split("/")[-1] for model in available_models if "gemini" in model.name]
                    
                    if gemini_models:
                        print(f"Available Google Gemini models: {gemini_models}")
                        return gemini_models
                    else:
                        # Fall back to predefined list
                        print("No Gemini models found via API. Using default list.")
                        return self.GOOGLE_MODELS
                except Exception as e:
                    print(f"Error listing models from Google API: {str(e)}")
                    # Fall back to predefined list
                    return self.GOOGLE_MODELS
            except Exception as e:
                print(f"Error fetching Google models: {str(e)}")
                # Fall back to predefined list
                return self.GOOGLE_MODELS
        else:
            return []
    
    def update_settings(
        self, 
        provider: Optional[str] = None, 
        model: Optional[str] = None, 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        google_api_key: Optional[str] = None
    ):
        """
        Update the LLM service settings.
        
        Args:
            provider: The LLM provider to use
            model: The specific model to use from the provider
            temperature: Controls randomness in responses
            max_tokens: Maximum number of tokens to generate in the response
            api_key: Direct OpenAI API key input
            ollama_base_url: Base URL for Ollama API
            google_api_key: API key for Google Gemini
        """
        if provider is not None:
            self.provider = provider
        
        if model is not None:
            self.model = model
        
        if temperature is not None:
            self.temperature = temperature
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            
        if api_key is not None:
            self.openai_api_key = api_key
            # Re-initialize the client with the new API key
            if self.provider == "openai":
                self.client = OpenAI(api_key=self.openai_api_key)
                
        if ollama_base_url is not None:
            self.ollama_base_url = ollama_base_url
            
        if google_api_key is not None:
            self.google_api_key = google_api_key
            # Re-initialize the Google client with the new API key
            if self.provider == "google":
                genai.configure(api_key=self.google_api_key)
    
    def update_api_key(self, api_key: str, provider: str = None):
        """
        Update just the API key.
        
        Args:
            api_key: The new API key
            provider: The provider to update the API key for (defaults to current provider)
        """
        current_provider = provider or self.provider
        
        if current_provider == "openai":
            self.openai_api_key = api_key
            if self.provider == "openai":
                self.client = OpenAI(api_key=self.openai_api_key)
        elif current_provider == "google":
            self.google_api_key = api_key
            if self.provider == "google":
                genai.configure(api_key=self.google_api_key)
    
    def generate_response(self, message: str = None, chat_history: List[Dict[str, Any]] = None, history: List[Dict[str, Any]] = None, provider: str = None) -> str:
        """
        Generate a response from the LLM based on the message and chat history.
        
        Args:
            message: The user's message (optional if history is provided)
            chat_history: The full chat history in the format [{"role": "user", "content": "..."}, ...] (deprecated)
            history: The chat history in the format [{"role": "user", "content": "..."}, ...] (new parameter)
            provider: Override the provider to use
            
        Returns:
            The LLM's response as a string
        """
        # Use specified provider or fallback to instance provider
        current_provider = provider or self.provider
        print(f"Generate response using provider: {current_provider}")
        
        # Use new history parameter if provided, otherwise use chat_history
        messages = history or chat_history or []
        
        # Handle single message if provided without history
        if message and not messages:
            messages = [{"role": "user", "content": message}]
        elif message and messages and messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": message})
        
        # Imprimir información sobre el contexto
        if messages:
            total_messages = len(messages)
            print(f"Total de mensajes en el contexto: {total_messages}")
            
            # Obtener todo el texto del contexto para analizarlo
            all_context_text = " ".join([msg["content"] for msg in messages])
            total_chars = len(all_context_text)
            print(f"Longitud total del contexto: {total_chars} caracteres")
            
            # Imprimir los primeros y últimos 60 caracteres
            if total_chars > 0:
                first_60 = all_context_text[:min(60, total_chars)]
                last_60 = all_context_text[max(0, total_chars-60):] if total_chars > 60 else ""
                print(f"Primeros 60 caracteres: '{first_60}'")
                if total_chars > 60:
                    print(f"Últimos 60 caracteres: '{last_60}'")
        
        # Ensure we're using the correct method for the provider
        if current_provider == "openai":
            print("Using OpenAI method")
            if not self.openai_api_key:
                return "Error: OpenAI API key is required for OpenAI provider."
            return self._generate_openai_response(messages)
        elif current_provider == "ollama":
            print("Using Ollama method")
            # Forzar el uso del método Ollama para modelos de Ollama
            return self._generate_ollama_response(messages)
        elif current_provider == "google":
            print("Using Google method")
            if not self.google_api_key:
                return "Error: Google API key is required for Google provider."
            return self._generate_google_response(messages)
        else:
            raise ValueError(f"Unsupported provider: {current_provider}")
    
    def _generate_openai_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            messages: The chat messages
        
        Returns:
            The OpenAI model's response as a string
        """
        # Filter to include only valid roles for OpenAI
        filtered_messages = []
        for msg in messages:
            if msg["role"] in ["user", "assistant", "system"]:
                filtered_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        try:
            # Print filtered messages for debugging
            print(f"Filtered messages for OpenAI: {json.dumps(filtered_messages, indent=2)}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=filtered_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating OpenAI response: {str(e)}")
            return f"I'm sorry, I encountered an error: {str(e)}"

    def _generate_google_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the Google Gemini API.
        
        Args:
            messages: The chat messages
        
        Returns:
            The Google Gemini model's response as a string
        """
        try:
            # Declarar gemini_messages al inicio para tenerlo disponible en bloques de captura
            gemini_messages = []
            response_tried = False
            
            print(f"Generating response with Google Gemini model: {self.model}")
            
            # Si no hay mensajes, devolver error claro
            if not messages or len(messages) == 0:
                return "Error: No hay mensajes para procesar."
            
            # Comprobar si el modelo es válido y existe en la API
            try:
                available_models = genai.list_models()
                available_model_names = [model.name.split("/")[-1] for model in available_models if "gemini" in model.name]
                print(f"Modelos disponibles en la API de Google: {available_model_names}")
                
                model_exists = False
                # Primero, verificar si el modelo ya está en el formato correcto
                if self.model in available_model_names:
                    model_exists = True
                    print(f"Modelo {self.model} encontrado en la API")
                # Luego, verificar si es el modelo experimental con nombre especial
                elif "2.5 pro exp 03 25" in self.model.lower() or "gemini 2.5 pro exp 03 25" in self.model.lower():
                    print(f"Detectado modelo experimental: {self.model}")
                    original_model = self.model
                    # Ajustar al nombre exacto en la API
                    self.model = "gemini-1.5-flash"
                    print(f"Ajustando al nombre exacto de la API: {self.model}")
                    
                    # Verificar si este nombre exacto existe
                    if self.model in available_model_names:
                        model_exists = True
                        print(f"Modelo {self.model} encontrado en la API después del ajuste")
                
                # Si el modelo no existe, usar fallback
                if not model_exists:
                    print(f"Advertencia: El modelo {self.model} no aparece en la lista de modelos disponibles")
                    if "gemini-1.5-flash" in available_model_names:
                        self.model = "gemini-1.5-flash"
                        print(f"Usando modelo alternativo: {self.model}")
                    elif "gemini-1.5-pro" in available_model_names:  # Otra opción más estable
                        self.model = "gemini-1.5-pro"
                        print(f"Usando modelo alternativo: {self.model}")
                    elif "gemini-pro" in available_model_names:  # Opción más estable y antigua
                        self.model = "gemini-pro"
                        print(f"Usando modelo alternativo estable: {self.model}")
                    elif len(available_model_names) > 0:
                        # Usar el primer modelo disponible como último recurso
                        self.model = available_model_names[0]
                        print(f"Usando primer modelo disponible: {self.model}")
                    else:
                        # Si no hay modelos disponibles, usar un valor predeterminado común
                        self.model = "gemini-pro"
                        print(f"No se encontraron modelos disponibles. Usando modelo predeterminado: {self.model}")
                
            except Exception as e:
                print(f"Error al listar modelos disponibles: {str(e)}")
                # Si hay error al listar modelos, usar un modelo común como fallback
                self.model = "gemini-pro"  # Usar modelo más estable como fallback
                print(f"Usando modelo fallback estable debido al error: {self.model}")
            
            # Convert our message format to Google's format
            # Google Gemini requires different formatting than OpenAI
            try:
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    # Verificar que role y content no estén vacíos para evitar errores
                    if not role or not content:
                        print(f"Advertencia: Mensaje con role o content vacío: {msg}")
                        continue
                
                # Map roles: user->user, assistant->model, system->user (with special prefix)
                if role == "user":
                        # Usar el formato actualizado para mensajes del usuario
                    gemini_messages.append({"role": "user", "parts": [{"text": content}]})
                elif role == "assistant":
                        # Usar el formato actualizado para mensajes del asistente
                    gemini_messages.append({"role": "model", "parts": [{"text": content}]})
                elif role == "system":
                        # Usar el formato actualizado para mensajes del sistema (como instrucciones al usuario)
                    gemini_messages.append({"role": "user", "parts": [{"text": f"[SYSTEM INSTRUCTION] {content}"}]})
            except Exception as e:
                print(f"Error al convertir mensajes al formato de Gemini: {str(e)}")
                # Si hay error al convertir mensajes, intentar regenerar con formato mínimo
                gemini_messages = []
                if len(messages) > 0 and "content" in messages[-1]:
                    try:
                        last_content = messages[-1].get("content", "")
                        if last_content:
                            gemini_messages.append({"role": "user", "parts": [{"text": last_content}]})
                            print("Regenerando mensajes con formato mínimo (solo último mensaje)")
                    except Exception as inner_e:
                        print(f"Error también al regenerar mensajes mínimos: {str(inner_e)}")
            
            # Verificar si hay mensajes para procesar después de la conversión
            if len(gemini_messages) == 0:
                return "Error: No hay mensajes válidos para procesar después de la conversión al formato de Gemini."
            
            # Configure the Gemini model with less restrictive safety settings
            try:
                # Usar el formato actualizado para generar el modelo
                model = genai.GenerativeModel(
                    model_name=self.model,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                        "top_p": 0.95,
                    },
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_ONLY_HIGH"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_ONLY_HIGH"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_ONLY_HIGH"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_ONLY_HIGH"
                        }
                    ]
                )
            except Exception as e:
                error_str = str(e)
                print(f"Error al configurar el modelo: {error_str}")
                return f"Error al configurar el modelo Gemini: {error_str}"
            
            # Si la consulta contiene términos potencialmente sensibles en seguros médicos, reformulamos
            reformulated_message = False
            if len(gemini_messages) > 0:
                # Acceder al último mensaje de manera segura
                try:
                    last_msg = gemini_messages[-1]["parts"][0]["text"]
                    if any(term in last_msg.lower() for term in ["seguro médico", "seguro_medico", "medico", "médico", "hospital", "clínic"]):
                        print("Detectada consulta relacionada con seguros médicos o salud. Añadiendo contexto seguro.")
                        if "**Seguro_Medico**" not in last_msg:
                            # Añadir instrucciones para aclarar el contexto informativo de la consulta
                            gemini_messages[-1]["parts"][0]["text"] = (
                                "Quiero información educativa sobre: " + last_msg + 
                                ". Mi consulta es puramente académica sobre el sector de seguros en Ecuador."
                            )
                            reformulated_message = True
                            print(f"Mensaje reformulado para evitar bloqueos: {gemini_messages[-1]['parts'][0]['text']}")
                except (IndexError, KeyError) as e:
                    print(f"Error al intentar reformular el mensaje: {str(e)}")
                    # No detener el proceso, simplemente continuar sin reformular
            
            print(f"Mensajes a enviar: {gemini_messages}")
            
            # If we have multiple messages, use chat mode
            content = None
            response_tried = False
            
            # Intentos máximos en caso de errores 503
            max_retries = 3
            retry_count = 0
            retry_delay = 2  # segundos
            
            while retry_count < max_retries:
                try:
                    if len(gemini_messages) > 1:
                        # Proteger la obtención del último mensaje con try-except
                        try:
                            # Start a chat session
                            chat = model.start_chat(history=gemini_messages[:-1])
                            # Send the last message to get a response
                            last_msg = gemini_messages[-1]["parts"][0]["text"]
                            response = chat.send_message(last_msg, stream=False)
                            response_tried = True
                            
                            if hasattr(response, 'text') and response.text:
                                content = response.text
                                break  # Salir del bucle de reintentos al tener éxito
                            elif hasattr(response, 'parts') and response.parts:
                                # Handle case where text is in parts
                                parts_text = []
                                for part in response.parts:
                                    if hasattr(part, 'text') and part.text:
                                        parts_text.append(part.text)
                                content = ''.join(parts_text)
                                break  # Salir del bucle de reintentos al tener éxito
                        except (IndexError, KeyError) as e:
                            print(f"Error al acceder al último mensaje: {str(e)}")
                            # Si hay error al acceder al último mensaje, intentar con solo el primer mensaje
                            # Esto podría funcionar en casos donde el último mensaje es el que causa problemas
                            try:
                                if len(gemini_messages) > 0:
                                    first_msg = gemini_messages[0]["parts"][0]["text"]
                                    print(f"Intentando con solo el primer mensaje: {first_msg}")
                                    response = model.generate_content(first_msg)
                                    if hasattr(response, 'text') and response.text:
                                        content = response.text
                                        break  # Salir del bucle de reintentos al tener éxito
                            except Exception as e2:
                                print(f"Error también al usar solo el primer mensaje: {str(e2)}")
                    elif len(gemini_messages) == 1:
                        try:
                            # For single messages, just generate a response
                            last_msg = gemini_messages[0]["parts"][0]["text"]
                            print(f"Enviando mensaje directo a Gemini: {last_msg}")
                            
                            # Intenta primero con generate_content
                            response = model.generate_content(last_msg)
                            response_tried = True
                            
                            if hasattr(response, 'text') and response.text:
                                content = response.text
                                break  # Salir del bucle de reintentos al tener éxito
                            elif hasattr(response, 'parts') and response.parts:
                                # Handle case where text is in parts
                                parts_text = []
                                for part in response.parts:
                                    if hasattr(part, 'text') and part.text:
                                        parts_text.append(part.text)
                                content = ''.join(parts_text)
                                break  # Salir del bucle de reintentos al tener éxito
                            else:
                                content = "Lo siento, no pude generar una respuesta. Intente de nuevo con una pregunta diferente."
                                # No hacemos break aquí, intentamos de nuevo
                        except Exception as inner_e:
                            inner_error_str = str(inner_e)
                            print(f"Error en generate_content: {inner_error_str}")
                            
                            # Detectar específicamente error 503
                            if "503" in inner_error_str and "overloaded" in inner_error_str:
                                print(f"Modelo sobrecargado (503). Reintento {retry_count+1}/{max_retries}")
                                retry_count += 1
                                if retry_count < max_retries:
                                    print(f"Esperando {retry_delay} segundos antes de reintentar...")
                                    time.sleep(retry_delay)
                                    retry_delay *= 2  # Backoff exponencial
                                    continue  # Pasar a la siguiente iteración del bucle
                            
                            # Si tenemos el error específico de prompt bloqueado y NO hemos reformulado, intentar reformular
                            if ("response.parts quick accessor only works for a single candidate" in inner_error_str or 
                                "Check the response.prompt_feedback" in inner_error_str) and not reformulated_message:
                                print("Intentando reformular el prompt para evitar bloqueos...")
                                try:
                                    # Reformular el mensaje para evitar bloqueos
                                    reformulated_msg = (
                                        "Estoy haciendo una investigación académica sobre este tema: " + 
                                        last_msg + 
                                        ". Por favor proporciona información educativa y objetiva al respecto."
                                    )
                                    
                                    print(f"Mensaje reformulado: {reformulated_msg}")
                                    
                                    # Intentar nuevamente con el mensaje reformulado
                                    reformulated_response = model.generate_content(reformulated_msg)
                                    
                                    if hasattr(reformulated_response, 'text') and reformulated_response.text:
                                        content = reformulated_response.text
                                        break  # Salir del bucle de reintentos al tener éxito
                                    elif hasattr(reformulated_response, 'parts') and reformulated_response.parts:
                                        parts_text = []
                                        for part in reformulated_response.parts:
                                            if hasattr(part, 'text') and part.text:
                                                parts_text.append(part.text)
                                        content = ''.join(parts_text)
                                        break  # Salir del bucle de reintentos al tener éxito
                                except Exception as reformulate_error:
                                    print(f"Error al intentar reformular: {str(reformulate_error)}")
                                    
                                    # Si reformular también falla, intentar con el modelo de fallback
                                    try:
                                        print("Intentando con modelo de fallback Gemini Pro...")
                                        fallback_model = genai.GenerativeModel(
                                            model_name="gemini-pro",  # Usar el modelo más estable
                                            generation_config={
                                                "temperature": self.temperature,
                                                "max_output_tokens": self.max_tokens,
                                                "top_p": 0.95,
                                            }
                                        )
                                        fallback_response = fallback_model.generate_content(last_msg)
                                        if hasattr(fallback_response, 'text') and fallback_response.text:
                                            content = "[Usando modelo Gemini Pro como alternativa] " + fallback_response.text
                                            break  # Salir del bucle de reintentos al tener éxito
                                    except Exception as fallback_error:
                                        print(f"Error al usar modelo de fallback: {str(fallback_error)}")
                    else:
                        # Caso inesperado: no hay mensajes después de la conversión
                        return "Error: No se pudieron procesar los mensajes adecuadamente para el formato de Gemini."
                        
                except Exception as e:
                    error_str = str(e)
                    print(f"Error en respuesta de Google Gemini: {error_str}")
                    
                    # Detectar específicamente error 503
                    if "503" in error_str and "overloaded" in error_str:
                        print(f"Modelo sobrecargado (503). Reintento {retry_count+1}/{max_retries}")
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"Esperando {retry_delay} segundos antes de reintentar...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Backoff exponencial
                            continue  # Pasar a la siguiente iteración del bucle
                    
                    # Si reformular también falla, intentar con el modelo de fallback
                    if retry_count >= max_retries - 1:  # En el último intento
                        try:
                            print("Intentando con modelo de fallback Gemini Pro como último recurso...")
                            
                            # Extraer el último mensaje de manera segura
                            last_attempt_msg = ""
                            try:
                                if len(gemini_messages) > 0:
                                    last_attempt_msg = gemini_messages[0]["parts"][0]["text"]
                            except (IndexError, KeyError):
                                # Si no podemos acceder al último mensaje, crear uno simple
                                if len(messages) > 0 and "content" in messages[-1]:
                                    last_attempt_msg = messages[-1]["content"]
                                else:
                                    last_attempt_msg = "Por favor proporciona información educativa sobre seguros médicos."
                            
                            fallback_model = genai.GenerativeModel(
                                model_name="gemini-pro",  # Usar el modelo más estable
                                generation_config={
                                    "temperature": self.temperature,
                                    "max_output_tokens": self.max_tokens,
                                    "top_p": 0.95,
                                }
                            )
                            fallback_response = fallback_model.generate_content(last_attempt_msg)
                            if hasattr(fallback_response, 'text') and fallback_response.text:
                                content = "[Usando modelo Gemini Pro como alternativa] " + fallback_response.text
                                break  # Salir del bucle de reintentos al tener éxito
                        except Exception as fallback_error:
                            print(f"Error también con el modelo de fallback: {str(fallback_error)}")
                
                # Incrementar contador de reintentos si llegamos a este punto (hubo error)
                retry_count += 1
                
                # Si ya hemos intentado varias veces, cambiar a un modelo diferente en el siguiente intento
                if retry_count == 1:
                    try:
                        print("Cambiando a modelo Gemini Pro para el siguiente intento")
                        model = genai.GenerativeModel(
                            model_name="gemini-pro",
                            generation_config={
                                "temperature": self.temperature,
                                "max_output_tokens": self.max_tokens,
                                "top_p": 0.95,
                            }
                        )
                    except Exception as model_change_error:
                        print(f"Error al cambiar de modelo: {str(model_change_error)}")
            
            # Si tenemos contenido después de todos los intentos, devolverlo
            if content:
                return content
            
            # Si llegamos a este punto, devolver un mensaje de error general
            # Generar mensaje según el problema
            if retry_count >= max_retries:
                return "Error: El modelo de Google Gemini está sobrecargado. Por favor, intente más tarde o use un modelo diferente."
            else:
                return "Error: No se pudo generar una respuesta con Google Gemini después de varios intentos. Por favor, intente reformular su pregunta."
            
        except Exception as e:
            error_str = str(e)
            print(f"Error generating Google Gemini response: {error_str}")
            
            # Si no hemos intentado ninguna respuesta, intentar con modelo de fallback
            if not response_tried:
                try:
                    # Obtener el último mensaje para intentar con un modelo de fallback
                    try:
                        last_attempt_msg = ""
                        if len(gemini_messages) > 0 and "parts" in gemini_messages[-1]:
                            last_attempt_msg = gemini_messages[-1]["parts"][0]["text"]
                        elif len(messages) > 0 and "content" in messages[-1]:
                            last_attempt_msg = messages[-1]["content"]
                        
                        if last_attempt_msg:
                            print(f"Intentando con modelo Gemini Pro como último recurso")
                            fallback_model = genai.GenerativeModel(
                                model_name="gemini-pro",  # Usar el modelo más estable
                                generation_config={
                                    "temperature": self.temperature,
                                    "max_output_tokens": self.max_tokens,
                                    "top_p": 0.95,
                                }
                            )
                            fallback_response = fallback_model.generate_content(last_attempt_msg)
                            if hasattr(fallback_response, 'text') and fallback_response.text:
                                return "[Respuesta de Gemini Pro] " + fallback_response.text
                    except (IndexError, KeyError) as ie:
                        print(f"Error al intentar acceder al mensaje para fallback: {str(ie)}")
                except Exception as fallback_error:
                    print(f"Error también con el modelo de fallback: {str(fallback_error)}")
            
            # Provide more helpful error messages based on the type of error
            if "503" in error_str and "overloaded" in error_str:
                return "Error: El modelo de Google Gemini está sobrecargado. Por favor, intente más tarde o use un modelo diferente como 'gemini-pro'."
            elif "response.parts quick accessor only works for a single candidate, but none were returned" in error_str:
                return "Error: Su consulta fue bloqueada por los sistemas de seguridad de Google. Esto puede ocurrir al preguntar sobre temas sensibles como salud o seguros. Por favor, reformule su pregunta de manera más general o educativa."
            elif "Check the response.prompt_feedback" in error_str:
                return "Error: Su consulta fue bloqueada por los filtros de contenido. Por favor reformule la pregunta enfocándose en aspectos educativos o informativos generales."
            elif "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in error_str:
                return "Error: Su clave API de Google no tiene permisos suficientes. Obtenga una nueva clave API con los permisos adecuados desde Google AI Studio."
            elif "NOT_FOUND" in error_str and "model" in error_str.lower():
                return f"Error: El modelo '{self.model}' no se encontró. Por favor, verifique los modelos disponibles en la configuración o use 'gemini-pro'."
            elif "PERMISSION_DENIED" in error_str:
                return "Error: Permiso denegado. Verifique que su clave API sea válida y tenga acceso a la API de Gemini."
            elif "RESOURCE_EXHAUSTED" in error_str:
                return "Error: Cuota de API excedida. Intente de nuevo más tarde o verifique la configuración de su cuota de Google Cloud."
            elif "INVALID_ARGUMENT" in error_str:
                return "Error: Solicitud no válida a la API de Gemini. Intente con un mensaje diferente o verifique su configuración."
            elif "list index out of range" in error_str:
                return "Error: Se produjo un error al acceder a los mensajes. Por favor, intente de nuevo con una consulta clara y directa."
            else:
                return f"Lo siento, encontré un error: {error_str}. Por favor, intente con un modelo más estable como 'gemini-pro'."
    
    def _generate_ollama_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the Ollama CLI directly, bypassing API issues.
        
        Args:
            messages: The chat messages
        
        Returns:
            The Ollama model's response as a string
        """
        # Filter to include only valid roles for Ollama
        filtered_messages = []
        for msg in messages:
            if msg["role"] in ["user", "assistant", "system"]:
                filtered_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        print(f"Mensaje filtrados para Ollama: {len(filtered_messages)} mensajes")
        
        try:
            # Try using Ollama API first (preferred method)
            try:
                return self._generate_ollama_api_response(filtered_messages)
            except Exception as api_error:
                print(f"Error using Ollama API: {str(api_error)}. Falling back to CLI.")
            
            # Get the list of available models
            import subprocess
            print("Getting available models from ollama list")
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            available_models = []
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:  # Skip header row
                        parts = line.split()
                        if parts:
                            available_models.append(parts[0])
            
            print(f"Available models from ollama list: {available_models}")
            
            # Select model to use
            if self.model in available_models:
                model_to_use = self.model
            elif available_models:
                # Use first available model as fallback
                model_to_use = available_models[0]
                print(f"Model {self.model} not available, using {model_to_use} instead")
            else:
                # Fallback to a common model name
                model_to_use = "llama2"
                print(f"No models available, trying with {model_to_use}")
            
            # Get last user message
            last_user_message = ""
            for msg in reversed(filtered_messages):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break
            
            if not last_user_message:
                return "No user message found to process."
            
            # Mostrar información sobre el último mensaje del usuario
            if len(last_user_message) > 120:
                print(f"Enviando a Ollama - Primeros 60 chars: '{last_user_message[:60]}'")
                print(f"Enviando a Ollama - Últimos 60 chars: '{last_user_message[-60:]}'")
            else:
                print(f"Enviando a Ollama mensaje completo: '{last_user_message}'")
            
            # Configurar timeout más largo para modelos grandes
            timeout = 60  # 60 segundos para modelos grandes
            if "3b" in model_to_use or "7b" in model_to_use or "13b" in model_to_use or "70b" in model_to_use:
                timeout = 120  # 2 minutos para modelos grandes
            
            print(f"Running ollama with model {model_to_use} and message of length: {len(last_user_message)} chars")
            
            # Usar comandos que funcionan mejor con modelos grandes
            cmd = ['ollama', 'run', model_to_use, last_user_message, '--nowordwrap']
            
            print(f"Executing command: {' '.join(cmd)}")
            
            # Use subprocess.run instead of Popen for simpler execution
            cmd_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if cmd_result.returncode != 0:
                error_msg = f"Error running ollama: {cmd_result.stderr}"
                print(error_msg)
                return f"I'm sorry, I encountered an error: {error_msg}"
            
            response = cmd_result.stdout.strip()
            
            # Intentar extraer solo la respuesta del modelo (eliminar eco del prompt)
            if last_user_message in response:
                response = response.replace(last_user_message, "", 1).strip()
            
            # Mostrar información sobre la respuesta
            if len(response) > 120:
                print(f"Respuesta de Ollama (primeros 60 chars): '{response[:60]}'")
                print(f"Respuesta de Ollama (últimos 60 chars): '{response[-60:]}'")
            else:
                print(f"Respuesta completa de Ollama: '{response}'")
            
            return response
            
        except subprocess.TimeoutExpired:
            return "I'm sorry, the model took too long to respond. Please try again with a shorter question or a smaller model."
        except Exception as e:
            # Log the error and return a friendly message
            error_msg = f"Error generating Ollama response: {str(e)}"
            print(error_msg)
            return f"I'm sorry, I encountered an error: {error_msg}"
    
    def _generate_ollama_api_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the Ollama API.
        
        Args:
            messages: The chat messages
        
        Returns:
            The model's response
        """
        import requests
        import json
        
        # Filter to include only valid roles for Ollama (user, assistant, system)
        filtered_messages = []
        for msg in messages:
            if msg["role"] in ["user", "assistant", "system"]:
                filtered_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Log the message context being used
        print(f"Ollama non-streaming with {len(filtered_messages)} messages in context")
        for i, msg in enumerate(filtered_messages):
            print(f"Message {i+1}: role={msg['role']}, content_length={len(msg['content'])}")
            
        # Construir el prompt completo para Ollama
        # Ollama no procesa directamente la estructura de mensajes como OpenAI,
        # así que tenemos que construir un único prompt con todos los mensajes
        system_content = ""
        prompt_parts = []
        
        # Primero, extraer todos los mensajes del sistema y combinarlos
        for msg in filtered_messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n\n"
                
        # Añadir el contenido del sistema al principio si existe
        if system_content:
            prompt_parts.append(f"System: {system_content.strip()}\n\n")
            
        # Ahora añadir la conversación entre user y assistant
        for msg in filtered_messages:
            if msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content'].strip()}\n")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content'].strip()}\n")
        
        # Finalizar el prompt para que el modelo sepa que es su turno de responder
        prompt_parts.append("Assistant: ")
        
        # Juntar todo en un único prompt
        full_prompt = "".join(prompt_parts)
        
        # Log mensaje para debugging
        print(f"Full prompt length: {len(full_prompt)} chars")
        print(f"Prompt preview (first 300 chars): {full_prompt[:300]}...")
        
        # Create the API request
        url = f"{self.ollama_base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }
        
        print(f"Sending request to Ollama API: {url}")
        print(f"Using model: {self.model}")
        
        try:
            response = requests.post(url, json=data)
            
            if response.status_code != 200:
                error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                print(error_msg)
                return f"Error: {error_msg}"
            
            result = response.json()
            response_text = result.get("response", "")
            print(f"Ollama API response received, length: {len(response_text)} chars")
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error communicating with Ollama API: {str(e)}"
            print(error_msg)
            return f"Error: {error_msg}"
    
    def generate_streaming_response(self, message: str = None, chat_history: List[Dict[str, Any]] = None, 
                                  history: List[Dict[str, Any]] = None, provider: str = None):
        """
        Generator function that yields chunks of the response as they are generated.
        
        Args:
            message: The user's message (optional if history is provided)
            chat_history: The full chat history in the format [{"role": "user", "content": "..."}, ...] (deprecated)
            history: The chat history in the format [{"role": "user", "content": "..."}, ...] (new parameter)
            provider: Override the provider to use
            
        Yields:
            Chunks of the LLM's response as they are generated
        """
        # Use specified provider or fallback to instance provider
        current_provider = provider or self.provider
        print(f"Generate streaming response using provider: {current_provider}")
        
        # Use new history parameter if provided, otherwise use chat_history
        messages = history or chat_history or []
        
        # Handle single message if provided without history
        if message and not messages:
            messages = [{"role": "user", "content": message}]
        elif message and messages and messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": message})
        
        if current_provider == "ollama":
            yield from self._generate_ollama_streaming_response(messages)
        elif current_provider == "openai":
            yield from self._generate_openai_streaming_response(messages)
        elif current_provider == "google":
            yield from self._generate_google_streaming_response(messages)
        else:
            # For non-streaming providers, generate the full response and yield it
            full_response = self.generate_response(message, chat_history, history, provider)
            yield full_response
    
    def _generate_ollama_streaming_response(self, messages: List[Dict[str, Any]]):
        """
        Generate a streaming response using the Ollama API.
        
        Args:
            messages: The chat messages
        
        Yields:
            Chunks of the Ollama model's response as they are generated
        """
        import requests
        import json
        import time
        
        # Verificar que tenemos la URL base para Ollama
        if not hasattr(self, 'ollama_base_url') or not self.ollama_base_url:
            self.ollama_base_url = "http://localhost:11434"
            print(f"Using default Ollama URL: {self.ollama_base_url}")
            
        # Convertir mensajes al formato de Ollama
        ollama_messages = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if not role or not content:
                continue
            
            # Ollama usa 'system', 'user', 'assistant'
            if role in ["system", "user", "assistant"]:
                ollama_messages.append({
                    "role": role, 
                    "content": content
                })
            else:
                # Tratar mensajes con roles desconocidos como mensajes de usuario
                ollama_messages.append({
                    "role": "user",
                    "content": content
                })
        
        try:
            # Preparar la solicitud para la API de chat de Ollama
            url = f"{self.ollama_base_url}/api/chat"
            data = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": True,  # Activar streaming
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            print(f"Sending streaming request to Ollama API: {json.dumps(data, indent=2)}")
            
            # Realizar la solicitud en modo streaming
            with requests.Session() as session:
                response = session.post(url, json=data, stream=True, timeout=180)
                
                if not response.ok:
                    error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                    print(error_msg)
                    yield error_msg
                    return
                
                print("Connected to Ollama API, streaming started")
                
                # Variables para el streaming
                current_chunk = ""
                
                # Procesar la respuesta streaming línea por línea
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    # Decodificar y procesar la línea JSON
                    try:
                        chunk_data = json.loads(line.decode('utf-8'))
                        
                        # Verificar si la generación ha terminado
                        if chunk_data.get("done", False):
                            print("Ollama streaming completed")
                            # Enviar cualquier contenido restante
                            if current_chunk:
                                yield current_chunk
                            break
                        
                        # Extraer el contenido de la respuesta
                        content = None
                        if "message" in chunk_data and "content" in chunk_data["message"]:
                            content = chunk_data["message"]["content"]
                        elif "response" in chunk_data:
                            content = chunk_data["response"]
                        
                        if content:
                            # Ver si debemos enviar el contenido actual
                            current_chunk += content
                            
                            # Enviar contenido si es un token significativo (palabra completa, puntuación, espacio)
                            if (" " in content or "." in content or "," in content or 
                                "?" in content or "!" in content or ":" in content or 
                                ";" in content or "\n" in content):
                                print(f"Yielding chunk: '{current_chunk}'")
                                yield current_chunk
                                current_chunk = ""
                            
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from Ollama: {str(e)}")
                        continue
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error with Ollama API: {str(e)}"
            print(error_msg)
            yield error_msg
            
        except Exception as e:
            error_msg = f"Error in Ollama streaming: {str(e)}"
            print(error_msg)
            yield error_msg
    
    def _generate_openai_streaming_response(self, messages: List[Dict[str, Any]]):
        """
        Generate a streaming response using the OpenAI API.
        
        Args:
            messages: The chat messages
        
        Yields:
            Chunks of the OpenAI model's response as they are generated
        """
        try:
            print(f"Starting OpenAI streaming with model: {self.model}")
            print(f"Number of messages in history: {len(messages)}")
            
            # Check if we're using a Google model with OpenAI provider
            if "gemini" in self.model.lower():
                error_msg = f"Error: Model '{self.model}' is a Google Gemini model but OpenAI provider was selected. Please use the Google provider for Gemini models."
                print(error_msg)
                yield error_msg
                return
            
            # Filter to include only valid roles for OpenAI
            filtered_messages = []
            for msg in messages:
                if msg["role"] in ["user", "assistant", "system"]:
                    filtered_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Create a new client instance for each request to ensure clean state
            client = OpenAI(api_key=self.openai_api_key)
            
            # Create the chat completion with streaming enabled - only use necessary parameters
            stream = client.chat.completions.create(
                model=self.model,
                messages=filtered_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            print("Stream created, waiting for first token...")
            
            # Process the streaming response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    yield token
                    
            print("Streaming completed")
                    
        except Exception as e:
            print(f"Error in OpenAI streaming response: {str(e)}")
            yield f"Error: {str(e)}"
    
    def _generate_google_streaming_response(self, messages: List[Dict[str, Any]]):
        """
        Generate a streaming response using the Google Gemini API.
        
        Args:
            messages: The chat messages
        
        Yields:
            Chunks of the Google Gemini model's response as they are generated
        """
        try:
            print(f"Starting Google Gemini streaming with model: {self.model}")
            print(f"Number of messages in history: {len(messages)}")
            
            # Verify API key is set
            if not self.google_api_key:
                error_msg = "Google API key is not configured. Please set your API key in settings."
                print(error_msg)
                yield f"Error: {error_msg}"
                return
                
            # Si no hay mensajes, devolver error claro
            if not messages or len(messages) == 0:
                error_msg = "No hay mensajes para procesar en streaming."
                print(error_msg)
                yield f"Error: {error_msg}"
                return
            
            # Log all messages for debugging
            print(f"All messages to process in streaming:")
            for i, msg in enumerate(messages):
                print(f"  Message {i+1}: role={msg.get('role', 'unknown')}, content_preview={msg.get('content', '')[:50]}...")
            
            # Configure Google Gemini with API key
            genai.configure(api_key=self.google_api_key)
            
            # Convertir mensajes al formato de Google
            gemini_messages = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # Verificar que role y content no estén vacíos para evitar errores
                if not role or not content:
                    print(f"Advertencia: Mensaje con role o content vacío: {msg}")
                    continue
                
                # Map roles: user->user, assistant->model, system->user (with special prefix)
                if role == "user":
                    gemini_messages.append({"role": "user", "parts": [{"text": content}]})
                elif role == "assistant":
                    gemini_messages.append({"role": "model", "parts": [{"text": content}]})
                elif role == "system":
                    # System messages need to be sent as user messages with a special prefix
                    gemini_messages.append({"role": "user", "parts": [{"text": f"[SYSTEM INSTRUCTION] {content}"}]})
            
            # Verificar si hay mensajes para procesar
            if len(gemini_messages) == 0:
                error_msg = "No hay mensajes válidos para procesar en streaming después de la conversión."
                print(error_msg)
                yield f"Error: {error_msg}"
                return
            
            # Debug messages after conversion
            print(f"Gemini messages after conversion:")
            for i, msg in enumerate(gemini_messages):
                print(f"  Gemini message {i+1}: role={msg.get('role', 'unknown')}, parts_preview={msg.get('parts', [{}])[0].get('text', '')[:50] if msg.get('parts') else 'empty'}...")
            
            # Configure the Gemini model with safety settings
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 0.95,
                },
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    }
                ]
            )
            
            print(f"Sending messages in streaming mode: {gemini_messages}")
            
            # Get the last message to send to the model
            try:
                # If we have multiple messages, use chat mode
                if len(gemini_messages) > 1:
                    # Start a chat session with history
                    chat = model.start_chat(history=gemini_messages[:-1])
                    
                    # Send the last message and stream the response
                    last_msg = gemini_messages[-1]["parts"][0]["text"]
                    print(f"Streaming with chat history. Last message: {last_msg[:50]}...")
                    
                    stream = chat.send_message(last_msg, stream=True)
                    for chunk in stream:
                        if hasattr(chunk, 'text') and chunk.text:
                            yield chunk.text
                        elif hasattr(chunk, 'parts') and chunk.parts:
                            for part in chunk.parts:
                                if hasattr(part, 'text') and part.text:
                                    yield part.text
                
                # For single message conversations
                elif len(gemini_messages) == 1:
                    msg = gemini_messages[0]["parts"][0]["text"]
                    print(f"Streaming with single message: {msg[:50]}...")
                    
                    stream = model.generate_content(msg, stream=True)
                    for chunk in stream:
                        if hasattr(chunk, 'text') and chunk.text:
                            yield chunk.text
                        elif hasattr(chunk, 'parts') and chunk.parts:
                            for part in chunk.parts:
                                if hasattr(part, 'text') and part.text:
                                    yield part.text
            
            except Exception as e:
                error_str = str(e)
                print(f"Error in streaming: {error_str}")
                yield f"Error: {error_str}"
                        
            print("Google Gemini streaming completed")
                    
        except Exception as e:
            error_str = str(e)
            print(f"Error in Google Gemini streaming response: {error_str}")
            
            # Provide specific error message based on the type of error
            if "503" in error_str and "overloaded" in error_str:
                yield "Error: El modelo de Google Gemini está sobrecargado. Por favor, intente más tarde."
            elif "response.parts quick accessor only works for a single candidate, but none were returned" in error_str:
                yield "Error: Su pregunta fue bloqueada por los sistemas de seguridad de Google. Por favor, modifique su pregunta e intente de nuevo."
            elif "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in error_str:
                yield "Error: Su clave API de Google no tiene permisos suficientes para streaming."
            elif "list index out of range" in error_str:
                yield "Error: Se produjo un error al acceder a los mensajes."
            else:
                yield f"Error: {error_str}"
    
    # Additional methods for other providers can be added here
    # def _generate_anthropic_response(self, message, chat_history):
    #     # Implementation for Anthropic's Claude
    #     pass
    
    # def _generate_huggingface_response(self, message, chat_history):
    #     # Implementation for Hugging Face models
    #     pass 