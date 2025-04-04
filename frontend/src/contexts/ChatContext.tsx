import React, { createContext, useState, useEffect, useContext, ReactNode, useRef } from 'react';
import axios from 'axios';

// Define types
export interface Message {
  id?: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  isStreaming?: boolean;  // Indica si el mensaje está en proceso de streaming
  responseTimeMs?: number; // Tiempo de respuesta en milisegundos
  model?: string;  // Modelo usado para el mensaje
  provider?: string;  // Proveedor usado para el mensaje (openai o ollama)
  endTimestamp?: string; // Timestamp cuando se completa el mensaje (para calcular tiempo total)
  tokenCount?: number;  // Número exacto de tokens en la respuesta
  totalElapsedTime?: number; // Tiempo total transcurrido en ms (para mantener consistencia al navegar entre chats)
  timePerToken?: number; // Tiempo por token en ms (para calcular tiempo de generación)
  tokensPerSecond?: number; // Nueva métrica: tokens por segundo
  wordsPerSecond?: number; // Nueva métrica: palabras por segundo
  wordCount?: number; // Número total de palabras en la respuesta
}

export interface Chat {
  id: string;
  title: string;
  created_at: string;
  last_message: string;
}

interface ChatContextType {
  chats: Chat[];
  currentChat: string | null;
  messages: Message[];
  loading: boolean;
  streamingMessage: boolean;  // Indica si hay un mensaje en streaming actualmente
  sendMessage: (message: string) => Promise<void>;
  createNewChat: () => void;
  selectChat: (chatId: string) => Promise<boolean>;
  deleteChat: (chatId: string) => Promise<void>;
  uploadFile: (file: File) => Promise<void>;
  cancelStreaming: () => void; // Nueva función para cancelar streaming
  initialPrompt: string; // New variable to store initial prompt
  setInitialPrompt: (prompt: string) => void; // New function to set initial prompt
}

// Claves para localStorage
const CURRENT_CHAT_KEY = 'bayteq_gpt_current_chat';

// Create context
const ChatContext = createContext<ChatContextType | undefined>(undefined);

// Provider component
export const ChatProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChat, setCurrentChat] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [streamingMessage, setStreamingMessage] = useState<boolean>(false);
  const [cancelController, setCancelController] = useState<AbortController | null>(null);
  const [initialPrompt, setInitialPrompt] = useState<string>(''); // New state for initial prompt

  // Implementar un sistema de throttling para actualizar la interfaz más frecuentemente
  const streamUpdateInterval = useRef<NodeJS.Timeout | null>(null);
  const currentStreamContent = useRef<string>('');
  const currentStreamId = useRef<string | null>(null);
  
  // Función para actualizar el contenido de un mensaje específico
  const updateMessageContent = (messageId: string, content: string) => {
    setMessages(prev => {
      const newMessages = [...prev];
      const index = newMessages.findIndex(m => m.id === messageId);
      
      if (index !== -1) {
        // Reemplazar el mensaje completo para evitar problemas de renderizado
        newMessages[index] = {
          ...newMessages[index],
          content: content,
          isStreaming: true
        };
      }
      
      return newMessages;
    });
  };
  
  // Función para manejar actualizaciones más frecuentes durante streaming
  const throttledStreamUpdate = (content: string, messageId: string) => {
    // Guardar el contenido actual
    currentStreamContent.current = content;
    currentStreamId.current = messageId;
    
    // Si no hay un intervalo activo, crear uno
    if (!streamUpdateInterval.current) {
      // Actualizar la UI inmediatamente con el primer fragmento
      updateMessageContent(messageId, content);
      
      // Forzar scroll hacia el final para ver el contenido nuevo
      setTimeout(() => {
        const messagesContainer = document.querySelector('.messages-container');
        if (messagesContainer) {
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
          console.log('Forzando scroll inicial al contenido nuevo');
        }
      }, 50);
      
      // Configurar un intervalo para actualizar más frecuentemente
      streamUpdateInterval.current = setInterval(() => {
        if (currentStreamId.current && currentStreamContent.current) {
          updateMessageContent(currentStreamId.current, currentStreamContent.current);
          
          // Forzar scroll hacia el final en cada actualización
          setTimeout(() => {
            const messagesContainer = document.querySelector('.messages-container');
            if (messagesContainer) {
              messagesContainer.scrollTop = messagesContainer.scrollHeight;
              console.log('Forzando scroll al contenido nuevo durante streaming');
            }
          }, 10);
        }
      }, 30); // Actualizar cada 30ms para un scroll más suave
    }
  };
  
  // Limpiar el intervalo cuando se complete el streaming
  const clearStreamInterval = () => {
    if (streamUpdateInterval.current) {
      clearInterval(streamUpdateInterval.current);
      streamUpdateInterval.current = null;
    }
    currentStreamContent.current = '';
    currentStreamId.current = null;
  };

  // Función para guardar el chat actual en localStorage
  const saveCurrentChatToStorage = (chatId: string | null) => {
    if (chatId) {
      console.log('Guardando chat en localStorage:', chatId);
      localStorage.setItem(CURRENT_CHAT_KEY, chatId);
    } else {
      console.log('Eliminando chat de localStorage');
      localStorage.removeItem(CURRENT_CHAT_KEY);
    }
  };

  // Función para cargar el chat actual desde localStorage
  const loadCurrentChatFromStorage = (): string | null => {
    const savedChat = localStorage.getItem(CURRENT_CHAT_KEY);
    console.log('Chat cargado desde localStorage:', savedChat);
    return savedChat;
  };

  // Fetch chats on component mount and try to restore the previous session
  useEffect(() => {
    const initializeChats = async () => {
      try {
        // Primero obtenemos la lista de chats
        const response = await axios.get('/api/chats');
        const chatsList = response.data;
        setChats(chatsList);
        
        // Luego intentamos cargar el chat anterior desde localStorage
        const savedChatId = loadCurrentChatFromStorage();
        
        if (savedChatId) {
          // Verificar si el chat existe en la lista de chats cargados
          const chatExists = chatsList.some((chat: Chat) => chat.id === savedChatId);
          
          if (chatExists) {
            console.log('Restoring previous chat session:', savedChatId);
            // Intentamos seleccionar el chat y manejamos el resultado
            const success = await selectChat(savedChatId);
            if (!success) {
              console.error('Failed to restore chat session, creating new chat');
              setCurrentChat(null);
              setMessages([]);
              saveCurrentChatToStorage(null);
            }
          } else {
            // Si el chat no existe, limpiar localStorage
            console.log('Saved chat no longer exists, creating new chat');
            saveCurrentChatToStorage(null);
            setCurrentChat(null);
            setMessages([]);
          }
        }
      } catch (error) {
        console.error('Error initializing chats:', error);
        // En caso de error, asegurarnos de que la aplicación siga funcionando
        saveCurrentChatToStorage(null);
        setCurrentChat(null);
        setMessages([]);
      }
    };

    initializeChats();
  }, []);

  // Asegurarse de guardar el chat actual cuando el usuario cierre o recargue la página
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Si hay un chat actual, asegurarnos de que esté guardado
      if (currentChat) {
        console.log('Guardando chat antes de cerrar:', currentChat);
        saveCurrentChatToStorage(currentChat);
      }
    };

    // Agregar el event listener
    window.addEventListener('beforeunload', handleBeforeUnload);

    // Limpieza
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [currentChat]);

  // Modificamos este useEffect para que no sobreescriba lo que hicimos en el inicializador
  useEffect(() => {
    if (currentChat && currentChat !== loadCurrentChatFromStorage()) {
      fetchMessages(currentChat);
      // Guardar en localStorage
      saveCurrentChatToStorage(currentChat);
    } else if (!currentChat) {
      setMessages([]);
      // Limpiar localStorage si no hay chat actual
      saveCurrentChatToStorage(null);
    }
  }, [currentChat]);

  // Fetch all chats
  const fetchChats = async () => {
    try {
      const response = await axios.get('/api/chats');
      setChats(response.data);
      return response.data;
    } catch (error) {
      console.error('Error fetching chats:', error);
      return [];
    }
  };

  // Fetch messages for a specific chat
  const fetchMessages = async (chatId: string) => {
    try {
      setLoading(true);
      const response = await axios.get(`/api/chat/${chatId}`);
      
      // Process messages to ensure all assistant messages have required properties
      const processedMessages = response.data.map((msg: Message) => {
        if (msg.role === 'assistant') {
          // Add default values for missing properties in assistant messages
          return {
            ...msg,
            responseTimeMs: msg.responseTimeMs || 0,
            model: msg.model || 'unknown',
            provider: msg.provider || 'unknown',
            tokenCount: msg.tokenCount || 0,
            totalElapsedTime: msg.totalElapsedTime || 0,
            timePerToken: msg.timePerToken || 0,
            tokensPerSecond: msg.tokensPerSecond || 0,
            wordCount: msg.wordCount || 0,
            wordsPerSecond: msg.wordsPerSecond || 0
          };
        }
        return msg;
      });
      
      setMessages(processedMessages);
      return processedMessages;
    } catch (error) {
      console.error('Error fetching messages:', error);
      setMessages([]);
      return [];
    } finally {
      setLoading(false);
    }
  };

  // Update message metrics in the backend
  const updateMessageMetrics = async (chatId: string, updatedMessage: Message) => {
    try {
      // First get all current messages
      const currentMessages = await fetchMessages(chatId);
      
      // Find and update the specified message
      const updatedMessages = currentMessages.map((msg: Message) => 
        msg.id === updatedMessage.id ? updatedMessage : msg
      );
      
      // Send the updated messages to the backend
      await axios.post(`/api/chat/${chatId}/update-messages`, updatedMessages);
      console.log('Message metrics updated successfully');
    } catch (error) {
      console.error('Error updating message metrics:', error);
    }
  };

  // Generate a title for the chat based on the initial message
  const generateChatTitle = (message: string): string => {
    // Limit to first 30 characters and add ellipsis if longer
    if (message.length <= 30) {
      return message;
    }
    
    // Try to find a natural break point (space, period, comma, etc.)
    const breakPoints = [' ', '.', ',', ':', ';', '!', '?'];
    
    // Look for the closest break point before the 30 character limit
    let lastBreak = 0;
    for (let i = 0; i < message.length && i < 30; i++) {
      if (breakPoints.includes(message[i])) {
        lastBreak = i;
      }
    }
    
    // If we found a good break point, use it
    if (lastBreak > 10) {
      return message.substring(0, lastBreak) + '...';
    }
    
    // Otherwise just cut at 30 characters
    return message.substring(0, 30) + '...';
  };

  // Send a message
  const sendMessage = async (message: string) => {
    // Generar IDs temporales para los mensajes
    const tempUserMessageId = `temp-user-${Date.now()}`;
    const tempAssistantMessageId = `temp-assistant-${Date.now()}`;
    
    try {
      const settingsResponse = await axios.get('/api/settings');
      const currentSettings = settingsResponse.data;
      
      // Agregar el mensaje del usuario inmediatamente
      const userMessage: Message = {
        id: tempUserMessageId,
        role: 'user',
        content: message,
        timestamp: new Date().toISOString(),
        model: currentSettings.model,
        provider: currentSettings.provider
      };
      
      // Agregar un mensaje vacío del asistente con indicador de streaming
      const initialAssistantMessage: Message = {
        id: tempAssistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isStreaming: true,
        model: currentSettings.model,
        provider: currentSettings.provider
      };
      
      // Actualizar los mensajes con el mensaje del usuario y el mensaje vacío del asistente
      setMessages(prevMessages => [...prevMessages, userMessage, initialAssistantMessage]);
      
      // Si no hay chat actual, crear uno nuevo
      let chatId = currentChat;
      if (!chatId) {
        try {
          const newChat = await axios.post('/api/chats', {
            title: generateChatTitle(message)
          });
          chatId = newChat.data.id;
          setCurrentChat(chatId);
          saveCurrentChatToStorage(chatId);
          
          // Actualizar la lista de chats
          const updatedChats = await fetchChats();
          setChats(updatedChats);
          
          // Esperar un momento para asegurar que el estado se actualice
          await new Promise(resolve => setTimeout(resolve, 100));
        } catch (error) {
          console.error('Error creating new chat:', error);
          throw error;
        }
      }
      
      // Usar el endpoint de streaming
      setStreamingMessage(true);
      await streamChatMessage(message, chatId, tempAssistantMessageId);
      setStreamingMessage(false);
      
    } catch (error) {
      console.error('Error sending message:', error);
      // Actualizar el mensaje del asistente con el error
      setMessages(prevMessages => {
        const newMessages = [...prevMessages];
        const assistantMessageIndex = newMessages.findIndex(m => m.id === tempAssistantMessageId);
        if (assistantMessageIndex !== -1) {
          newMessages[assistantMessageIndex] = {
            ...newMessages[assistantMessageIndex],
            content: 'Lo siento, hubo un error al procesar tu mensaje. Por favor, intenta nuevamente.',
            isStreaming: false
          };
        }
        return newMessages;
      });
    }
  };

  // Función para cancelar el streaming
  const cancelStreaming = () => {
    if (cancelController) {
      console.log('Cancelando streaming por solicitud del usuario');
      
      // Primero abortar la conexión del lado del cliente
      cancelController.abort();
      setCancelController(null);
      
      // Si hay un chat actual, notificar al servidor para que también detenga el procesamiento
      if (currentChat) {
        try {
          // Llamar al endpoint de cancelación en el servidor
          axios.post(`/api/chat/stream/cancel/${currentChat}`)
            .then(response => {
              console.log('Cancelación de streaming notificada al servidor', response.data);
            })
            .catch(error => {
              console.error('Error al notificar cancelación al servidor:', error);
            });
        } catch (error) {
          console.error('Error al intentar cancelar el streaming en el servidor:', error);
        }
      }
      
      // Limpiar el intervalo si está usando throttling
      clearStreamInterval();
      
      // Actualizar el estado
      setStreamingMessage(false);
      
      // Actualizar el mensaje para indicar que fue cancelado por el usuario
      setMessages(prevMessages => {
        return prevMessages.map(msg => {
          if (msg.isStreaming) {
            return {
              ...msg,
              isStreaming: false,
              content: msg.content + " [Cancelado por el usuario]",
              endTimestamp: new Date().toISOString()
            };
          }
          return msg;
        });
      });
    }
  };

  // Stream chat message using Server-Sent Events
  const streamChatMessage = async (message: string, chatId: string | null, tempAssistantMessageId: string) => {
    console.log('Initiating streaming chat message');
    
    // Crear un nuevo controlador de cancelación
    const controller = new AbortController();
    setCancelController(controller);
    
    // Calculate API endpoint
    const apiEndpoint = '/api/chat/stream';
    const payload = chatId ? { message, chat_id: chatId } : { message };
    
    console.log(`Sending request to ${apiEndpoint} with payload:`, payload);
    
    // Registrar el tiempo de inicio para el tiempo de respuesta
    const streamStartTime = Date.now();
    
    try {
      // Primero, hacer la solicitud POST para iniciar el streaming
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body reader available');
      }
      
      console.log('Stream connection established successfully');
      
      const decoder = new TextDecoder();
      
      let responseTimeMs: number | null = null;
      let responseContent = '';
      let modelName = ''; 
      let providerName = '';
      let tokenCount = 0;
      let receivedFirstToken = false;
        
      // Procesar el stream
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            console.log('Stream done, total content received:', responseContent.length);
            break;
          }
          
          // Decodificar los datos recibidos
          const chunk = decoder.decode(value, { stream: true });
          console.log('Received chunk:', chunk);
          
          // Procesar cada línea por separado (formato Server-Sent Events)
          const lines = chunk.split('\n\n');
          
          for (const line of lines) {
            if (!line.trim()) continue;
            
            // Procesar líneas con formato "event: tipo\ndata: {...}"
            const parts = line.split('\n');
            if (parts.length >= 2 && parts[0].startsWith('event:') && parts[1].startsWith('data:')) {
              const eventType = parts[0].substring(7).trim();
              const jsonData = parts[1].substring(5).trim();
              
              try {
                const data = JSON.parse(jsonData);
                console.log(`Received ${eventType} event:`, data);
                
                switch (eventType) {
                  case 'metadata':
                    if (data.provider) providerName = data.provider;
                    if (data.model) modelName = data.model;
                    break;
                    
                  case 'token':
                    // Si es el primer token, registrar tiempo de respuesta
                    if (!receivedFirstToken) {
                      responseTimeMs = data.response_time_ms || (Date.now() - streamStartTime);
                      receivedFirstToken = true;
                      console.log(`First token received. Response time: ${responseTimeMs}ms`);
                    }
                    
                    // Añadir token al contenido acumulado
                    responseContent += data.token;
                    tokenCount++;
                    
                    // Actualizar la UI con el contenido actual
                    throttledStreamUpdate(responseContent, tempAssistantMessageId);
                    break;
                    
                  case 'completion':
                    console.log('Received completion event, finishing stream');
                    break;
                    
                  case 'error':
                    console.error('Server returned error:', data.error);
                    responseContent = "Error: " + data.error;
                    throttledStreamUpdate(responseContent, tempAssistantMessageId);
                    break;
                }
              } catch (error) {
                console.error('Error parsing event data:', error, jsonData);
              }
            }
            // Formato antiguo: "data: {...}"
            else if (line.startsWith('data:')) {
              try {
                const jsonStr = line.substring(5).trim();
                const data = JSON.parse(jsonStr);
                
                // Para compatibilidad con el formato anterior
                if (data.type === 'metadata') {
                  if (data.provider) providerName = data.provider;
                  if (data.model) modelName = data.model;
                } else if (data.type === 'token') {
                  if (!receivedFirstToken) {
                    responseTimeMs = data.response_time_ms || (Date.now() - streamStartTime);
                    receivedFirstToken = true;
                  }
                  responseContent += data.token;
                  tokenCount++;
                  throttledStreamUpdate(responseContent, tempAssistantMessageId);
                }
              } catch (error) {
                console.error('Error parsing data event:', error, line);
              }
            }
          }
        }
      } catch (error: any) {
        if (error.name === 'AbortError') {
          console.log('Stream cancelled by user');
        } else {
          console.error('Error reading stream:', error);
        }
      } finally {
        clearStreamInterval();
      }
      
      // Update final message once streaming is complete
      const totalElapsedTime = Date.now() - streamStartTime;
      // Calculate metrics
      const tokensPerSecond = tokenCount > 0 ? (tokenCount / (totalElapsedTime / 1000)) : 0;
      const content = responseContent.trim();
      const words = content.split(/\s+/).filter(Boolean);
      const wordCount = words.length;
      const wordsPerSecond = wordCount > 0 ? (wordCount / (totalElapsedTime / 1000)) : 0;
      
      const endTime = new Date().toISOString();
      
      // Final update to message when streaming is complete
      setMessages(prevMessages => {
        return prevMessages.map(msg => {
          if (msg.id === tempAssistantMessageId) {
            return {
              ...msg,
              content: responseContent,
              isStreaming: false,
              responseTimeMs: responseTimeMs || 0,
              endTimestamp: endTime,
              totalElapsedTime: totalElapsedTime,
              model: modelName || msg.model,
              provider: providerName || msg.provider,
              tokenCount: tokenCount,
              timePerToken: tokenCount > 0 ? (totalElapsedTime / tokenCount) : 0,
              tokensPerSecond: tokensPerSecond,
              wordCount: wordCount,
              wordsPerSecond: wordsPerSecond
            };
          }
          return msg;
        });
      });
      
      // Asegurarse de tener el chat actualizado
      if (chatId) {
        fetchChats();
      }
      
    } catch (error) {
      console.error('Error in streaming chat:', error);
      
      // Update message with error
      setMessages(prevMessages => {
        return prevMessages.map(msg => {
          if (msg.id === tempAssistantMessageId) {
            return {
              ...msg,
              content: 'Error: No se pudo obtener respuesta del servidor.',
              isStreaming: false,
              endTimestamp: new Date().toISOString()
            };
          }
          return msg;
        });
      });
    } finally {
      // Clean up
      setCancelController(null);
      setStreamingMessage(false);
      clearStreamInterval();
    }
  };

  // Create a new chat
  const createNewChat = () => {
    setCurrentChat(null);
    setMessages([]);
    // Limpiar el chat actual de localStorage
    saveCurrentChatToStorage(null);
  };

  // Select a chat
  const selectChat = async (chatId: string) => {
    try {
      console.log('Seleccionando chat:', chatId);
      // Primero verificamos si el chat existe en el servidor
      const response = await axios.get(`/api/chats/${chatId}`);
      if (response.status === 200) {
        setCurrentChat(chatId);
        // Guardar en localStorage
        saveCurrentChatToStorage(chatId);
        // Cargar mensajes del chat
        await fetchMessages(chatId);
        return true;
      } else {
        console.error('Error al seleccionar chat: No se encontró el chat');
        return false;
      }
    } catch (error) {
      console.error('Error al seleccionar chat:', error);
      return false;
    }
  };

  // Delete a chat
  const deleteChat = async (chatId: string) => {
    try {
      await axios.delete(`/api/chats/${chatId}`);
      
      // If the deleted chat is the current chat, create a new chat
      if (chatId === currentChat) {
        createNewChat();
      }
      
      // Refresh chat list
      fetchChats();
    } catch (error) {
      console.error('Error deleting chat:', error);
    }
  };

  // Upload a file
  const uploadFile = async (file: File) => {
    if (!currentChat) {
      // Create a new chat first if there's no current chat
      const response = await axios.post('/api/chat', {
        message: 'Uploaded a file',
      });
      setCurrentChat(response.data.chat_id);
      // Guardar en localStorage
      saveCurrentChatToStorage(response.data.chat_id);
      setMessages(response.data.history);
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('chat_id', currentChat!);

    try {
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      // Send a message about the uploaded file
      await sendMessage(`I've uploaded a file: ${file.name}`);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <ChatContext.Provider
      value={{
        chats,
        currentChat,
        messages,
        loading,
        streamingMessage,
        sendMessage,
        createNewChat,
        selectChat,
        deleteChat,
        uploadFile,
        cancelStreaming,
        initialPrompt,
        setInitialPrompt
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

// Custom hook to use the chat context
export const useChat = () => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}; 