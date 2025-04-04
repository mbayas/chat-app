import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Typography,
  Paper,
  CircularProgress,
  Divider,
  useTheme,
  Chip,
  Tooltip,
  Button,
} from '@mui/material';
import {
  Send as SendIcon,
  Menu as MenuIcon,
  AttachFile as AttachFileIcon,
  Cloud as CloudIcon,
  Computer as ComputerIcon,
  PictureAsPdf as PdfIcon,
  KeyboardArrowUp as KeyboardArrowUpIcon,
  Stop as StopIcon,
  AutoAwesome as AutoAwesomeIcon,
} from '@mui/icons-material';
import { useChat, Message } from '../contexts/ChatContext';
import { useSettings } from '../contexts/SettingsContext';
import MessageItem from './MessageItem';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { useProject } from '../contexts/ProjectContext';
import { styled } from '@mui/system';
import InputBar from './InputBar';

interface ChatWindowProps {
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}

interface ProjectSuggestionProps {
  isVisible: boolean;
  suggestions: string[];
  onSelect: (name: string) => void;
}

const SuggestionContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  bottom: '60px',
  left: '10px',
  maxHeight: '200px',
  width: 'calc(100% - 20px)',
  backgroundColor: '#1F2937',
  border: '1px solid #2a2e42',
  borderRadius: '8px',
  overflowY: 'auto',
  zIndex: 10,
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
}));

const SuggestionItem = styled(Box)(({ theme }) => ({
  padding: '8px 12px',
  cursor: 'pointer',
  '&:hover': {
    backgroundColor: 'rgba(204, 120, 50, 0.1)',
  },
}));

const ProjectSuggestion: React.FC<ProjectSuggestionProps> = ({ isVisible, suggestions, onSelect }) => {
  if (!isVisible || suggestions.length === 0) return null;
  
  return (
    <SuggestionContainer>
      {suggestions.map((name, index) => (
        <SuggestionItem key={index} onClick={() => onSelect(name)}>
          <Typography variant="body2" sx={{ color: '#e0e0e0' }}>
            {name}
          </Typography>
        </SuggestionItem>
      ))}
    </SuggestionContainer>
  );
};

const ChatWindow: React.FC<ChatWindowProps> = ({ sidebarOpen, toggleSidebar }) => {
  const { messages, sendMessage, loading, uploadFile, currentChat, streamingMessage, cancelStreaming, initialPrompt, setInitialPrompt } = useChat();
  const { settings } = useSettings();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();
  const [dragActive, setDragActive] = useState(false);
  const [exportingPdf, setExportingPdf] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  
  // Estado para el historial de prompts
  const [promptHistory, setPromptHistory] = useState<string[]>([]);
  const [promptHistoryIndex, setPromptHistoryIndex] = useState(-1);
  
  // NUEVA SOLUCIÓN: Mantener el último prompt enviado, independiente del historial
  const [lastSentPrompt, setLastSentPrompt] = useState<string>('');
  
  // Referencia directa al último mensaje enviado (para acceso síncrono)
  const lastMessageRef = useRef<string>('');
  
  // Estado para mantener el prefijo del proyecto si el último mensaje fue RAG
  const [lastProjectPrefix, setLastProjectPrefix] = useState<string | null>(null);
  // Estado para controlar si el usuario borró manualmente el prefijo
  const [userRemovedPrefix, setUserRemovedPrefix] = useState<boolean>(false);

  const { getAllProjectNames } = useProject();
  const [showProjectSuggestions, setShowProjectSuggestions] = useState(false);
  const [partialProjectName, setPartialProjectName] = useState('');
  
  const projectNames = useMemo(() => getAllProjectNames(), [getAllProjectNames]);
  
  const filteredSuggestions = useMemo(() => {
    if (!partialProjectName) return projectNames;
    return projectNames.filter(name => 
      name.toLowerCase().includes(partialProjectName.toLowerCase())
    );
  }, [partialProjectName, projectNames]);

  // Estado para controlar cuándo se debe hacer auto-scroll
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true);
  
  // Referencia para el temporizador de auto-scroll durante streaming
  const autoScrollTimerRef = useRef<NodeJS.Timeout | null>(null);

  const [lastScrollPosition, setLastScrollPosition] = useState<number>(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Check for initial prompt and set it to the input
  useEffect(() => {
    if (initialPrompt) {
      setInput(initialPrompt);
      // Clear the initial prompt after using it
      setInitialPrompt('');
    }
  }, [initialPrompt, setInitialPrompt]);

  // Cargar el historial de prompts al cambiar de chat
  useEffect(() => {
    if (currentChat) {
      // Cargar historial del chat actual desde localStorage
      const savedHistory = localStorage.getItem(`promptHistory_${currentChat}`);
      if (savedHistory) {
        try {
          const parsedHistory = JSON.parse(savedHistory);
          console.log("Historial cargado desde localStorage:", parsedHistory);
          setPromptHistory(parsedHistory);
        } catch (e) {
          console.error('Error loading prompt history:', e);
          setPromptHistory([]);
        }
      } else {
        // No hay historial guardado para este chat
        console.log("No hay historial guardado para este chat:", currentChat);
        setPromptHistory([]);
      }
      
      // También cargar el último prompt enviado
      const lastPrompt = localStorage.getItem(`lastPrompt_${currentChat}`);
      if (lastPrompt) {
        console.log("Último prompt cargado:", lastPrompt);
        setLastSentPrompt(lastPrompt);
        lastMessageRef.current = lastPrompt;
        // También guardar en window para acceso global
        (window as any).lastChatPrompt = lastPrompt;
      } else {
        setLastSentPrompt('');
        lastMessageRef.current = '';
        (window as any).lastChatPrompt = '';
      }
    } else {
      // No hay chat seleccionado, limpiar historial
      console.log("No hay chat seleccionado, limpiando historial");
      setPromptHistory([]);
      setLastSentPrompt('');
      lastMessageRef.current = '';
      (window as any).lastChatPrompt = '';
    }
    
    // Reiniciar el índice de historial
    setPromptHistoryIndex(-1);
  }, [currentChat]);

  // Guardar el historial en localStorage cuando cambie
  useEffect(() => {
    if (currentChat && promptHistory.length > 0) {
      localStorage.setItem(`promptHistory_${currentChat}`, JSON.stringify(promptHistory));
    }
  }, [promptHistory, currentChat]);

  // Asegurar que los mensajes nuevos sean visibles de inmediato
  useEffect(() => {
    // Detectar cuando se agrega un nuevo mensaje (usuario o asistente)
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      
      // Si es un nuevo mensaje del usuario, hacer scroll inmediato sin animación
      if (lastMessage.role === 'user') {
        console.log('Nuevo mensaje del usuario detectado. Scrolling inmediato.');
        scrollToBottom('auto');
        
        // Habilitar auto-scroll para la respuesta que vendrá
        setAutoScrollEnabled(true);
      }
    }
  }, [messages.length]);
  
  // Efecto especializado para el scrolling durante streaming
  useEffect(() => {
    // Si hay streaming activo y auto-scroll está habilitado
    if (streamingMessage && autoScrollEnabled) {
      console.log('Iniciando seguimiento de scroll para streaming');
      
      // Limpiar cualquier temporizador existente
      if (autoScrollTimerRef.current) {
        clearInterval(autoScrollTimerRef.current);
      }
      
      // Función para hacer scroll al mensaje más reciente
      const scrollToLatestMessage = () => {
        if (messagesContainerRef.current) {
          // Hacer scroll al final del contenedor de mensajes
          messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
          console.log('Scroll al final del contenedor durante streaming');
        }
      };
      
      // Hacer scroll inicial al último mensaje
      scrollToLatestMessage();
      
      // Configurar un intervalo para hacer scroll automático durante el streaming
      autoScrollTimerRef.current = setInterval(scrollToLatestMessage, 75);
      
      // Limpiar el intervalo cuando cambie el estado de streaming
      return () => {
        if (autoScrollTimerRef.current) {
          clearInterval(autoScrollTimerRef.current);
          autoScrollTimerRef.current = null;
        }
        console.log('Deteniendo seguimiento de scroll para streaming');
      };
    }
  }, [streamingMessage, autoScrollEnabled]);

  // Scroll to bottom when messages change with improved behavior
  useEffect(() => {
    // No hacemos scroll si el usuario ha scrolleado hacia arriba (más de 200px desde el fondo)
    if (messagesContainerRef.current) {
      const container = messagesContainerRef.current;
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 200;
      
      // Si el usuario scrolleó manualmente hacia arriba mientras hay streaming,
      // desactivar el auto-scroll hasta que vuelva a estar cerca del fondo
      if (!isNearBottom && streamingMessage) {
        setAutoScrollEnabled(false);
      }
      
      // Reactivar auto-scroll si el usuario ha vuelto cerca del fondo
      if (isNearBottom && !autoScrollEnabled) {
        setAutoScrollEnabled(true);
      }
      
      // Siempre hacer scroll si estamos cerca del fondo o si auto-scroll está activo
      if ((isNearBottom && streamingMessage) || (messages.length > 0 && !streamingMessage)) {
        scrollToBottom();
      }
      
      // Mostrar botón de scroll si no estamos cerca del fondo
      setShowScrollButton(!isNearBottom);
    }
  }, [messages, streamingMessage, autoScrollEnabled]);

  // Agregar un manejador de scroll para detectar cuando el usuario scrollea manualmente
  useEffect(() => {
    const handleScroll = () => {
      if (messagesContainerRef.current) {
        const container = messagesContainerRef.current;
        const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 200;
        
        // Actualizar el estado del botón de scroll
        setShowScrollButton(!isNearBottom);
        
        // Si el usuario scrollea manualmente hacia arriba durante el streaming,
        // desactivar el auto-scroll hasta que vuelva a estar cerca del fondo
        if (!isNearBottom && streamingMessage) {
          setAutoScrollEnabled(false);
        }
        
        // Reactivar auto-scroll si el usuario ha vuelto cerca del fondo
        if (isNearBottom && !autoScrollEnabled) {
          setAutoScrollEnabled(true);
        }
      }
    };

    const container = messagesContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, [streamingMessage, autoScrollEnabled]);

  // Event listener global para capturar la tecla arriba en un chat nuevo
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      // Solo procesar la tecla arriba
      if (e.key !== 'ArrowUp') {
        return;
      }
      
      // Solo procesar si es un input de texto o textarea
      const target = e.target as HTMLElement;
      if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
        return;
      }
      
      // Si el inputRef está vacío pero tenemos un último mensaje reciente
      if (input === '' && lastMessageRef.current) {
        console.log('Global listener: Recuperando último mensaje:', lastMessageRef.current);
        setInput(lastMessageRef.current);
        e.preventDefault();
      }
    };
    
    // Añadir listener global
    window.addEventListener('keydown', handleGlobalKeyDown, true);
    
    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown, true);
    };
  }, [input, currentChat]);

  // REFUERZO ESPECIAL: Detectar el primer mensaje del usuario 
  // y usarlo como fuente para el historial
  useEffect(() => {
    // Si hay mensajes pero el historial está vacío, extraer los mensajes del usuario
    if (messages.length > 0 && promptHistory.length === 0 && currentChat) {
      // Filtrar solo mensajes del usuario y obtener su contenido
      const userMessages = messages
        .filter(msg => msg.role === 'user')
        .map(msg => msg.content);
      
      if (userMessages.length > 0) {
        console.log("Recuperando mensajes del usuario desde el chat:", userMessages);
        
        // Actualizar el historial con estos mensajes
        setPromptHistory(userMessages);
        
        // Guardar también la referencia al último mensaje
        const lastUserMessage = userMessages[0];
        setLastSentPrompt(lastUserMessage);
        lastMessageRef.current = lastUserMessage;
        (window as any).lastChatPrompt = lastUserMessage;
        
        // También guardar en localStorage
        localStorage.setItem(`lastPrompt_${currentChat}`, lastUserMessage);
        localStorage.setItem(`promptHistory_${currentChat}`, JSON.stringify(userMessages));
      }
    }
  }, [messages, promptHistory.length, currentChat]);

  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    if (messagesEndRef.current) {
      // Intentar primero con scrollIntoView
      try {
        messagesEndRef.current.scrollIntoView({ 
          behavior: behavior,
          block: 'end'
        });
      } catch (error) {
        console.error('Error en scrollIntoView:', error);
        
        // Método alternativo si scrollIntoView falla
        if (messagesContainerRef.current) {
          messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
        }
      }
    } else if (messagesContainerRef.current) {
      // Si no tenemos la referencia al final, usar contenedor
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  const handleScrollButtonClick = () => {
    scrollToBottom();
    setAutoScrollEnabled(true);
  };

  // Mejorar la función scrollToLastUserMessage para que sea más precisa y efectiva
  const scrollToLastUserMessage = () => {
    console.log('Posicionando mensaje de usuario en parte superior');
    
    try {
      if (!messagesContainerRef.current) {
        console.error('Error: No se encontró el contenedor de mensajes');
        return;
      }
      
      // Primero hacer scroll hasta el final para asegurar que todos los mensajes están visibles
      scrollToBottom('auto');
      
      // Dar un pequeño tiempo para que el DOM se actualice
      setTimeout(() => {
        // Verificar nuevamente que el ref sigue siendo válido
        if (!messagesContainerRef.current) {
          console.error('Error: Contenedor de mensajes ya no disponible');
          return;
        }
        
        // Buscar todos los mensajes de usuario
        const userElements = document.querySelectorAll('[data-message-role="user"]');
        if (userElements.length === 0) {
          console.warn('No se encontraron mensajes del usuario');
          return;
        }
        
        // Obtener el último mensaje (el más reciente)
        const lastUserElement = userElements[userElements.length - 1] as HTMLElement;
        if (!lastUserElement) {
          console.warn('No se pudo acceder al elemento del mensaje');
          return;
        }
        
        // Verificar nuevamente el ref antes de acceder a sus propiedades
        if (!messagesContainerRef.current) {
          console.error('Error: Contenedor de mensajes ya no disponible');
          return;
        }
        
        // Obtener dimensiones para calcular la posición
        const containerRect = messagesContainerRef.current.getBoundingClientRect();
        const messageRect = lastUserElement.getBoundingClientRect();
        
        // Verificar una vez más antes de realizar la operación final
        if (!messagesContainerRef.current) {
          console.error('Error: Contenedor de mensajes ya no disponible');
          return;
        }
        
        // Calcular la posición para colocar el mensaje arriba con margen
        const newScrollPosition = messagesContainerRef.current.scrollTop + 
                                (messageRect.top - containerRect.top) - 20;
        
        // Aplicar scroll inmediatamente
        messagesContainerRef.current.scrollTo({
          top: newScrollPosition,
          behavior: 'auto'
        });
        
        console.log('Mensaje posicionado en parte superior con éxito');
      }, 50);
    } catch (error) {
      console.error('Error al intentar posicionar mensaje:', error);
    }
  };

  // Manejar el envío de mensajes
  const handleSend = async () => {
    if (input.trim() === '') return;
    
    const message = input.trim();
    
    // Guardar referencias y actualizar estado
    setLastSentPrompt(message);
    lastMessageRef.current = message;
    
    // Verificar si el mensaje es una consulta RAG
    const ragMatch = message.match(/^\*\*([^*]+)\*\*/);
    if (ragMatch) {
      const projectPrefix = `**${ragMatch[1]}**`;
      setLastProjectPrefix(projectPrefix);
      console.log(`RAG query detected. Saving project prefix: ${projectPrefix}`);
      
      // Guardar en localStorage si hay chat actual
      if (currentChat) {
        localStorage.setItem(`lastProjectPrefix_${currentChat}`, projectPrefix);
      }
    }
    
    // Guardar en localStorage
    if (currentChat) {
      localStorage.setItem(`lastPrompt_${currentChat}`, message);
    }
    
    // Guardar en window
    (window as any).lastChatPrompt = message;
    
    // Limpiar input antes de enviar
    setInput('');
    
    // Actualizar historial
    const newHistory = [...promptHistory];
    newHistory.unshift(message);
    
    // Limitar tamaño
    if (newHistory.length > 50) {
      newHistory.splice(50);
    }
    
    // Actualizar estado
    setPromptHistory(newHistory);
    
    // Guardar en localStorage
    if (currentChat) {
      localStorage.setItem(`promptHistory_${currentChat}`, JSON.stringify(newHistory));
    }
    
    // Resetear índice
    setPromptHistoryIndex(-1);
    
    // Activar auto-scroll
    setAutoScrollEnabled(true);
    
    try {
      // Enviar mensaje
      await sendMessage(message);
      
      // Aplicar scroll hacia el mensaje del usuario
      // Importante: usar un timeout suficientemente largo para asegurar que el DOM está actualizado
      setTimeout(scrollToLastUserMessage, 200);
    } catch (error) {
      console.error('Error al enviar mensaje:', error);
    }
  };
  
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };
  
  // SOLUCIÓN MEJORADA: Función para manejar las teclas arriba/abajo
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Ignorar teclas que no son flechas
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') {
      return;
    }
    
    e.preventDefault();
    
    if (e.key === 'ArrowUp') {
      // Estrategia 1: Buscar directamente en los mensajes si el historial está vacío
      if (promptHistory.length === 0 && messages.length > 0) {
        // Encontrar el primer mensaje del usuario
        const userMessage = messages.find(msg => msg.role === 'user')?.content;
        if (userMessage) {
          console.log("Recuperando desde los mensajes actuales:", userMessage);
          setInput(userMessage);
          return;
        }
      }
      
      // CASO ESPECIAL: Acceso directo al último mensaje enviado via refs
      if (input === '' && lastMessageRef.current) {
        console.log("Acceso directo al último mensaje:", lastMessageRef.current);
        setInput(lastMessageRef.current);
        return;
      }
      
      // Si el historial está vacío pero tenemos un último prompt, usarlo
      if (promptHistory.length === 0 && lastSentPrompt) {
        console.log("Mostrando último prompt enviado:", lastSentPrompt);
        setInput(lastSentPrompt);
        return;
      }
      
      // Intentar acceder al prompt global (solución de último recurso)
      if (promptHistory.length === 0 && input === '' && (window as any).lastChatPrompt) {
        console.log("Recuperando desde window.lastChatPrompt:", (window as any).lastChatPrompt);
        setInput((window as any).lastChatPrompt);
        return;
      }
      
      // Si el historial no está vacío y estamos en el índice inicial
      if (promptHistory.length > 0 && promptHistoryIndex === -1) {
        console.log("Mostrando primer prompt del historial:", promptHistory[0]);
        setPromptHistoryIndex(0);
        setInput(promptHistory[0]);
        return;
      }
      
      // Si ya estamos navegando por el historial
      if (promptHistory.length > 0 && promptHistoryIndex < promptHistory.length - 1) {
        const nextIndex = promptHistoryIndex + 1;
        console.log("Navegando historial, índice:", nextIndex, "prompt:", promptHistory[nextIndex]);
        setPromptHistoryIndex(nextIndex);
        setInput(promptHistory[nextIndex]);
        return;
      }
    }
    
    if (e.key === 'ArrowDown') {
      // Si no estamos en el historial, no hacer nada
      if (promptHistoryIndex === -1) {
        return;
      }
      
      // Si estamos en el primer elemento, volver al estado inicial
      if (promptHistoryIndex === 0) {
        console.log("Volviendo al estado inicial");
        setPromptHistoryIndex(-1);
        setInput('');
        return;
      }
      
      // Si estamos navegando por el historial
      if (promptHistoryIndex > 0) {
        const prevIndex = promptHistoryIndex - 1;
        console.log("Navegando historial hacia abajo, índice:", prevIndex);
        setPromptHistoryIndex(prevIndex);
        setInput(promptHistory[prevIndex]);
        return;
      }
    }
  };

  // File upload handling with react-dropzone
  const onDrop = async (acceptedFiles: File[]) => {
    setDragActive(false);
    if (acceptedFiles.length > 0) {
      await uploadFile(acceptedFiles[0]);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    noClick: true,
    noKeyboard: true,
  });

  // Update drag state
  useEffect(() => {
    setDragActive(isDragActive);
  }, [isDragActive]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await uploadFile(e.target.files[0]);
      // Reset the input
      e.target.value = '';
    }
  };

  // Función auxiliar para formatear el nombre del modelo para mostrar
  const formatModelName = (modelId: string) => {
    if (settings.provider === 'ollama') {
      return modelId;
    }
    
    // Para OpenAI, limpiar los nombres de los modelos
    const name = modelId.replace(/^gpt-/, '').replace(/-/g, ' ');
    return name.charAt(0).toUpperCase() + name.slice(1);
  };

  // Función para exportar el chat a PDF
  const handleExportPdf = async () => {
    if (!currentChat) return;
    
    try {
      setExportingPdf(true);
      
      // Hacer la solicitud para obtener el PDF
      const response = await axios.get(`/api/chats/${currentChat}/export-pdf`, {
        responseType: 'blob', // Importante: especificar que esperamos un blob
      });
      
      // Crear un objeto URL para el blob recibido
      const url = window.URL.createObjectURL(new Blob([response.data]));
      
      // Crear un elemento <a> temporal para descargar el archivo
      const link = document.createElement('a');
      link.href = url;
      
      // Obtener el nombre del archivo de la respuesta si está disponible, o generar uno
      const filename = response.headers['content-disposition']
        ? response.headers['content-disposition'].split('filename=')[1].replace(/"/g, '')
        : `chat_export_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.pdf`;
      
      link.setAttribute('download', filename);
      
      // Añadir el elemento al DOM, hacer clic en él y luego eliminarlo
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Liberar el objeto URL
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting chat to PDF:', error);
      alert('Hubo un error al exportar el chat a PDF. Por favor intenta nuevamente.');
    } finally {
      setExportingPdf(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const prevValue = input;
    setInput(value);
    
    // Detectar si el usuario borró el prefijo del proyecto
    if (lastProjectPrefix && 
        prevValue.startsWith(lastProjectPrefix) && 
        !value.startsWith(lastProjectPrefix) &&
        value.length < prevValue.length) {
      // El usuario ha borrado o modificado el prefijo manualmente
      console.log("Usuario borró el prefijo del proyecto. Desactivando auto-prefijo.");
      setUserRemovedPrefix(true);
    }
    
    // Check if we should show project suggestions (when user types **)
    if (value.endsWith('**')) {
      // User just completed a project name with ** at the end
      setShowProjectSuggestions(false);
    } else if (value.includes('**') && !value.endsWith('**')) {
      // User is typing a project name after **
      const startIndex = value.lastIndexOf('**') + 2;
      setPartialProjectName(value.substring(startIndex));
      setShowProjectSuggestions(true);
    } else {
      setShowProjectSuggestions(false);
    }
  };

  const handleSelectProject = (name: string) => {
    // We need to replace the partial project name with the selected full name
    const startIndex = input.lastIndexOf('**') + 2;
    const newInput = input.substring(0, startIndex) + name + '**';
    setInput(newInput);
    setShowProjectSuggestions(false);
  };

  // Efecto para establecer el prefijo del proyecto después de recibir una respuesta
  useEffect(() => {
    // Verificamos si tenemos un prefijo de proyecto guardado y si la respuesta ha terminado de cargar
    if (lastProjectPrefix && !loading && !streamingMessage && input === '' && !userRemovedPrefix) {
      // Establecer automáticamente el prefijo como el inicio del siguiente mensaje
      setInput(lastProjectPrefix + ' ');
    }
  }, [loading, streamingMessage, lastProjectPrefix, input, userRemovedPrefix]);

  // Efecto para resetear userRemovedPrefix cuando se envía un mensaje
  useEffect(() => {
    if (loading) {
      // Cuando se está enviando un mensaje (loading=true), resetear la bandera
      setUserRemovedPrefix(false);
    }
  }, [loading]);

  // Cargar el último prefijo de proyecto al cambiar de chat
  useEffect(() => {
    if (currentChat) {
      // Cargar último prefijo de proyecto desde localStorage
      const savedPrefix = localStorage.getItem(`lastProjectPrefix_${currentChat}`);
      if (savedPrefix) {
        setLastProjectPrefix(savedPrefix);
      } else {
        setLastProjectPrefix(null);
      }
      // Resetear la bandera al cambiar de chat
      setUserRemovedPrefix(false);
    }
  }, [currentChat]);

  // Modificar el useEffect para cuando cambian los mensajes (cuando se carga un chat existente)
  useEffect(() => {
    // Al seleccionar un chat o cargar mensajes, siempre hacemos scroll al final
    if (messages.length > 0) {
      // Para mensajes iniciales, usar 'auto' para que no haya animación
      scrollToBottom('auto');
    }
  }, [currentChat, messages.length]);

  // Detector de nuevos mensajes del usuario
  useEffect(() => {
    // Solo ejecutar si hay mensajes
    if (messages.length === 0) return;
    
    // Obtener el último mensaje
    const lastMessage = messages[messages.length - 1];
    
    // Si el último mensaje es del usuario, posicionarlo en la parte superior
    if (lastMessage.role === 'user') {
      console.log('Nuevo mensaje de usuario detectado - posicionando en parte superior');
      
      // Dar tiempo para que el DOM se actualice
      setTimeout(() => {
        // Primero asegurar que todos los mensajes son visibles
        scrollToBottom('auto');
        
        // Luego posicionar el último mensaje del usuario arriba
        setTimeout(scrollToLastUserMessage, 100);
      }, 200);
    }
  }, [messages.length]);

  // Añadir función para limpiar el contexto
  const handleClearContext = async () => {
    if (!currentChat) return;
    
    try {
      // Llamar a la API para borrar los mensajes del chat
      await axios.delete(`/api/chat/${currentChat}`);
      
      // Recargar la página para actualizar todo
      window.location.reload();
    } catch (error) {
      console.error('Error clearing context:', error);
      alert('Error al limpiar el contexto. Por favor intenta de nuevo.');
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        position: 'relative',
        backgroundColor: '#0a0c10',
      }}
    >
      {/* Barra superior con título y botones */}
      <Box sx={{ 
        display: 'flex', 
        p: 1,
        borderBottom: 'none',
        justifyContent: 'space-between',
        alignItems: 'center',
        position: 'relative',
        backgroundColor: '#0a0c10',
        zIndex: 1,
        width: '100%'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton onClick={toggleSidebar} sx={{ mr: 1 }}>
            <MenuIcon />
          </IconButton>
          <Tooltip title={`${
            settings.provider === 'openai' ? 'OpenAI' : 
            settings.provider === 'google' ? 'Google' : 'Ollama'
          }: ${formatModelName(settings.model)}`}>
            <Chip
              icon={
                settings.provider === 'openai' ? <CloudIcon fontSize="small" /> : 
                settings.provider === 'google' ? <AutoAwesomeIcon fontSize="small" /> : 
                <ComputerIcon fontSize="small" />
              }
              label={`${
                settings.provider === 'openai' ? 'OpenAI' : 
                settings.provider === 'google' ? 'Google' : 'Ollama'
              }: ${formatModelName(settings.model)}`}
              size="small"
              color={
                settings.provider === 'openai' ? 'primary' : 
                settings.provider === 'google' ? 'primary' : 'secondary'
              }
              sx={{ 
                mr: 1,
                backgroundColor: '#0d1117',
                border: 'none',
                '& .MuiChip-label': {
                  color: '#e0e0e0'
                }
              }}
            />
          </Tooltip>
          
          {/* Estado del contexto */}
          <Chip
            label={settings.maintain_context ? "Context ON" : "Context OFF"}
            size="small"
            variant="outlined"
            sx={{
              mr: 1,
              backgroundColor: '#0d1117',
              border: 'none',
              '& .MuiChip-label': {
                color: '#e0e0e0'
              }
            }}
          />

          {/* Botón para limpiar todo el contexto */}
          <Button 
            variant="outlined" 
            size="small"
            onClick={handleClearContext}
            sx={{ 
              minWidth: 'auto', 
              textTransform: 'none',
              bgcolor: '#0d1117',
              borderColor: 'grey.700',
              color: '#e0e0e0',
              '&:hover': {
                borderColor: 'grey.500',
                bgcolor: '#12151b'
              }
            }}
          >
            Clear Context
          </Button>
        </Box>

        <Box>
          {/* Botón para exportar a PDF */}
          <Tooltip title="Export as PDF">
            <span>
              <IconButton
                onClick={handleExportPdf}
                disabled={messages.length === 0 || exportingPdf}
                sx={{ color: exportingPdf ? 'grey.500' : 'inherit' }}
              >
                {exportingPdf ? <CircularProgress size={24} /> : <PdfIcon />}
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      </Box>

      {/* Message area */}
      <Box
        ref={messagesContainerRef}
        className="messages-container"
        sx={{
          flexGrow: 1,
          overflow: 'auto',
          position: 'relative',
          backgroundColor: '#0d1117',
          height: 'calc(100vh - 130px)', // Asegurar que ocupa el espacio disponible
          scrollBehavior: 'smooth', // Hacer scroll más suave para todos los métodos
        }}
        {...getRootProps()}
      >
        {/* Hidden file input */}
        <input {...getInputProps()} />

        {/* Welcome message when no messages */}
        {messages.length === 0 && (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              textAlign: 'center',
              px: 4,
            }}
          >
            <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 400 }}>
              <Box component="span" sx={{ color: '#e0e0e0' }}>Welcome to{' '}
                <Box component="span" sx={{ color: '#CC7832', display: 'inline' }}>G</Box>
                ep
                <Box component="span" sx={{ color: '#CC7832', display: 'inline' }}>p</Box>
                et
                <Box component="span" sx={{ color: '#CC7832', display: 'inline' }}>t</Box>
                o
              </Box>
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ maxWidth: '600px' }}>
              Start a conversation by typing a message below. Your chat history will appear here.
            </Typography>
          </Box>
        )}

        {/* Files drag overlay */}
        {dragActive && (
          <Box
            sx={{
              position: 'absolute',
              width: '100%',
              height: '100%',
              top: 0,
              right: 0,
              bottom: 0,
              left: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1,
            }}
            onDragEnter={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setDragActive(false);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setDragActive(false);
            }}
          >
            <AttachFileIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" sx={{ mb: 1 }}>
              Arrastra archivos aquí
            </Typography>
            <Typography variant="body2">
              Formatos soportados: .txt, .pdf, .csv, etc.
            </Typography>
          </Box>
        )}

        {/* Messages */}
        {messages.map((message) => (
          <MessageItem key={`${message.role}-${message.timestamp}`} message={message} />
        ))}
        
        {/* Reference div for scrolling to the bottom */}
        <div ref={messagesEndRef} />
        
        {/* Botón flotante para desplazarse hacia abajo cuando hay contenido fuera de pantalla */}
        {showScrollButton && (
          <IconButton
            onClick={handleScrollButtonClick}
            size="medium"
            sx={{
              position: 'absolute',
              bottom: 20,
              right: 20,
              backgroundColor: 'rgba(204, 120, 50, 0.8)',
              color: 'white',
              '&:hover': {
                backgroundColor: 'rgba(204, 120, 50, 1)',
              },
              zIndex: 2,
              boxShadow: '0 3px 5px rgba(0, 0, 0, 0.3)',
            }}
          >
            <KeyboardArrowUpIcon sx={{ transform: 'rotate(180deg)' }} />
          </IconButton>
        )}
      </Box>

      {/* Input area at the bottom */}
      <Box
        sx={{
          position: 'sticky',
          bottom: 0,
          width: '100%',
          p: 2,
          backgroundColor: '#0a0c10'
        }}
      >
        <InputBar
          input={input}
          setInput={setInput}
          handleSend={handleSend}
          handleKeyDown={handleKeyDown}
          inputRef={inputRef}
          disabled={loading || streamingMessage}
          placeholder={lastSentPrompt ? "Send a follow-up..." : "Type a message..."}
          onFileUpload={handleFileSelect}
          isDragActive={isDragActive}
        />
      </Box>

      {showProjectSuggestions && (
        <ProjectSuggestion
          isVisible={showProjectSuggestions}
          suggestions={filteredSuggestions}
          onSelect={handleSelectProject}
        />
      )}
    </Box>
  );
};

export default ChatWindow; 