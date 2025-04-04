import React, { useMemo, useCallback, useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, useTheme, keyframes, CircularProgress } from '@mui/material';
import { Message } from '../contexts/ChatContext';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useSettings } from '../contexts/SettingsContext';

// Definir animaciones
const blink = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
`;

const fadeInOut = keyframes`
  0% { opacity: 0.3; }
  50% { opacity: 1; }
  100% { opacity: 0.3; }
`;

// Agregar una clase keyframe para el cursor parpadeante
const cursorBlink = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
`;

// Animación para los puntos de carga
const dotAnimation1 = keyframes`
  0%, 100% { opacity: 0.3; }
  25% { opacity: 1; }
`;

const dotAnimation2 = keyframes`
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
`;

const dotAnimation3 = keyframes`
  0%, 100% { opacity: 0.3; }
  75% { opacity: 1; }
`;

interface MessageItemProps {
  message: Message;
}

const MessageItem: React.FC<MessageItemProps> = ({ message }) => {
  const theme = useTheme();
  const { settings } = useSettings();
  const isUser = message.role === 'user';
  const isStreaming = message.isStreaming;
  const contentRef = useRef<HTMLDivElement>(null);

  // Evitamos re-renders innecesarios durante el streaming
  const [renderKey, setRenderKey] = useState(0);
  
  // Optimización: solo re-renderizamos el componente completo cuando se complete el streaming
  // o cada cierto número de caracteres para evitar problemas de rendimiento
  useEffect(() => {
    if (!isStreaming) {
      // Si no está en streaming, forzamos un re-render
      setRenderKey(prev => prev + 1);
    } else {
      // Para streaming, actualizamos más frecuentemente (cada ~40 caracteres)
      const length = message.content?.length || 0;
      if (length % 40 === 0 && length > 0) {
        setRenderKey(prev => prev + 1);
      }
    }
  }, [isStreaming, message.content]);

  // Optimizar la manipulación directa del DOM para mejor feedback visual
  useEffect(() => {
    if (isStreaming && contentRef.current) {
      try {
        // Asegurarnos que tenemos un solo nodo de texto durante streaming
        if (contentRef.current.childNodes.length > 1) {
          // Si hay más de un nodo hijo, limpiar y empezar de nuevo
          while (contentRef.current.firstChild) {
            contentRef.current.removeChild(contentRef.current.firstChild);
          }
        }
        
        // Actualizar atributos de datos en el contenedor para facilitar selección vía CSS
        const messageContainer = contentRef.current.closest('[data-message-role="assistant"]');
        if (messageContainer) {
          messageContainer.setAttribute('data-is-streaming', 'true');
        }
        
        // Si no hay contenido o el contenedor está vacío, crear el contenedor de streaming
        if (message.content && (!contentRef.current.firstChild || contentRef.current.firstChild.nodeType === Node.TEXT_NODE)) {
          // Limpiar cualquier texto plano anterior
          while (contentRef.current.firstChild) {
            contentRef.current.removeChild(contentRef.current.firstChild);
          }
          
          // Crear un div para contener el texto formateado
          const formattedContainer = document.createElement('div');
          formattedContainer.className = 'streaming-content';
          formattedContainer.style.whiteSpace = 'pre-wrap'; // Respetar saltos de línea
          formattedContainer.style.wordBreak = 'break-word';
          formattedContainer.style.fontSize = '0.95rem';
          formattedContainer.style.lineHeight = '1.5';
          
          // Procesar el texto para formatear código inline
          let formattedContent = message.content;
          
          // Detectar bloques de código
          formattedContent = formattedContent.replace(/```([a-z]*)\n([\s\S]*?)```/g, (match, language, code) => {
            return `<div style="background-color: #0F172A; border-radius: 4px; padding: 8px; margin: 8px 0; overflow-x: auto; font-family: monospace;">${code}</div>`;
          });
          
          // Detectar código inline
          formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code style="background-color: #0F172A; border-radius: 3px; padding: 2px 4px; font-family: monospace;">$1</code>');
          
          // Convertir URLs en links
          formattedContent = formattedContent.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" style="color: #CC7832; text-decoration: none;" target="_blank">$1</a>');
          
          // Convertir saltos de línea en <br>
          formattedContent = formattedContent.replace(/\n/g, '<br>');
          
          // Establecer el contenido HTML
          formattedContainer.innerHTML = formattedContent;
          
          // Agregar al DOM
          contentRef.current.appendChild(formattedContainer);
          
          // Agregar cursor al final
          const cursorElement = document.createElement('span');
          cursorElement.className = 'streaming-cursor';
          cursorElement.style.display = 'inline-block';
          cursorElement.style.width = '0.5em';
          cursorElement.style.height = '1em';
          cursorElement.style.backgroundColor = '#CC7832'; // Color naranja más visible
          cursorElement.style.verticalAlign = 'middle';
          cursorElement.style.marginLeft = '2px';
          cursorElement.style.animation = 'blink 0.7s step-end infinite';
          
          // Agregar el cursor al final del texto
          contentRef.current.appendChild(cursorElement);
        } 
        // Si ya existe un contenedor de streaming, actualizar su contenido
        else if (message.content && contentRef.current.querySelector('.streaming-content')) {
          const formattedContainer = contentRef.current.querySelector('.streaming-content');
          if (formattedContainer) {
            // Procesar el texto para formatear código inline
            let formattedContent = message.content;
            
            // Detectar bloques de código
            formattedContent = formattedContent.replace(/```([a-z]*)\n([\s\S]*?)```/g, (match, language, code) => {
              return `<div style="background-color: #0F172A; border-radius: 4px; padding: 8px; margin: 8px 0; overflow-x: auto; font-family: monospace;">${code}</div>`;
            });
            
            // Detectar código inline
            formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code style="background-color: #0F172A; border-radius: 3px; padding: 2px 4px; font-family: monospace;">$1</code>');
            
            // Convertir URLs en links
            formattedContent = formattedContent.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" style="color: #CC7832; text-decoration: none;" target="_blank">$1</a>');
            
            // Convertir saltos de línea en <br>
            formattedContent = formattedContent.replace(/\n/g, '<br>');
            
            // Establecer el contenido HTML
            formattedContainer.innerHTML = formattedContent;
          }
        }
      } catch (error) {
        console.error('Error al actualizar contenido directo:', error);
      }
    } else if (!isStreaming) {
      // Al finalizar el streaming, remover el cursor
      try {
        const cursorElement = contentRef.current?.querySelector('.streaming-cursor');
        if (cursorElement && cursorElement.parentNode) {
          cursorElement.parentNode.removeChild(cursorElement);
        }
        
        // Actualizar el atributo data-is-streaming
        const messageContainer = contentRef.current?.closest('[data-message-role="assistant"]');
        if (messageContainer) {
          messageContainer.setAttribute('data-is-streaming', 'false');
        }
      } catch (error) {
        console.error('Error al limpiar indicadores de streaming:', error);
      }
    }
  }, [isStreaming, message.content]);

  // Preformatea el contenido para evitar parpadeo durante renderizado de markdown
  const preformattedContent = useMemo(() => {
    if (!message.content) return '';
    return message.content;
  }, [message.content]);

  // Format the timestamp
  const formattedTime = new Date(message.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit'
  });

  // Format response time if it exists (only for assistant messages)
  const formatResponseTime = (responseTimeMs?: number) => {
    if (responseTimeMs === undefined || responseTimeMs === null || responseTimeMs === 0) {
      return 'Response time --:--:--';
    }
    
    // Convert to minutes, seconds, and centiseconds
    const totalSeconds = responseTimeMs / 1000;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    const centiseconds = Math.floor((totalSeconds % 1) * 100);
    
    // Format as mm:ss.dd
    return `Response time ${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(centiseconds).padStart(2, '0')}`;
  };

  // Format model name for display
  const formatModelName = (modelId: string, provider: string) => {
    if (!modelId) return 'unknown';
    
    if (provider === 'ollama') {
      return modelId;
    }
    
    // For OpenAI, clean up model names
    const name = modelId.replace(/^gpt-/, '').replace(/-/g, ' ');
    return name.charAt(0).toUpperCase() + name.slice(1);
  };

  // Format provider name for display
  const formatProviderName = (provider: string) => {
    if (provider === 'openai') return 'OpenAI';
    if (provider === 'google') return 'Google';
    return provider.charAt(0).toUpperCase() + provider.slice(1); // Capitalize provider name
  };

  // Estimate tokens - more accurate than simple character count
  const estimateTokens = (text: string): number => {
    if (!text) return 0;
    
    // Approximate tokenization rules (GPT models):
    // 1. ~4 chars per token on average for English text
    // 2. But code, spaces, and non-English characters have different ratios
    
    // Count different content types
    const codeBlockMatches = text.match(/```[\s\S]*?```/g) || [];
    const codeBlocks = codeBlockMatches.join('');
    const nonCodeText = text.replace(/```[\s\S]*?```/g, '');
    
    // Different token ratios for different content
    const codeTokens = Math.ceil(codeBlocks.length / 3); // Code is slightly more efficient in tokens
    const textTokens = Math.ceil(nonCodeText.length / 4); // Normal text is ~4 chars per token
    
    return codeTokens + textTokens;
  };
  
  // Get the exact token count from the message or estimate it
  const totalTokens = useMemo(() => {
    // If we have an exact token count from the backend, use it
    if (message.tokenCount !== undefined) {
      return message.tokenCount;
    }
    
    // Store token count in a data attribute for consistency
    // This ensures the count doesn't change when navigating between chats
    if (message.content) {
      // Only calculate once
      return estimateTokens(message.content);
    }
    
    return 0;
  }, [message.tokenCount, message.content, estimateTokens]);

  // Calculate total response time (for non-streaming completed messages)
  const totalResponseTime = useMemo(() => {
    if (isUser || isStreaming || !message.timestamp) return null;
    
    // Usar directamente el valor persistido si está disponible
    if (message.totalElapsedTime) {
      return message.totalElapsedTime;
    }
    
    // Solo como fallback si no tenemos el valor persistido
    if (message.endTimestamp && message.timestamp) {
      const endTime = new Date(message.endTimestamp).getTime();
      const startTime = new Date(message.timestamp).getTime();
      return endTime - startTime;
    }
    
    // Último fallback
    return message.responseTimeMs || null;
  }, [isUser, isStreaming, message.timestamp, message.endTimestamp, message.responseTimeMs, message.totalElapsedTime]);
  
  // Format time in mm:ss.dd format
  const formatTimeInMs = useCallback((timeMs?: number | null) => {
    if (!timeMs) return "--:--:--";
    
    const totalSeconds = timeMs / 1000;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    const centiseconds = Math.floor((totalSeconds % 1) * 100);
    
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(centiseconds).padStart(2, '0')}`;
  }, []);
  
  // Calculate time per token
  const timePerToken = useMemo(() => {
    // Usar directamente el valor persistido, sin recalcular
    if (message.timePerToken) {
      return message.timePerToken;
    }
    
    // Default value if nothing available
    return 50;
  }, [message.timePerToken]);
  
  // Get tokens per second value
  const tokensPerSecond = useMemo(() => {
    // Usar directamente el valor persistido
    if (message.tokensPerSecond !== undefined) {
      return message.tokensPerSecond;
    }
    
    // Default value if nothing available
    return 0;
  }, [message.tokensPerSecond]);
  
  // Format time values - add proper units
  const formattedStatistics = useMemo(() => {
    const totalTimeFormatted = formatTimeInMs(totalResponseTime);
    // Usar directamente el conteo de tokens persistido
    const tokenDisplay = message.tokenCount !== undefined ? message.tokenCount : totalTokens;
    const timePerTokenFormatted = formatTimeInMs(timePerToken);
    
    // Formatear tokens por segundo con 2 decimales
    const tokensPerSecondFormatted = tokensPerSecond !== undefined ? tokensPerSecond.toFixed(2) : "0.00";
    
    // Formatear palabras por segundo con 2 decimales
    const wordsPerSecondValue = message.wordsPerSecond !== undefined ? Number(message.wordsPerSecond) : 0;
    const wordsPerSecondFormatted = wordsPerSecondValue.toFixed(2);

    // Formatear conteo de palabras
    const wordCountValue = message.wordCount !== undefined ? message.wordCount : 0;
    
    return `Total Response Time ${totalTimeFormatted} • # of Tokens ${tokenDisplay} • Time per Token ${timePerTokenFormatted} • Tokens/sec ${tokensPerSecondFormatted} • Words/sec ${wordsPerSecondFormatted} (${wordCountValue} words)`;
  }, [totalResponseTime, timePerToken, message.tokenCount, message.wordsPerSecond, message.wordCount, totalTokens, tokensPerSecond, formatTimeInMs]);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
      data-message-role={message.role}
      data-message-id={message.id || `${message.role}-${message.timestamp}`}
      data-is-streaming={isStreaming ? "true" : "false"}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          mb: 0.5,
          px: 1,
        }}
      >
        <Typography 
          variant="caption" 
          color="text.secondary"
          sx={{ 
            fontSize: '0.75rem', 
            opacity: 0.8, 
            fontFamily: 'monospace',
            backgroundColor: 'rgba(18, 21, 31, 0.4)',
            px: 1,
            py: 0.5,
            borderRadius: '4px',
            display: 'inline-block',  // Asegurar que fluya correctamente
            maxWidth: '100%',         // Evitar desbordamiento horizontal
            overflow: 'hidden',       // Prevenir contenido largo desbordando
            textOverflow: 'ellipsis', // Añadir elipsis si es necesario
            whiteSpace: 'nowrap'      // Mantener en una línea
          }}
        >
          {isUser ? 'You' : 'Assistant'} • {formattedTime}
          {isUser && (
            <span> • {totalTokens} Tokens</span>
          )}
          {!isUser && (
            <span>
              {` • ${formatResponseTime(message.responseTimeMs)}`}
              {` • ${formatProviderName(message.provider || settings.provider)} ${formatModelName(message.model || settings.model, message.provider || settings.provider)}`}
              {` • Context ${settings.maintain_context ? 'ON' : 'OFF'}`}
            </span>
          )}
        </Typography>
      </Box>

      <Paper
        elevation={0}
        sx={{
          p: isUser ? 2 : '16px 20px',
          maxWidth: '85%',
          borderRadius: '12px',
          backgroundColor: isUser 
            // // ? '#1F2937' // Dark slate for user messages
            // ? '#5a84b2' // Azul mas fuerte #0066ff #5a84b2 
            ? '#0660c3' // Azul mas suave
            // : 'rgba(17, 20, 31, 0.7)', // Oscuro similar al Refiner de la imagen
            : '#2a3746', // Mas claro #2a3746
            // : 'rgba(18, 21, 31, 0.7)',
          color: isUser ? theme.palette.text.primary : '#DADADA', // Light gray specifically 
          border: `1px solid ${isUser ? 'rgba(255, 255, 255, 0.08)' : 'rgba(87, 96, 111, 0.35)'}`,
          wordBreak: 'break-word', // Mejorar manejo de texto largo
          fontWeight: 'bold', // Texto en bold
        }}
        data-message-role={message.role}
        data-is-streaming={isStreaming ? "true" : "false"}
        id={`message-${message.id}`}
      >
        {isUser ? (
          <Typography sx={{ 
            fontSize: '0.95rem', 
            fontWeight: 415,
            // textShadow: '0 1px 2px rgba(0,0,0,0.3)',
            color: '#DADADA',
            // letterSpacing: '0.5px',
          }}>{message.content}</Typography>
        ) : (
          <>
            {/* Contenedor para el contenido del mensaje con optimización para streaming */}
            <Box 
              ref={contentRef}
              key={renderKey}
              sx={{ 
                // Aseguramos que no haya animaciones que causen parpadeo 
                transition: 'none',
                // Evitamos problemas de desbordamiento
                maxWidth: '100%',
                overflowWrap: 'break-word',
                // Mejora de estilo para streaming
                ...(isStreaming && {
                  position: 'relative',
                  '& .streaming-cursor': {
                    animation: `${cursorBlink} 1s step-end infinite`,
                  },
                  fontSize: '0.95rem',
                  lineHeight: 1.5,
                })
              }}
              data-is-streaming={isStreaming ? "true" : "false"}
            >
              {/* Para streaming, usamos texto plano para evitar parpadeo */}
              {isStreaming ? (
                // Para streaming, el contenido se maneja directamente vía DOM en useEffect
                null
              ) : (
                <ReactMarkdown
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '');
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={atomDark as any}
                          language={match[1]}
                          PreTag="div"
                          customStyle={{ 
                            borderRadius: '4px',
                            background: '#0F172A', // Darker background for code blocks
                          }}
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code 
                          className={className} 
                          style={{ 
                            backgroundColor: '#0F172A', 
                            borderRadius: '3px',
                            padding: '0.1em 0.3em',
                          }}
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    },
                    p: ({ children }) => (
                      <Typography sx={{ fontSize: '0.95rem', my: 0.5, color: '#DADADA' }}>
                        {children}
                      </Typography>
                    ),
                    // Componente personalizado para enlaces
                    a: ({ node, href, children }) => (
                      <a 
                        href={href}
                        style={{ 
                          color: '#CC7832',
                          textDecoration: 'none'
                        }}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {children}
                      </a>
                    ),
                  }}
                >
                  {preformattedContent}
                </ReactMarkdown>
              )}
            </Box>
            
            {/* Mostrar el indicador de carga durante el streaming solo cuando no hay contenido */}
            {isStreaming && !message.content && (
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1, height: 20 }}>
                <Box
                  component="span"
                  sx={{
                    display: 'inline-block',
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: '#CC7832',
                    mx: 0.5,
                    animation: `${dotAnimation1} 1.5s infinite`
                  }}
                />
                <Box
                  component="span"
                  sx={{
                    display: 'inline-block',
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: '#CC7832',
                    mx: 0.5,
                    animation: `${dotAnimation2} 1.5s infinite`
                  }}
                />
                <Box
                  component="span"
                  sx={{
                    display: 'inline-block',
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: '#CC7832',
                    mx: 0.5,
                    animation: `${dotAnimation3} 1.5s infinite`
                  }}
                />
              </Box>
            )}
          </>
        )}
      </Paper>
      
      {/* Footer with response statistics for assistant messages (only when not streaming) */}
      {!isUser && !isStreaming && message.content && (
        <Box 
          sx={{ 
            display: 'flex', 
            flexDirection: 'row',
            alignItems: 'center',
            mt: 0.5,
            px: 1
          }}
        >
          <Typography 
            variant="caption" 
            color="text.secondary" 
            sx={{ 
              fontSize: '0.75rem', 
              opacity: 0.8, 
              fontFamily: 'monospace',
              backgroundColor: 'rgba(18, 21, 31, 0.4)',
              px: 1,
              py: 0.5,
              borderRadius: '4px'
            }}
          >
            {formattedStatistics}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default React.memo(MessageItem); 