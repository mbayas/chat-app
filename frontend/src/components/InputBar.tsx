import React from 'react';
import { Box, TextField, IconButton, Tooltip } from '@mui/material';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import SendIcon from '@mui/icons-material/Send';
import StopIcon from '@mui/icons-material/Stop';
import { useChat } from '../contexts/ChatContext';

interface InputBarProps {
  input: string;
  setInput: (input: string) => void;
  handleSend: () => void;
  handleKeyDown: (e: React.KeyboardEvent) => void;
  inputRef?: React.RefObject<HTMLInputElement>;
  disabled: boolean;
  placeholder: string;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  isDragActive?: boolean;
}

const InputBar: React.FC<InputBarProps> = ({
  input,
  setInput,
  handleSend,
  handleKeyDown,
  inputRef,
  disabled,
  placeholder,
  onFileUpload,
  isDragActive
}) => {
  const { streamingMessage, cancelStreaming } = useChat();
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  // Manejar la pulsación de tecla Enter
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    } else {
      // Delegar a handleKeyDown para otras teclas (como flechas arriba/abajo)
      handleKeyDown(e);
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        backgroundColor: '#1F2937',
        border: '1px solid #2a2e42',
        borderRadius: '8px',
        p: 1.5,
        mx: 0,
        my: 0,
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
        width: '100%'
      }}
    >
      <TextField
        fullWidth
        multiline
        maxRows={4}
        placeholder={placeholder}
        variant="standard"
        value={input}
        onChange={handleInputChange}
        onKeyDown={handleKeyPress}
        inputRef={inputRef}
        InputProps={{
          disableUnderline: true,
          style: { 
            color: '#e0e0e0',
            fontSize: '0.95rem',
          }
        }}
        sx={{ 
          ml: 1,
          '& .MuiInputBase-root': {
            backgroundColor: 'transparent',
          }
        }}
      />
      
      <input
        type="file"
        id="file-upload"
        style={{ display: 'none' }}
        onChange={onFileUpload}
      />
      <label htmlFor="file-upload">
        <IconButton component="span" color="primary" sx={{ opacity: 0.8, '&:hover': { color: '#CC7832' } }}>
          <AttachFileIcon />
        </IconButton>
      </label>
      
      {streamingMessage ? (
        // Botón de cancelación sin texto
        <IconButton
          color="error"
          onClick={cancelStreaming}
          sx={{ 
            ml: 1,
            '&:hover': { color: '#CC7832' },
            animation: 'pulse 2s infinite',
            '@keyframes pulse': {
              '0%': { opacity: 0.6 },
              '50%': { opacity: 1 },
              '100%': { opacity: 0.6 }
            }
          }}
        >
          <StopIcon />
        </IconButton>
      ) : (
        // Botón de envío normal
        <IconButton
          color="primary"
          onClick={handleSend}
          disabled={input.trim() === '' || disabled}
          sx={{ 
            ml: 1,
            '&:hover': { color: '#CC7832' },
            '&.Mui-disabled': {
              opacity: 0.4,
            }
          }}
        >
          <SendIcon />
        </IconButton>
      )}
    </Box>
  );
};

export default InputBar; 