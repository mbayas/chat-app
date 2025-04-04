import React, { useState, useEffect } from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  IconButton,
  Typography,
  Divider,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  InputAdornment,
  Alert,
  FormControlLabel,
  Switch,
  FormHelperText,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  Close as CloseIcon,
  Key as KeyIcon,
  Computer as ComputerIcon,
  Cloud as CloudIcon,
  AutoAwesome as AutoAwesomeIcon,
  Edit as EditIcon,
} from '@mui/icons-material';
import { useChat, Chat } from '../contexts/ChatContext';
import { useSettings } from '../contexts/SettingsContext';
import { useProject, Project } from '../contexts/ProjectContext';
import { useNavigate } from 'react-router-dom';

interface SidebarProps {
  onClose: () => void;
}

interface Settings {
  provider: string;
  model: string;
  temperature: number;
  max_tokens: number;
  maintain_context: boolean;
}

const Sidebar = ({ onClose }: SidebarProps) => {
  const { chats, currentChat, createNewChat, selectChat, deleteChat, setInitialPrompt } = useChat();
  const { settings, availableModels, updateSettings, updateApiKey } = useSettings();
  const { projects, createNewProject, selectProject, deleteProject } = useProject();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);
  const [projectToDelete, setProjectToDelete] = useState<string | null>(null);
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState('http://localhost:11434');
  const [ollamaConnectionStatus, setOllamaConnectionStatus] = useState<'unknown' | 'connected' | 'error'>('unknown');
  const [newProjectDialogOpen, setNewProjectDialogOpen] = useState(false);
  const [newProjectTitle, setNewProjectTitle] = useState('');
  const [newProjectTitleError, setNewProjectTitleError] = useState('');
  const navigate = useNavigate();
  
  // Modelos específicos por proveedor para usar durante la edición
  const [localOllamaModels, setLocalOllamaModels] = useState<string[]>([]);
  const [localOpenAIModels, setLocalOpenAIModels] = useState<string[]>([
    'gpt-3.5-turbo',
    'gpt-4',
    'gpt-4-turbo-preview'
  ]);
  const [localGoogleModels, setLocalGoogleModels] = useState<string[]>([
    'gemini-1.5-pro',
    'gemini-1.5-flash',
    'gemini-1.0-pro',
    'gemini-1.0-ultra'
  ]);

  // Settings state
  const [tempSettings, setTempSettings] = useState({ ...settings });
  const [apiKey, setApiKey] = useState('');
  const [googleApiKey, setGoogleApiKey] = useState('');

  // Effect para actualizar los modelos de OpenAI cuando se abren las configuraciones
  useEffect(() => {
    if (settingsOpen && tempSettings.provider === 'openai') {
      // Filtrar modelos de OpenAI del availableModels global
      const openaiModels = availableModels.filter(model => 
        model.startsWith('gpt-') || 
        ['text-davinci-003', 'text-davinci-002', 'davinci'].includes(model)
      );
      
      if (openaiModels.length > 0) {
        console.log('Setting local OpenAI models:', openaiModels);
        setLocalOpenAIModels(openaiModels);
      } else {
        // Usar modelos por defecto si no hay disponibles
        console.log('Using default OpenAI models');
        setLocalOpenAIModels([
          'gpt-3.5-turbo',
          'gpt-4',
          'gpt-4-turbo-preview'
        ]);
      }
    }
  }, [settingsOpen, availableModels]);

  // Effect for updating Google models when the settings dialog opens
  useEffect(() => {
    if (settingsOpen && tempSettings.provider === 'google') {
      // Filter Google models from the global availableModels
      const googleModels = availableModels.filter(model => 
        model.startsWith('gemini-')
      );
      
      if (googleModels.length > 0) {
        console.log('Setting local Google models:', googleModels);
        setLocalGoogleModels(googleModels);
      } else {
        // Use default models if none are available
        console.log('Using default Google models');
        setLocalGoogleModels([
          'gemini-1.5-pro',
          'gemini-1.5-flash',
          'gemini-1.0-pro',
          'gemini-1.0-ultra'
        ]);
      }
    }
  }, [settingsOpen, availableModels, tempSettings.provider]);

  // Check Ollama connection when settings dialog opens and provider is Ollama
  useEffect(() => {
    if (settingsOpen && tempSettings.provider === 'ollama') {
      checkOllamaConnection();
    }
  }, [settingsOpen, tempSettings.provider, ollamaBaseUrl]);

  const checkOllamaConnection = async () => {
    try {
      // Intentar obtener la lista de modelos de Ollama
      console.log("Checking Ollama connection at", ollamaBaseUrl);
      const response = await fetch(`${ollamaBaseUrl}/api/tags`);
      
      console.log("Ollama connection check response:", response.status);
      
      if (response.ok) {
        setOllamaConnectionStatus('connected');
        
        // Obtener los modelos disponibles directamente de Ollama
        const data = await response.json();
        console.log("Modelos disponibles en Ollama:", data);
        
        // Extraer los nombres de los modelos
        let ollamaModels: string[] = [];
        if (data.models && Array.isArray(data.models)) {
          // Formato API más reciente
          ollamaModels = data.models.map((model: any) => model.name);
        } else if (Array.isArray(data)) {
          // Formato API antiguo
          ollamaModels = data.map((model: any) => model.name);
        }
        
        console.log("Modelos de Ollama detectados:", ollamaModels);
        
        // Guardar la lista de modelos detectados localmente
        setLocalOllamaModels(ollamaModels.length > 0 ? ollamaModels : []);
        
        // Si hay modelos detectados y el modelo actual no está en la lista
        if (ollamaModels.length > 0 && !ollamaModels.includes(tempSettings.model)) {
          console.log(`El modelo ${tempSettings.model} no está disponible. Cambiando al modelo ${ollamaModels[0]}`);
          setTempSettings({
            ...tempSettings,
            model: ollamaModels[0]
          });
        }
      } else {
        setOllamaConnectionStatus('error');
        setLocalOllamaModels([]);
      }
    } catch (error) {
      console.error('Error connecting to Ollama:', error);
      setOllamaConnectionStatus('error');
      setLocalOllamaModels([]);
    }
  };

  const handleCreateNewChat = () => {
    createNewChat();
    onClose();
  };

  const handleSelectChat = (chatId: string) => {
    selectChat(chatId);
    onClose();
  };

  const handleDeleteClick = (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();
    setChatToDelete(chatId);
    setProjectToDelete(null);
    setDeleteConfirmOpen(true);
  };

  const handleDeleteProjectClick = (e: React.MouseEvent, projectId: string) => {
    e.stopPropagation();
    setProjectToDelete(projectId);
    setChatToDelete(null);
    setDeleteConfirmOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (chatToDelete) {
      deleteChat(chatToDelete);
      setDeleteConfirmOpen(false);
      setChatToDelete(null);
    } else if (projectToDelete) {
      await deleteProject(projectToDelete);
      setDeleteConfirmOpen(false);
      setProjectToDelete(null);
    }
  };

  const handleCancelDelete = () => {
    setDeleteConfirmOpen(false);
    setChatToDelete(null);
    setProjectToDelete(null);
  };

  const handleOpenSettings = () => {
    // Establecer settings temporales basados en los actuales
    setTempSettings({ ...settings });
    
    // Inicializar modelos locales según el proveedor actual
    if (settings.provider === 'openai') {
      // Filtrar modelos de OpenAI
      const openaiModels = availableModels.filter(model => 
        model.startsWith('gpt-') || 
        ['text-davinci-003', 'text-davinci-002', 'davinci'].includes(model)
      );
      
      if (openaiModels.length > 0) {
        setLocalOpenAIModels(openaiModels);
      }
    } else if (settings.provider === 'ollama') {
      // Los modelos de Ollama se obtendrán mediante checkOllamaConnection
    } else if (settings.provider === 'google') {
      // Filter Google models
      const googleModels = availableModels.filter(model => 
        model.startsWith('gemini-')
      );
      
      if (googleModels.length > 0) {
        setLocalGoogleModels(googleModels);
      }
    }
    
    setApiKey(''); // Reset API key input field
    setGoogleApiKey(''); // Reset Google API key input field
    setOllamaBaseUrl(settings.ollama_base_url || 'http://localhost:11434');
    setSettingsOpen(true);
  };

  const handleCloseSettings = () => {
    setSettingsOpen(false);
  };

  const handleSaveSettings = async () => {
    try {
      // Actualizar la configuración global
      updateSettings(tempSettings);
      
      // Si se ha introducido una nueva API key, actualizarla
      if (apiKey && tempSettings.provider === 'openai') {
        await updateApiKey(apiKey, 'openai');
      }
      
      // Si se ha introducido una nueva API key de Google, actualizarla
      if (googleApiKey && tempSettings.provider === 'google') {
        await updateApiKey(googleApiKey, 'google');
      }
      
      // Cerrar el diálogo
      handleCloseSettings();
    } catch (error) {
      console.error('Error saving settings:', error);
      // Aquí podrías mostrar un mensaje de error al usuario
    }
  };

  const handleProviderChange = (e: SelectChangeEvent) => {
    const provider = e.target.value;
    
    // Seleccionar un modelo apropiado para el proveedor
    if (provider === 'openai') {
      // Restaurar a gpt-3.5-turbo para OpenAI
      setTempSettings(prev => ({
        ...prev,
        provider,
        model: localOpenAIModels.includes('gpt-3.5-turbo') ? 'gpt-3.5-turbo' : localOpenAIModels[0] || 'gpt-3.5-turbo'
      }));
    } else if (provider === 'ollama') {
      // Si hay modelos de Ollama detectados, usar el primero
      setTempSettings(prev => ({
        ...prev,
        provider,
        model: localOllamaModels.length > 0 ? localOllamaModels[0] : 'llama2'
      }));
    } else if (provider === 'google') {
      // Usar un modelo de Gemini para Google
      setTempSettings(prev => ({
        ...prev,
        provider,
        model: localGoogleModels.includes('gemini-1.5-pro') ? 'gemini-1.5-pro' : localGoogleModels[0] || 'gemini-1.5-pro'
      }));
    }
  };

  const handleModelChange = (e: SelectChangeEvent) => {
    setTempSettings({
      ...tempSettings,
      model: e.target.value
    });
  };

  const handleTemperatureChange = (_: Event, value: number | number[]) => {
    setTempSettings({ ...tempSettings, temperature: value as number });
  };

  const handleMaxTokensChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value)) {
      setTempSettings({ ...tempSettings, max_tokens: value });
    }
  };

  const handleApiKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setApiKey(e.target.value);
  };

  const handleOllamaBaseUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setOllamaBaseUrl(e.target.value);
  };

  const handleMaintainContextChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTempSettings({ ...tempSettings, maintain_context: event.target.checked });
  };

  const formatModelName = (modelId: string) => {
    // Convert model IDs like "gpt-3.5-turbo" to nicer display names like "GPT-3.5 Turbo"
    if (modelId.startsWith('gpt-')) {
      return modelId.replace('gpt-', 'GPT-').replace(/-([a-z])/g, (_, letter) => ' ' + letter.toUpperCase());
    } else if (modelId.startsWith('gemini-')) {
      return modelId.replace('gemini-', 'Gemini ').replace(/-([a-z])/g, (_, letter) => ' ' + letter.toUpperCase());
    }
    return modelId.charAt(0).toUpperCase() + modelId.slice(1); // Capitalize first letter
  };

  const getModelsForCurrentProvider = () => {
    switch (tempSettings.provider) {
      case 'openai':
        return localOpenAIModels;
      case 'ollama':
        return localOllamaModels;
      case 'google':
        return localGoogleModels;
      default:
        return [];
    }
  };

  const handleCreateNewProject = () => {
    setNewProjectDialogOpen(true);
    setNewProjectTitle('');
  };

  const handleSelectProject = async (projectId: string) => {
    // Find the project data first
    const projectData = projects.find(p => p.id === projectId);
    if (projectData) {
      // Set the initial prompt with the project name format
      setInitialPrompt(`**${projectData.title}**`);
    }
    
    // Select the project
    await selectProject(projectId);
    
    // Then create a new chat
    createNewChat();
    
    // Close the sidebar (on mobile)
    onClose();
  };

  const handleEditProject = (e: React.MouseEvent, projectId: string) => {
    e.stopPropagation(); // Prevent triggering the ListItemButton click
    navigate(`/project/${projectId}`);
  };

  const handleSaveNewProject = async () => {
    if (newProjectTitle.trim() === '') {
      setNewProjectTitleError('Project title cannot be empty');
      return;
    }
    
    // Validar que el título solo contenga caracteres permitidos
    const validNameRegex = /^[a-zA-Z0-9_-]+$/;
    if (!validNameRegex.test(newProjectTitle)) {
      setNewProjectTitleError('Project title can only contain letters, numbers, underscores (_) and hyphens (-). No spaces or special characters allowed.');
      return;
    }
    
    try {
      const projectId = await createNewProject(newProjectTitle);
      setNewProjectDialogOpen(false);
      setNewProjectTitle('');
      setNewProjectTitleError('');
      // Navigate to the project page with the new project ID
      navigate(`/project/${projectId}`);
    } catch (error: any) {
      // Capturar errores del backend, por ejemplo si el nombre ya existe
      if (error.response && error.response.data && error.response.data.error) {
        setNewProjectTitleError(error.response.data.error);
      } else {
        setNewProjectTitleError('Error creating project. Please try again.');
      }
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', bgcolor: '#0B0F18' }}>
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="subtitle1" sx={{ fontSize: '1.3rem', fontWeight: 500 }}>
            <Box component="span" sx={{ color: '#CC7832' }}>G</Box>
            <Box component="span" sx={{ color: '#DADADA' }}>e</Box>
            <Box component="span" sx={{ color: '#DADADA' }}>p</Box>
            <Box component="span" sx={{ color: '#CC7832' }}>p</Box>
            <Box component="span" sx={{ color: '#DADADA' }}>e</Box>
            <Box component="span" sx={{ color: '#DADADA' }}>t</Box>
            <Box component="span" sx={{ color: '#CC7832' }}>t</Box>
            <Box component="span" sx={{ color: '#DADADA' }}>o</Box>
          </Typography>
        </Box>
        <Box>
          <IconButton onClick={handleOpenSettings} size="small" sx={{ color: '#a0a0a0', '&:hover': { color: '#CC7832' } }}>
            <SettingsIcon />
          </IconButton>
          <IconButton onClick={onClose} size="small" sx={{ display: { md: 'none' }, color: '#a0a0a0', '&:hover': { color: '#CC7832' } }}>
            <CloseIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Projects Section */}
      {projects.length > 0 && (
        <>
          <Box sx={{ p: 2 }}>
            <List sx={{ 
              maxHeight: '200px', 
              overflow: 'auto',
              '&::-webkit-scrollbar': {
                width: '8px',
              },
              '&::-webkit-scrollbar-track': {
                background: 'rgba(31, 41, 55, 0.1)',
              },
              '&::-webkit-scrollbar-thumb': {
                background: 'rgba(204, 120, 50, 0.5)',
                borderRadius: '4px',
              },
              '&::-webkit-scrollbar-thumb:hover': {
                background: 'rgba(204, 120, 50, 0.7)',
              }
            }}>
              {projects.map((project: Project) => (
                <ListItem
                  key={project.id}
                  disablePadding
                  secondaryAction={
                    <Box sx={{ mr: 1 }}>
                      <IconButton
                        edge="end"
                        onClick={(e) => handleEditProject(e, project.id)}
                        size="small"
                        sx={{ 
                          color: '#a0a0a0',
                          '&:hover': { color: '#CC7832' }
                        }}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        edge="end"
                        onClick={(e) => handleDeleteProjectClick(e, project.id)}
                        size="small"
                        sx={{ 
                          color: '#a0a0a0',
                          '&:hover': { color: '#CC7832' }
                        }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  }
                >
                  <ListItemButton
                    onClick={() => handleSelectProject(project.id)}
                    sx={{ 
                      '&:hover': {
                        backgroundColor: 'rgba(31, 41, 55, 0.5)'
                      },
                      py: 1.5,
                      pr: 7
                    }}
                  >
                    <ListItemText
                      primary={project.title}
                      primaryTypographyProps={{
                        noWrap: true,
                        style: { maxWidth: '220px', color: '#DADADA', fontWeight: 500 }
                      }}
                    />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Box>
          <Divider sx={{ backgroundColor: '#2a2e42' }} />
        </>
      )}

      {/* New Project Button */}
      <Box sx={{ p: 2 }}>
        <Button
          variant="contained"
          fullWidth
          onClick={handleCreateNewProject}
          sx={{ 
            backgroundColor: '#CC7832', 
            color: '#FFFFFF',
            '&:hover': {
              backgroundColor: '#e48f4c'
            }
          }}
        >
          New Project
        </Button>
      </Box>

      {/* New Chat Button */}
      <Box sx={{ p: 2 }}>
        <Button
          variant="contained"
          fullWidth
          onClick={handleCreateNewChat}
          sx={{ 
            backgroundColor: '#1F2937', 
            color: '#DADADA',
            '&:hover': {
              backgroundColor: '#374151'
            }
          }}
        >
          New Chat
        </Button>
      </Box>

      <Divider sx={{ backgroundColor: '#2a2e42' }} />

      {/* Chat List */}
      <List sx={{ 
        flexGrow: 1, 
        overflow: 'auto',
        '&::-webkit-scrollbar': {
          width: '8px',
        },
        '&::-webkit-scrollbar-track': {
          background: 'rgba(31, 41, 55, 0.1)',
        },
        '&::-webkit-scrollbar-thumb': {
          background: 'rgba(204, 120, 50, 0.5)',
          borderRadius: '4px',
        },
        '&::-webkit-scrollbar-thumb:hover': {
          background: 'rgba(204, 120, 50, 0.7)',
        }
      }}>
        {chats.length === 0 ? (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" sx={{ color: '#a0a0a0' }}>
              No chats yet. Start a new conversation!
            </Typography>
          </Box>
        ) : (
          chats.map((chat: Chat) => (
            <ListItem
              key={chat.id}
              disablePadding
              secondaryAction={
                <Box sx={{ mr: 1 }}>
                  <IconButton
                    edge="end"
                    onClick={(e) => handleDeleteClick(e, chat.id)}
                    size="small"
                    sx={{ 
                      color: '#a0a0a0',
                      '&:hover': { color: '#CC7832' }
                    }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </Box>
              }
            >
              <ListItemButton
                selected={currentChat === chat.id}
                onClick={() => handleSelectChat(chat.id)}
                sx={{ 
                  '&.Mui-selected': {
                    backgroundColor: '#1F2937',
                    '&:hover': {
                      backgroundColor: '#374151'
                    }
                  },
                  '&:hover': {
                    backgroundColor: 'rgba(31, 41, 55, 0.5)'
                  },
                  py: 1.5,
                  pr: 7
                }}
              >
                <ListItemText
                  primary={chat.title}
                  primaryTypographyProps={{
                    noWrap: true,
                    style: { maxWidth: '220px', color: '#DADADA', fontWeight: 500 }
                  }}
                  secondary={new Date(chat.created_at).toLocaleDateString(undefined, { 
                    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' 
                  })}
                  secondaryTypographyProps={{
                    noWrap: true,
                    variant: 'caption',
                    style: { color: '#a0a0a0' }
                  }}
                />
              </ListItemButton>
            </ListItem>
          ))
        )}
      </List>

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={handleCloseSettings} maxWidth="sm" fullWidth>
        <DialogTitle>Chat Settings</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth margin="normal">
              <InputLabel>Provider</InputLabel>
              <Select
                value={tempSettings.provider}
                label="Provider"
                onChange={handleProviderChange}
              >
                <MenuItem value="openai" sx={{ display: 'flex', alignItems: 'center' }}>
                  <CloudIcon fontSize="small" sx={{ mr: 1 }} />
                  OpenAI
                </MenuItem>
                <MenuItem value="ollama" sx={{ display: 'flex', alignItems: 'center' }}>
                  <ComputerIcon fontSize="small" sx={{ mr: 1 }} />
                  Ollama (Local)
                </MenuItem>
                <MenuItem value="google" sx={{ display: 'flex', alignItems: 'center' }}>
                  <AutoAwesomeIcon fontSize="small" sx={{ mr: 1 }} />
                  Google
                </MenuItem>
              </Select>
            </FormControl>
            
            {/* Ollama-specific settings */}
            {tempSettings.provider === 'ollama' && (
              <>
                <TextField
                  fullWidth
                  margin="normal"
                  label="Ollama API URL"
                  value={ollamaBaseUrl}
                  onChange={handleOllamaBaseUrlChange}
                  helperText="Default: http://localhost:11434"
                />
                
                {ollamaConnectionStatus === 'connected' && (
                  <Alert severity="success" sx={{ mt: 1 }}>
                    Successfully connected to Ollama server. Models found!
                    <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                      Current model: <strong>{tempSettings.model}</strong>
                    </Typography>
                  </Alert>
                )}
                
                {ollamaConnectionStatus === 'error' && (
                  <Alert severity="error" sx={{ mt: 1 }}>
                    Could not connect to Ollama server. Please ensure Ollama is running and the URL is correct.
                  </Alert>
                )}
              </>
            )}

            {/* OpenAI-specific settings */}
            {tempSettings.provider === 'openai' && (
              <TextField
                fullWidth
                margin="normal"
                label="OpenAI API Key"
                type="password"
                value={apiKey}
                onChange={handleApiKeyChange}
                helperText="Enter your OpenAI API key directly instead of using environment variables"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <KeyIcon />
                    </InputAdornment>
                  ),
                }}
              />
            )}

            {/* Google-specific settings */}
            {tempSettings.provider === 'google' && (
              <TextField
                fullWidth
                margin="normal"
                label="Google API Key"
                type="password"
                value={googleApiKey}
                onChange={(e) => setGoogleApiKey(e.target.value)}
                helperText="Enter your Google Gemini API key"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <KeyIcon />
                    </InputAdornment>
                  ),
                }}
              />
            )}

            {/* Common settings */}
            <FormControl fullWidth margin="normal">
              <InputLabel>Model</InputLabel>
              <Select
                value={tempSettings.model}
                label="Model"
                onChange={handleModelChange}
              >
                {getModelsForCurrentProvider().map((model) => (
                  <MenuItem key={model} value={model}>
                    {formatModelName(model)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth margin="normal">
              <FormControlLabel
                control={
                  <Switch
                    checked={tempSettings.maintain_context}
                    onChange={handleMaintainContextChange}
                  />
                }
                label="Maintain conversation context"
              />
              <FormHelperText>
                Enabled: The model remembers previous messages. Disabled: Each message is treated individually.
              </FormHelperText>
            </FormControl>

            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>
                Temperature: {tempSettings.temperature.toFixed(1)}
              </Typography>
              <Slider
                value={tempSettings.temperature}
                min={0}
                max={1}
                step={0.1}
                onChange={handleTemperatureChange}
                valueLabelDisplay="auto"
              />
              <Typography variant="caption" color="text.secondary">
                Lower values make responses more deterministic, higher values make them more creative.
              </Typography>
            </Box>

            <TextField
              fullWidth
              margin="normal"
              label="Max Tokens"
              type="number"
              value={tempSettings.max_tokens}
              onChange={handleMaxTokensChange}
              helperText="Maximum length of the model's response"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseSettings}>Cancel</Button>
          <Button onClick={handleSaveSettings} variant="contained" sx={{ backgroundColor: '#CC7832', color: '#FFFFFF', '&:hover': { backgroundColor: '#e48f4c' } }}>
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* New Project Dialog */}
      <Dialog open={newProjectDialogOpen} onClose={() => setNewProjectDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Project</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Project Title"
            type="text"
            fullWidth
            value={newProjectTitle}
            onChange={(e) => setNewProjectTitle(e.target.value)}
            error={!!newProjectTitleError}
            helperText={newProjectTitleError || "Use only letters, numbers, underscores (_) and hyphens (-). No spaces or special characters."}
            placeholder="my_project_name"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setNewProjectDialogOpen(false);
            setNewProjectTitleError('');
          }}>Cancel</Button>
          <Button 
            onClick={handleSaveNewProject} 
            variant="contained" 
            sx={{ backgroundColor: '#CC7832', color: '#FFFFFF', '&:hover': { backgroundColor: '#e48f4c' } }}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteConfirmOpen} onClose={handleCancelDelete}>
        <DialogTitle>Delete {chatToDelete ? 'Chat' : 'Project'}</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this {chatToDelete ? 'chat' : 'project'}? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelDelete}>Cancel</Button>
          <Button onClick={handleConfirmDelete} sx={{ color: '#FFFFFF', backgroundColor: '#CC7832', '&:hover': { backgroundColor: '#e48f4c' } }}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Sidebar; 