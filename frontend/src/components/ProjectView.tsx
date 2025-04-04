import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  TextField, 
  Button, 
  Grid, 
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextareaAutosize
} from '@mui/material';
import {
  FolderOutlined as FolderIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Upload as UploadIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import { useProject, Project } from '../contexts/ProjectContext';
import { useChat } from '../contexts/ChatContext';

interface ProjectViewProps {
  projectId: string;
}

const ProjectView: React.FC<ProjectViewProps> = ({ projectId }) => {
  const { projects, updateSystemPrompt, uploadProjectFile } = useProject();
  const { sendMessage } = useChat();
  
  // States for dialogs
  const [fileUploadOpen, setFileUploadOpen] = useState(false);
  const [systemPromptOpen, setSystemPromptOpen] = useState(false);
  const [newChatMessage, setNewChatMessage] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  
  // File upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  // Get current project details
  const currentProject = projects.find(p => p.id === projectId);
  
  // Load system prompt from current project
  React.useEffect(() => {
    if (currentProject) {
      setSystemPrompt(currentProject.system_prompt || '');
    }
  }, [currentProject]);
  
  if (!currentProject) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h5" color="text.secondary">
          Project not found
        </Typography>
      </Box>
    );
  }
  
  // Handle starting a new chat in this project
  const handleStartChat = () => {
    if (newChatMessage.trim()) {
      sendMessage(newChatMessage);
      setNewChatMessage('');
    }
  };
  
  // Handle key press in chat input
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleStartChat();
    }
  };
  
  // Handle file upload
  const handleFileUpload = async () => {
    if (selectedFile && currentProject) {
      try {
        await uploadProjectFile(currentProject.id, selectedFile);
        setSelectedFile(null);
        setFileUploadOpen(false);
      } catch (error) {
        console.error('Error uploading file:', error);
      }
    }
  };
  
  // Handle system prompt update
  const handleSystemPromptUpdate = async () => {
    if (currentProject) {
      try {
        await updateSystemPrompt(currentProject.id, systemPrompt);
        setSystemPromptOpen(false);
      } catch (error) {
        console.error('Error updating system prompt:', error);
      }
    }
  };
  
  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%', 
      p: 3, 
      backgroundColor: '#0B0F18' 
    }}>
      {/* Project Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
        <FolderIcon sx={{ fontSize: 48, color: '#a0a0a0', mr: 2 }} />
        <Typography variant="h4" component="h1" sx={{ color: '#DADADA', fontWeight: 500 }}>
          {currentProject.title}
        </Typography>
      </Box>
      
      {/* New Chat Input */}
      <Paper
        elevation={0}
        sx={{
          p: 2,
          mb: 4,
          backgroundColor: '#1F2937',
          borderRadius: 2,
          border: '1px solid #2a2e42'
        }}
      >
        <TextField
          fullWidth
          placeholder="New chat in this project"
          variant="outlined"
          value={newChatMessage}
          onChange={(e) => setNewChatMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          sx={{
            mb: 2,
            '& .MuiOutlinedInput-root': {
              backgroundColor: '#1F2937',
              color: '#DADADA',
              '&:hover fieldset': {
                borderColor: '#3f4562',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#CC7832',
              },
            }
          }}
        />
        
        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            onClick={handleStartChat}
            disabled={!newChatMessage.trim()}
            sx={{
              backgroundColor: '#CC7832',
              color: '#FFFFFF',
              '&:hover': {
                backgroundColor: '#e48f4c',
              },
              '&.Mui-disabled': {
                backgroundColor: '#3a3e52',
                color: '#a0a0a0',
              }
            }}
          >
            Start Chat
          </Button>
        </Box>
      </Paper>
      
      {/* Project Tools */}
      <Grid container spacing={3}>
        {/* Add Files Section */}
        <Grid item xs={12} md={6}>
          <Paper
            elevation={0}
            sx={{
              p: 3,
              height: '100%',
              backgroundColor: '#1F2937',
              borderRadius: 2,
              border: '1px solid #2a2e42',
              cursor: 'pointer',
              '&:hover': {
                borderColor: '#3f4562',
              }
            }}
            onClick={() => setFileUploadOpen(true)}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ color: '#DADADA', flex: 1 }}>
                Add files
              </Typography>
              <IconButton sx={{ color: '#CC7832' }}>
                <UploadIcon />
              </IconButton>
            </Box>
            <Typography variant="body2" sx={{ color: '#a0a0a0' }}>
              Chats in this project can access file content
            </Typography>
          </Paper>
        </Grid>
        
        {/* Add Instructions Section */}
        <Grid item xs={12} md={6}>
          <Paper
            elevation={0}
            sx={{
              p: 3,
              height: '100%',
              backgroundColor: '#1F2937',
              borderRadius: 2,
              border: '1px solid #2a2e42',
              cursor: 'pointer',
              '&:hover': {
                borderColor: '#3f4562',
              }
            }}
            onClick={() => setSystemPromptOpen(true)}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ color: '#DADADA', flex: 1 }}>
                Add instructions
              </Typography>
              <IconButton sx={{ color: '#CC7832' }}>
                <EditIcon />
              </IconButton>
            </Box>
            <Typography variant="body2" sx={{ color: '#a0a0a0' }}>
              Tailor the way Geppetto responds in this project
            </Typography>
          </Paper>
        </Grid>
      </Grid>
      
      {/* File Upload Dialog */}
      <Dialog open={fileUploadOpen} onClose={() => setFileUploadOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          Upload File to Project
          <IconButton 
            onClick={() => setFileUploadOpen(false)}
            sx={{ position: 'absolute', right: 8, top: 8, color: '#a0a0a0' }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <Typography gutterBottom>
              Select a file to upload to this project. Uploaded files can be accessed by chats within this project.
            </Typography>
            <input
              type="file"
              onChange={handleFileChange}
              style={{ display: 'block', marginTop: '16px' }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFileUploadOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleFileUpload}
            variant="contained"
            disabled={!selectedFile}
            sx={{ 
              backgroundColor: '#CC7832', 
              color: '#FFFFFF',
              '&:hover': {
                backgroundColor: '#e48f4c',
              }
            }}
          >
            Upload
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* System Prompt Dialog */}
      <Dialog open={systemPromptOpen} onClose={() => setSystemPromptOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Project Instructions
          <IconButton 
            onClick={() => setSystemPromptOpen(false)}
            sx={{ position: 'absolute', right: 8, top: 8, color: '#a0a0a0' }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <Typography gutterBottom>
              Set instructions for how Geppetto should respond in this project. These instructions will be included with every message.
            </Typography>
            <TextareaAutosize
              minRows={10}
              style={{ 
                width: '100%', 
                marginTop: '16px',
                padding: '12px',
                backgroundColor: '#1F2937',
                color: '#DADADA',
                border: '1px solid #2a2e42',
                borderRadius: '4px',
                fontSize: '14px',
                resize: 'vertical'
              }}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Enter instructions for Geppetto..."
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSystemPromptOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleSystemPromptUpdate}
            variant="contained"
            sx={{ 
              backgroundColor: '#CC7832', 
              color: '#FFFFFF',
              '&:hover': {
                backgroundColor: '#e48f4c',
              }
            }}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProjectView; 