import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  IconButton, 
  Paper, 
  Button, 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  DialogActions, 
  TextField,
  Grid,
  useMediaQuery,
  useTheme,
  Divider,
  Tooltip,
  CircularProgress
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Edit as EditIcon,
  AttachFile as AttachFileIcon,
  Delete as DeleteIcon,
  UploadFile as UploadFileIcon
} from '@mui/icons-material';
import { useProject } from '../contexts/ProjectContext';
import Sidebar from './Sidebar';

const ProjectPage: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const { projects, selectProject, updateSystemPrompt, uploadProjectFile, deleteProjectFile, uploadSystemPromptFile } = useProject();
  
  const [project, setProject] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [systemPromptOpen, setSystemPromptOpen] = useState<boolean>(false);
  const [systemPrompt, setSystemPrompt] = useState<string>('');
  const [fileInputRef] = useState<React.RefObject<HTMLInputElement>>(React.createRef());
  const [promptFileInputRef] = useState<React.RefObject<HTMLInputElement>>(React.createRef());
  const [processing, setProcessing] = useState<boolean>(false);
  const [projectLoaded, setProjectLoaded] = useState<boolean>(false);

  useEffect(() => {
    const loadProject = async () => {
      if (projectId && !projectLoaded) {
        setLoading(true);
        const projectData = await selectProject(projectId);
        if (projectData) {
          setProject(projectData);
          setSystemPrompt(projectData.system_prompt || '');
          setProjectLoaded(true);
        } else {
          // Project not found, redirect to home
          navigate('/');
        }
        setLoading(false);
      }
    };

    loadProject();
  }, [projectId, selectProject, navigate, projectLoaded]);

  // Reset projectLoaded if projectId changes
  useEffect(() => {
    setProjectLoaded(false);
  }, [projectId]);

  const handleGoBack = () => {
    navigate('/');
  };

  const handleOpenSystemPrompt = () => {
    setSystemPromptOpen(true);
  };

  const handleCloseSystemPrompt = () => {
    setSystemPromptOpen(false);
  };

  const handleSaveSystemPrompt = async () => {
    if (projectId) {
      await updateSystemPrompt(projectId, systemPrompt);
      setSystemPromptOpen(false);
    }
  };

  const handleFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0 && projectId) {
      setLoading(true); // Mostrar estado de carga
      try {
        const file = e.target.files[0];
        await uploadProjectFile(projectId, file);
        
        // Recargar explícitamente los datos del proyecto para reflejar el nuevo archivo
        const updatedProject = await selectProject(projectId);
        if (updatedProject) {
          setProject(updatedProject);
        }
        
        console.log("Archivo cargado y proyecto actualizado:", updatedProject);
      } catch (error) {
        console.error("Error al cargar el archivo:", error);
      } finally {
        // Reset the file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        setLoading(false);
      }
    }
  };

  const handlePromptFileClick = () => {
    promptFileInputRef.current?.click();
  };

  const handlePromptFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0 && projectId) {
      setLoading(true);
      try {
        const file = e.target.files[0];
        await uploadSystemPromptFile(projectId, file);
        
        // Recargar explícitamente los datos del proyecto para reflejar el nuevo sistema prompt
        const updatedProject = await selectProject(projectId);
        if (updatedProject) {
          setProject(updatedProject);
          setSystemPrompt(updatedProject.system_prompt || '');
        }
        
        console.log("Archivo de instrucciones cargado y proyecto actualizado:", updatedProject);
      } catch (error) {
        console.error("Error al cargar el archivo de instrucciones:", error);
      } finally {
        // Reset the file input
        if (promptFileInputRef.current) {
          promptFileInputRef.current.value = '';
        }
        setLoading(false);
      }
    }
  };

  const handleDeleteFile = async (fileId: string) => {
    if (projectId) {
      setLoading(true);
      try {
        await deleteProjectFile(projectId, fileId);
        
        // Recargar explícitamente los datos del proyecto para reflejar los cambios
        const updatedProject = await selectProject(projectId);
        if (updatedProject) {
          setProject(updatedProject);
        }
        
        console.log("Archivo eliminado y proyecto actualizado:", updatedProject);
      } catch (error) {
        console.error("Error al eliminar el archivo:", error);
      } finally {
        setLoading(false);
      }
    }
  };

  // Helper function to truncate text to a specific number of words
  const truncateText = (text: string, wordLimit: number): string => {
    const words = text.split(/\s+/);
    if (words.length <= wordLimit) {
      return text;
    }
    return words.slice(0, wordLimit).join(' ') + ' ....';
  };

  // Function to handle RAG processing of files
  const handleProcess = async () => {
    if (!project || !project.files || project.files.length === 0) {
      alert('Please add at least one file to process');
      return;
    }
    
    // Prevent multiple clicks
    if (processing) {
      console.log('Processing already in progress');
      return;
    }
    
    try {
      setProcessing(true);
      console.log('Starting RAG processing for project:', projectId);
      
      // Provide immediate feedback
      alert('RAG processing started. This may take a minute or two depending on the size of your files.');
      
      // Call backend API to process files for RAG
      const response = await fetch(`/api/projects/${projectId}/process-rag`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          // Default parameters for chunking
          chunk_size: 1000,
          chunk_overlap: 150
        }),
      });
      
      console.log('RAG processing API response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response from server:', errorText);
        throw new Error(`Server returned ${response.status}: ${errorText}`);
      }
      
      const result = await response.json();
      console.log('RAG processing result:', result);
      
      if (result.status === 'success') {
        alert(`${result.message}\nProcessed ${result.processed_files.length} files with ${result.total_chunks} chunks.`);
      } else if (result.status === 'warning') {
        alert(`${result.message}\nPlease check that you have uploaded PDF files.`);
      } else {
        throw new Error(result.message || 'Unknown error occurred');
      }
    } catch (error) {
      console.error('Error processing files:', error);
      alert(`An error occurred while processing files: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      console.log('RAG processing completed');
      setProcessing(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Typography>Loading project...</Typography>
      </Box>
    );
  }

  if (!project) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Typography>Project not found</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', width: '100%', height: '100%' }}>
      <Grid container sx={{ height: '100%' }}>
        {/* Sidebar */}
        <Grid
          item
          sx={{
            width: sidebarOpen ? (isMobile ? '100%' : '300px') : '0px',
            height: '100%',
            display: sidebarOpen ? 'block' : 'none',
            position: isMobile ? 'absolute' : 'relative',
            zIndex: isMobile ? 1000 : 1,
            bgcolor: '#0B0F18',
            borderRight: '1px solid #2a2e42',
          }}
        >
          <Sidebar onClose={() => isMobile && setSidebarOpen(false)} />
        </Grid>

        {/* Project Content */}
        <Grid
          item
          xs
          sx={{
            height: '100%',
            width: sidebarOpen && !isMobile ? 'calc(100% - 300px)' : '100%',
            backgroundColor: '#0B0F18',
            color: '#e0e0e0',
            overflow: 'auto'
          }}
        >
          {/* Header */}
          <Box sx={{ 
            p: 2, 
            display: 'flex', 
            alignItems: 'center', 
            borderBottom: '1px solid #2a2e42',
            backgroundColor: '#0B0F18'
          }}>
            <IconButton onClick={handleGoBack} sx={{ color: '#a0a0a0', mr: 2 }}>
              <ArrowBackIcon />
            </IconButton>
            <Typography variant="h5" component="div" sx={{ flexGrow: 1 }}>
              {project.title}
            </Typography>
          </Box>

          {/* Project content */}
          <Box sx={{ p: 4, maxWidth: '1000px', mx: 'auto' }}>
            {/* Project folder icon and name */}
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
              <Box 
                sx={{ 
                  width: 64, 
                  height: 64, 
                  borderRadius: '50%', 
                  bgcolor: 'rgba(255, 255, 255, 0.05)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mr: 2
                }}
              >
                <Box sx={{ fontSize: 32 }}>📁</Box>
              </Box>
              <Typography variant="h3" component="h1" sx={{ color: '#e0e0e0' }}>
                {project.title}
              </Typography>
            </Box>

            {/* Project options */}
            <Box sx={{ mt: 8, display: 'flex', flexWrap: 'wrap', gap: 3, justifyContent: 'center' }}>
              {/* Add files section */}
              <Paper sx={{ 
                p: 4, 
                width: '48%', 
                bgcolor: '#1F2937', 
                borderRadius: '12px',
                border: '1px solid #2a2e42',
                display: 'flex',
                flexDirection: 'column'
              }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ color: '#e0e0e0', fontWeight: 'bold' }}>Add files</Typography>
                  <IconButton 
                    onClick={handleFileClick} 
                    sx={{ 
                      bgcolor: 'rgba(204, 120, 50, 0.1)',
                      '&:hover': { 
                        bgcolor: 'rgba(204, 120, 50, 0.2)',
                        color: '#CC7832'
                      }
                    }}
                    disabled={loading}
                  >
                    <AttachFileIcon sx={{ color: '#CC7832' }} />
                  </IconButton>
                </Box>
                <Typography variant="body2" sx={{ color: '#a0a0a0' }}>
                  Chats in this project can access file content
                </Typography>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />

                {/* List of files */}
                {project.files && project.files.length > 0 ? (
                  <Box sx={{ mt: 2 }}>
                    {project.files.map((file: any) => (
                      <Box 
                        key={file.id} 
                        sx={{ 
                          display: 'flex', 
                          justifyContent: 'space-between', 
                          alignItems: 'center',
                          p: 1,
                          borderRadius: '4px',
                          '&:hover': { bgcolor: 'rgba(42, 46, 66, 0.5)' }
                        }}
                      >
                        <Typography variant="body2" sx={{ color: '#e0e0e0' }}>
                          {file.original_name || file.name}
                        </Typography>
                        <IconButton 
                          size="small" 
                          onClick={() => handleDeleteFile(file.id)}
                          sx={{ 
                            color: '#a0a0a0',
                            '&:hover': { color: '#CC7832' }
                          }}
                          disabled={loading}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    ))}
                  </Box>
                ) : (
                  <Typography variant="body2" sx={{ color: '#a0a0a0', mt: 2 }}>
                    {loading ? 'Cargando archivos...' : 'No files uploaded yet'}
                  </Typography>
                )}

                {/* Indicador de carga superpuesto */}
                {loading && (
                  <Box sx={{ 
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    right: 0, 
                    bottom: 0, 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    bgcolor: 'rgba(0, 0, 0, 0.5)',
                    borderRadius: '12px',
                    zIndex: 10
                  }}>
                    <CircularProgress size={40} sx={{ color: '#CC7832' }} />
                  </Box>
                )}
              </Paper>

              {/* Edit instructions section */}
              <Paper sx={{ 
                p: 4, 
                width: '48%', 
                bgcolor: '#1F2937', 
                borderRadius: '12px',
                border: '1px solid #2a2e42',
                display: 'flex',
                flexDirection: 'column',
                position: 'relative'
              }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ color: '#e0e0e0', fontWeight: 'bold' }}>Add instructions</Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Tooltip title="Upload a file as instructions">
                      <IconButton 
                        onClick={handlePromptFileClick} 
                        sx={{ 
                          bgcolor: 'rgba(204, 120, 50, 0.1)',
                          '&:hover': { 
                            bgcolor: 'rgba(204, 120, 50, 0.2)',
                            color: '#CC7832'
                          }
                        }}
                        disabled={loading}
                      >
                        <UploadFileIcon sx={{ color: '#CC7832' }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit instructions">
                      <IconButton 
                        onClick={handleOpenSystemPrompt} 
                        sx={{ 
                          bgcolor: 'rgba(204, 120, 50, 0.1)',
                          '&:hover': { 
                            bgcolor: 'rgba(204, 120, 50, 0.2)',
                            color: '#CC7832'
                          }
                        }}
                        disabled={loading}
                      >
                        <EditIcon sx={{ color: '#CC7832' }} />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
                <Typography variant="body2" sx={{ color: '#a0a0a0' }}>
                  Tailor the way Geppeto responds in this project
                </Typography>
                
                <input
                  ref={promptFileInputRef}
                  type="file"
                  style={{ display: 'none' }}
                  onChange={handlePromptFileChange}
                  accept=".txt,.md"
                />
                
                {project.system_prompt ? (
                  <Box sx={{ mt: 2, p: 2, bgcolor: 'rgba(31, 41, 55, 0.5)', borderRadius: '4px' }}>
                    <Typography variant="body2" sx={{ color: '#e0e0e0' }}>
                      {truncateText(project.system_prompt, 30)}
                    </Typography>
                  </Box>
                ) : (
                  <Typography variant="body2" sx={{ color: '#a0a0a0', mt: 2 }}>
                    {loading ? 'Cargando instrucciones...' : 'No instructions set yet'}
                  </Typography>
                )}
                
                {/* Indicador de carga superpuesto */}
                {loading && (
                  <Box sx={{ 
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    right: 0, 
                    bottom: 0, 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    bgcolor: 'rgba(0, 0, 0, 0.5)',
                    borderRadius: '12px',
                    zIndex: 10
                  }}>
                    <CircularProgress size={40} sx={{ color: '#CC7832' }} />
                  </Box>
                )}
              </Paper>

              {/* Process files button */}
              <Paper sx={{ 
                p: 4, 
                width: '100%', 
                bgcolor: '#1F2937', 
                borderRadius: '12px',
                border: '1px solid #2a2e42',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
              }}>
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ color: '#e0e0e0', fontWeight: 'bold' }}>Process Documents</Typography>
                </Box>
                <Typography variant="body2" sx={{ color: '#a0a0a0', mb: 3, textAlign: 'center' }}>
                  Generate a knowledge base from uploaded files for AI retrieval
                </Typography>
                <Tooltip title="RAG generation of the uploaded files">
                  <Button 
                    variant="contained" 
                    sx={{ 
                      backgroundColor: '#CC7832', 
                      color: '#FFFFFF', 
                      '&:hover': { backgroundColor: '#e48f4c' },
                      px: 4,
                      py: 1,
                      fontSize: '1rem'
                    }}
                    onClick={handleProcess}
                    disabled={processing}
                  >
                    {processing ? (
                      <>
                        <Box component="span" sx={{ display: 'inline-block', mr: 1 }}>
                          <CircularProgress size={20} color="inherit" />
                        </Box>
                        Generating...
                      </>
                    ) : (
                      'Generate RAG'
                    )}
                  </Button>
                </Tooltip>
              </Paper>
            </Box>
          </Box>
        </Grid>
      </Grid>

      {/* System Prompt Dialog */}
      <Dialog 
        open={systemPromptOpen} 
        onClose={handleCloseSystemPrompt} 
        maxWidth="lg" 
        fullWidth
        PaperProps={{
          sx: {
            minHeight: '70vh',
            maxHeight: '85vh'
          }
        }}
      >
        <DialogTitle 
          sx={{ 
            fontSize: '1.5rem', 
            fontWeight: 'bold', 
            color: '#e0e0e0', 
            backgroundColor: '#1F2937'
          }}
        >
          Project Instructions
        </DialogTitle>
        <DialogContent 
          sx={{ 
            pt: 3, 
            backgroundColor: '#1F2937'
          }}
        >
          <TextField
            autoFocus
            margin="dense"
            label="System Instructions"
            type="text"
            fullWidth
            multiline
            rows={16}
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="Enter instructions for how the AI should behave in this project..."
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': {
                  borderColor: '#3a3f5c',
                },
                '&:hover fieldset': {
                  borderColor: '#CC7832',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#CC7832',
                },
                backgroundColor: 'rgba(42, 46, 66, 0.5)'
              },
              '& .MuiInputLabel-root': {
                color: '#a0a0a0',
              },
              '& .MuiInputBase-input': {
                color: '#e0e0e0',
              }
            }}
          />
        </DialogContent>
        <DialogActions sx={{ backgroundColor: '#1F2937', p: 2 }}>
          <Button 
            onClick={handleCloseSystemPrompt}
            sx={{ 
              color: '#a0a0a0',
              '&:hover': { backgroundColor: 'rgba(42, 46, 66, 0.8)' }
            }}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleSaveSystemPrompt} 
            variant="contained" 
            sx={{ 
              backgroundColor: '#CC7832', 
              color: '#FFFFFF', 
              '&:hover': { backgroundColor: '#e48f4c' },
              px: 3
            }}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProjectPage; 