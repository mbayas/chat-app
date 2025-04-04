import React, { createContext, useState, useEffect, useContext, ReactNode } from 'react';
import axios from 'axios';

// Define types
export interface ProjectFile {
  id: string;
  name: string;
  original_name: string;
  size: number;
  uploaded_at: string;
  url: string;
}

export interface Project {
  id: string;
  title: string;
  created_at: string;
  last_updated: string;
  system_prompt: string;
  files: ProjectFile[];
}

interface ProjectContextType {
  projects: Project[];
  currentProject: string | null;
  loading: boolean;
  error: string | null;
  createNewProject: (title: string, systemPrompt?: string) => Promise<string>;
  selectProject: (projectId: string) => Promise<Project | null>;
  deleteProject: (projectId: string) => Promise<void>;
  updateProject: (projectId: string, title: string) => Promise<void>;
  uploadProjectFile: (projectId: string, file: File) => Promise<void>;
  deleteProjectFile: (projectId: string, fileId: string) => Promise<void>;
  updateSystemPrompt: (projectId: string, prompt: string) => Promise<void>;
  uploadSystemPromptFile: (projectId: string, file: File) => Promise<void>;
  getAllProjectNames: () => string[];
  getProjectById: (projectId: string) => Project | undefined;
}

// Create context
const ProjectContext = createContext<ProjectContextType | undefined>(undefined);

// Provider component
export const ProjectProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [currentProject, setCurrentProject] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch projects on component mount
  useEffect(() => {
    fetchProjects();
  }, []);

  // Fetch all projects
  const fetchProjects = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/projects');
      setProjects(response.data);
      setError(null);
      return response.data;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      console.error('Error fetching projects:', err);
      return [];
    } finally {
      setLoading(false);
    }
  };

  // Create a new project
  const createNewProject = async (title: string, systemPrompt: string = ''): Promise<string> => {
    try {
      setLoading(true);
      const response = await axios.post('/api/projects', {
        title: title || 'New Project',
        system_prompt: systemPrompt,
      });
      
      await fetchProjects();
      setCurrentProject(response.data.id);
      return response.data.id;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create project');
      console.error('Error creating project:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Select a project
  const selectProject = async (projectId: string) => {
    try {
      setLoading(true);
      const response = await axios.get(`/api/projects/${projectId}`);
      if (response.status === 200) {
        setCurrentProject(projectId);
        setProjects(prev => 
          prev.map(p => p.id === projectId ? response.data : p)
        );
        setError(null);
        return response.data;
      } else {
        setError('Failed to fetch project');
        console.error('Error selecting project: Project not found');
        return null;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to select project');
      console.error('Error selecting project:', err);
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Delete a project
  const deleteProject = async (projectId: string) => {
    try {
      setLoading(true);
      await axios.delete(`/api/projects/${projectId}`);
      
      // If the deleted project is the current project, clear selection
      if (projectId === currentProject) {
        setCurrentProject(null);
      }
      
      // Refresh project list
      await fetchProjects();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete project');
      console.error('Error deleting project:', err);
    } finally {
      setLoading(false);
    }
  };

  // Update a project
  const updateProject = async (projectId: string, title: string) => {
    try {
      setLoading(true);
      const response = await axios.put(`/api/projects/${projectId}`, { title });
      
      // Refresh project list
      await fetchProjects();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update project');
      console.error('Error updating project:', err);
    } finally {
      setLoading(false);
    }
  };

  // Upload a file to a project
  const uploadProjectFile = async (projectId: string, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('project_id', projectId);

    try {
      setLoading(true);
      const response = await axios.post('/api/projects/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      console.log("Archivo subido correctamente:", response.data);
      
      // Refresh project list to get updated files
      await fetchProjects();
      
      // Refresh specific project data to ensure UI updates
      try {
        const projectResponse = await axios.get(`/api/projects/${projectId}`);
        if (projectResponse.status === 200) {
          // Update the project in the list
          setProjects(prev => 
            prev.map(p => p.id === projectId ? projectResponse.data : p)
          );
          console.log("Datos del proyecto actualizados después de subir archivo:", projectResponse.data);
        }
      } catch (projectErr) {
        console.error("Error al actualizar datos específicos del proyecto:", projectErr);
      }
      
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload file');
      console.error('Error uploading file to project:', err);
    } finally {
      setLoading(false);
    }
  };

  // Delete a file from a project
  const deleteProjectFile = async (projectId: string, fileId: string) => {
    try {
      setLoading(true);
      await axios.delete(`/api/projects/${projectId}/files/${fileId}`);
      
      console.log("Archivo eliminado correctamente, ID:", fileId);
      
      // Refresh project list to update files
      await fetchProjects();
      
      // Refresh specific project data to ensure UI updates
      try {
        const projectResponse = await axios.get(`/api/projects/${projectId}`);
        if (projectResponse.status === 200) {
          // Update the project in the list
          setProjects(prev => 
            prev.map(p => p.id === projectId ? projectResponse.data : p)
          );
          console.log("Datos del proyecto actualizados después de eliminar archivo:", projectResponse.data);
        }
      } catch (projectErr) {
        console.error("Error al actualizar datos específicos del proyecto:", projectErr);
      }
      
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete file');
      console.error('Error deleting file from project:', err);
    } finally {
      setLoading(false);
    }
  };

  // Update system prompt for a project
  const updateSystemPrompt = async (projectId: string, prompt: string) => {
    try {
      setLoading(true);
      await axios.put(`/api/projects/${projectId}/system-prompt`, { prompt });
      
      // Refresh project list to update system prompt
      await fetchProjects();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update system prompt');
      console.error('Error updating system prompt:', err);
    } finally {
      setLoading(false);
    }
  };

  // Upload a file as system prompt
  const uploadSystemPromptFile = async (projectId: string, file: File) => {
    try {
      setLoading(true);
      
      // Read the file content as text
      const fileContent = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target && typeof event.target.result === 'string') {
            resolve(event.target.result as string);
          } else {
            reject(new Error('Failed to read file content'));
          }
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
      });
      
      console.log(`Leyendo archivo ${file.name} como sistema prompt, tamaño: ${file.size} bytes`);
      
      // Update the system prompt with the file content
      await updateSystemPrompt(projectId, fileContent);
      
      // Refresh specific project data to ensure UI updates
      try {
        const projectResponse = await axios.get(`/api/projects/${projectId}`);
        if (projectResponse.status === 200) {
          // Update the project in the list
          setProjects(prev => 
            prev.map(p => p.id === projectId ? projectResponse.data : p)
          );
          console.log("Datos del proyecto actualizados después de cargar sistema prompt:", projectResponse.data);
        }
      } catch (projectErr) {
        console.error("Error al actualizar datos específicos del proyecto:", projectErr);
      }
      
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload system prompt file');
      console.error('Error uploading system prompt file:', err);
    } finally {
      setLoading(false);
    }
  };

  // Get all project names (useful for RAG queries with **ProjectName**)
  const getAllProjectNames = (): string[] => {
    return projects.map(project => project.title);
  };
  
  // Get a project by its ID
  const getProjectById = (projectId: string): Project | undefined => {
    return projects.find(project => project.id === projectId);
  };

  const value = {
    projects,
    currentProject,
    loading,
    error,
    createNewProject,
    selectProject,
    deleteProject,
    updateProject,
    uploadProjectFile,
    deleteProjectFile,
    updateSystemPrompt,
    uploadSystemPromptFile,
    getAllProjectNames,
    getProjectById
  };

  return (
    <ProjectContext.Provider value={value}>
      {children}
    </ProjectContext.Provider>
  );
};

// Custom hook to use the project context
export const useProject = () => {
  const context = useContext(ProjectContext);
  if (context === undefined) {
    throw new Error('useProject must be used within a ProjectProvider');
  }
  return context;
}; 