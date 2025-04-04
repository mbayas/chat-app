import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface Settings {
  model: string;
  temperature: number;
  max_tokens: number;
  provider: string;
  ollama_base_url?: string;
  maintain_context: boolean;
  google_api_key?: string;
}

interface SettingsContextType {
  settings: Settings;
  availableModels: string[];
  updateSettings: (newSettings: Settings) => Promise<void>;
  updateApiKey: (apiKey: string, provider?: string) => Promise<void>;
}

const defaultSettings: Settings = {
  model: 'gpt-3.5-turbo',
  temperature: 0.7,
  max_tokens: 800,
  provider: 'openai',
  ollama_base_url: 'http://localhost:11434',
  maintain_context: true
};

// Default models as fallback if API fails
const DEFAULT_OPENAI_MODELS = [
  'gpt-3.5-turbo',
  'gpt-4',
  'gpt-4-turbo-preview'
];

const DEFAULT_OLLAMA_MODELS = [
  'llama3.2:3b',
  'llama3.2:1b',
  'llama2',
  'mistral',
  'mixtral',
  'phi3',
  'gemma',
  'codellama',
  'llama2:13b',
  'llama2:70b',
  'neural-chat'
];

const DEFAULT_GOOGLE_MODELS = [
  'gemini-1.5-pro',
  'gemini-1.5-flash',
  'gemini-1.0-pro',
  'gemini-1.0-ultra'
];

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export const SettingsProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [availableModels, setAvailableModels] = useState<string[]>(DEFAULT_OPENAI_MODELS);

  // Fetch settings and models when component mounts
  useEffect(() => {
    const initializeSettings = async () => {
      await fetchSettings();
    };
    initializeSettings();
  }, []);

  // Fetch models whenever provider changes
  useEffect(() => {
    const updateModelsForProvider = async () => {
      await fetchModels();
    };
    updateModelsForProvider();
  }, [settings.provider]);

  const fetchSettings = async () => {
    try {
      const response = await fetch('/api/settings');
      if (response.ok) {
        const data = await response.json();
        setSettings(data);
      }
    } catch (error) {
      console.error('Error fetching settings:', error);
    }
  };

  const fetchModels = async () => {
    console.log('Fetching models for provider:', settings.provider); // Debug log

    // Establecer modelos por defecto inmediatamente según el proveedor
    const defaultModels = settings.provider === 'openai' ? DEFAULT_OPENAI_MODELS : settings.provider === 'ollama' ? DEFAULT_OLLAMA_MODELS : DEFAULT_GOOGLE_MODELS;
    setAvailableModels(defaultModels);
    console.log('Set initial default models:', defaultModels); // Debug log

    try {
      const response = await fetch('/api/models');
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }

      const data = await response.json();
      console.log('API response for models:', data); // Debug log

      if (data?.models && Array.isArray(data.models) && data.models.length > 0) {
        // Verificar que los modelos correspondan al proveedor actual
        const currentProvider = settings.provider;
        console.log('Current provider when setting models:', currentProvider); // Debug log

        // Solo actualizar si seguimos en el mismo proveedor
        if (currentProvider === settings.provider) {
          setAvailableModels(data.models);
          console.log('Updated models from API:', data.models); // Debug log
        }
      }
    } catch (error) {
      console.error('Error fetching models:', error);
      // Los modelos por defecto ya están establecidos, así que no necesitamos hacer nada más
    }
  };

  const updateSettings = async (newSettings: Settings) => {
    try {
      // Si el proveedor está cambiando, actualizar los modelos por defecto inmediatamente
      if (newSettings.provider !== settings.provider) {
        const defaultModels = newSettings.provider === 'openai' ? DEFAULT_OPENAI_MODELS : newSettings.provider === 'ollama' ? DEFAULT_OLLAMA_MODELS : DEFAULT_GOOGLE_MODELS;
        setAvailableModels(defaultModels);
        console.log('Provider changing, setting default models:', defaultModels); // Debug log
      }

      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newSettings),
      });

      if (response.ok) {
        // Actualizar settings después de una respuesta exitosa
        setSettings(newSettings);
        console.log('Settings updated successfully:', newSettings); // Debug log
      } else {
        throw new Error('Failed to update settings');
      }
    } catch (error) {
      console.error('Error updating settings:', error);
      // Revertir a los modelos por defecto del proveedor actual en caso de error
      const defaultModels = settings.provider === 'openai' ? DEFAULT_OPENAI_MODELS : settings.provider === 'ollama' ? DEFAULT_OLLAMA_MODELS : DEFAULT_GOOGLE_MODELS;
      setAvailableModels(defaultModels);
    }
  };

  const updateApiKey = async (apiKey: string, provider?: string) => {
    try {
      const response = await fetch('/api/settings/api-key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          api_key: apiKey,
          provider: provider || settings.provider 
        }),
      });

      if (response.ok) {
        // Actualizar modelos después de actualizar la API key
        await fetchModels();
      }
    } catch (error) {
      console.error('Error updating API key:', error);
    }
  };

  return (
    <SettingsContext.Provider value={{ settings, availableModels, updateSettings, updateApiKey }}>
      {children}
    </SettingsContext.Provider>
  );
};

export const useSettings = () => {
  const context = useContext(SettingsContext);
  if (context === undefined) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
}; 