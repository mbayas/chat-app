import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { ProjectProvider } from './contexts/ProjectContext';

// Create a custom theme inspired by Stable Diffusion UI
const stableDiffusionTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#CC7832', // Orange accent color (changed from blue)
    },
    secondary: {
      main: '#5d6483', // Muted blue-gray for secondary elements
    },
    background: {
      default: '#0B0F18', // Very dark blue-black background
      paper: '#1F2937', // Dark slate for cards/paper elements
    },
    text: {
      primary: '#e0e0e0', // Light gray for primary text
      secondary: '#a0a0a0', // Darker gray for secondary text
    },
    divider: '#2a2e42', // Subtle divider color
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    fontSize: 14,
    h6: {
      fontWeight: 500,
    },
    body1: {
      fontSize: '0.9rem',
    },
    caption: {
      fontSize: '0.8rem',
    },
  },
  shape: {
    borderRadius: 6, // Slightly rounded corners
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none', // Remove the default gradient
        },
      },
    },
    MuiInputBase: {
      styleOverrides: {
        root: {
          backgroundColor: '#1F2937', // Dark slate input fields
          borderRadius: 6,
          '&.MuiInputBase-multiline': {
            padding: '8px',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: '#1F2937', // Dark slate input background
            '& fieldset': {
              borderColor: '#2a2e42', // Subtle border
            },
            '&:hover fieldset': {
              borderColor: '#3f4562', // Slightly brighter border on hover
            },
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none', // No uppercase text
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          color: '#a0a0a0', // Default color for icon buttons
          '&:hover': {
            backgroundColor: 'rgba(204, 120, 50, 0.1)', // Subtle hover effect with orange
          },
        },
      },
    },
    MuiLink: {
      styleOverrides: {
        root: {
          color: '#CC7832', // Color solicitado para los enlaces
          textDecoration: 'none',
          '&:hover': {
            color: '#e48f4c', // Versión más clara para hover
            textDecoration: 'underline',
          },
        },
      },
    },
  },
});

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider theme={stableDiffusionTheme}>
        <CssBaseline />
        <ProjectProvider>
          <App />
        </ProjectProvider>
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals(); 