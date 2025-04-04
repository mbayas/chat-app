import React, { useState } from 'react';
import { Box, CssBaseline } from '@mui/material';
import { Routes, Route } from 'react-router-dom';
import ChatPage from './components/ChatPage';
import ProjectPage from './components/ProjectPage';
import { ChatProvider } from './contexts/ChatContext';
import { SettingsProvider } from './contexts/SettingsContext';
import { ProjectProvider } from './contexts/ProjectContext';

function App() {
  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <CssBaseline />
      <SettingsProvider>
        <ProjectProvider>
          <ChatProvider>
            <Routes>
              <Route path="/" element={<ChatPage />} />
              <Route path="/chat/:chatId" element={<ChatPage />} />
              <Route path="/project/:projectId" element={<ProjectPage />} />
            </Routes>
          </ChatProvider>
        </ProjectProvider>
      </SettingsProvider>
    </Box>
  );
}

export default App; 