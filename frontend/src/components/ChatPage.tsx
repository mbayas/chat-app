import React, { useState } from 'react';
import { Box, Grid, useMediaQuery, useTheme } from '@mui/material';
import Sidebar from './Sidebar';
import ChatWindow from './ChatWindow';
import { useChat } from '../contexts/ChatContext';

const ChatPage: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const { currentChat } = useChat();

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

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
            bgcolor: '#0a0c10',
            borderRight: 'none',
          }}
        >
          <Sidebar onClose={() => isMobile && setSidebarOpen(false)} />
        </Grid>

        {/* Chat Window */}
        <Grid
          item
          xs
          sx={{
            height: '100%',
            width: sidebarOpen && !isMobile ? 'calc(100% - 300px)' : '100%',
          }}
        >
          <ChatWindow 
            sidebarOpen={sidebarOpen} 
            toggleSidebar={toggleSidebar} 
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default ChatPage; 