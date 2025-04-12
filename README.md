# Geppetto

Geppetto is a chat application developed in React, Node.js, and Python.
It allows users to interact with different LLM providers (OpenAI, Google, Ollama) and create projects where users can upload their files and make queries about them using RAG.
It is oriented towards people who want to interact with multiple LLM models and measure their performance in terms of response time, resource consumption, and tokens used.

## Requirements

### Minimum Required Versions
- Python 3.8+ (recommended 3.10+)
- Node.js 16+ (recommended 18+ LTS)
- npm 8+ or yarn 1.22+

### Core Dependencies
- React 18.2+
- TypeScript 4.9+
- Material-UI (MUI) 5.11+

### Supported Browsers
- Chrome (latest version)
- Firefox (latest version)
- Safari (latest version)

## Project Structure

- `/frontend` - React Application
- `/backend` - Python Server

After installing the project, the directory structure will be as follows:

```
chat-app/
├── README.md
├── cleanup_projects.html
├── faiss_explorer.py
├── backend/
│   ├── app.py
│   ├── config.py
│   └── llm_service.py
└── frontend/
    ├── src/
    └── public/

```
When running the project, the directory structure will be as follows:


```
chat-app/
├── README.md
├── .DS_Store
├── cleanup_projects.html
├── faiss_explorer.py
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── llm_service.py
│   └── uploads/
├── frontend/
│   ├── src/
│   ├── public/
│   ├── build/
│   └── node_modules/
├── utils/
├── docs/
├── db/
│   └── [proyectos]/
│       ├── index.faiss
│       └── index.pkl
├── chats/
│   ├── [chat_id].json
│   └── [chat_id]_metadata.json
├── files/
└── uploads/
```


## Project Structure Description

### Main Directories

- `/backend`: Python Flask Server
  - `app.py`: Main application
  - `config.py`: Configurations
  - `llm_service.py`: LLM service
  - `uploads/`: Temporary files and PDFs generated

- `/frontend`: React Application
  - `src/`: Source code
  - `public/`: Static files
  - `build/`: Production build
  - `node_modules/`: Node.js dependencies

- 
## Requirements

Explanation of the main components:
1. Web Framework:
    - flask: The main web framework
    - flask-cors: For handling CORS in requests
2. LLM Providers:
    - openai: Official OpenAI client
    - google-generativeai: Official Google client for Gemini
3. Document Processing and RAG:
    - langchain: Framework for working with LLMs
    - langchain-community: Community components for LangChain
    - langchain-openai: OpenAI integrations
  
- `/db`: FAISS database
  - Subdirectories by project with FAISS indices

- `/chats`: Chat history
  - JSON files of conversations
  - Metadata files

- `/docs`: Project documentation
- `/utils`: Utilities and tools
- `/files` and `/uploads`: File storage

### Root Files

- `README.md`: Main documentation
- `.gitignore`: Git configuration
- `fix_metadata.py`: Utility script
- `faiss_explorer.py`: FAISS tool
- `cleanup_projects.html`: Cleanup utility
    - faiss-cpu: For vector similarity search
    - pypdf: For processing PDF files
4. PDF and Markdown Generation:
    - reportlab: For generating PDFs
    - markdown2: For converting markdown to HTML
5. Utilities:
    - python-magic: For MIME type detection
    - requests: For HTTP requests
    - typing-extensions: For type annotations



## Dependencies

### Main Packages
```python
# Framework Web
flask==2.3.3
werkzeug==2.3.7
flask-cors==3.0.10
gunicorn==20.1.0

# LLM Providers
openai==1.65.1
google-generativeai==0.3.2

# LangChain and RAG
langchain==0.3.21
langchain-community==0.3.20
langchain-openai==0.3.10
faiss-cpu

# PDF and Image Processing
reportlab==4.3.1
pillow==11.1.0
markdown2

# Utilities
python-dotenv==1.0.0
python-magic==0.4.27
requests==2.31.0
rich==13.9.4
typing-extensions==4.12.2
```

## Execution

### Backend
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
cd backend
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```
## Configuration

1. Copy `.env.example` to `.env` in the backend directory
2. Configure your API keys for Google and OpenAI in the .env file

## Execution

### Backend
```bash
cd backend
python app.py
```

### Frontend
```bash
cd frontend
npm start
```

### Utils
```bash
python faiss_explorer.py # Explore FAISS database
cleanup_projects.html # Cleanup projects, open in browser
```
