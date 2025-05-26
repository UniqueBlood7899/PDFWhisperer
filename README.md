# RAG Architectures Comparison

A web application for comparing different Retrieval-Augmented Generation (RAG) architectures using various LLM providers. Upload your PDF documents and ask questions to see how different RAG implementations respond.

## Features

- PDF document upload and processing
- Side-by-side comparison of three RAG architectures:
  - Basic RAG
  - Self-Query RAG
  - Reranker-enhanced RAG
- Multiple LLM provider integrations:
  - Groq
  - Google Gemini
  - OpenRouter (access to various models)
- Configurable model parameters (temperature)
- Visualization of retrieved document chunks with relevance scores
- Modern UI built with Next.js and Tailwind CSS

## Architecture

The application consists of:

- **Frontend**: Next.js application with a clean UI built using shadcn/ui components
- **Backend**: FastAPI server that handles:
  - PDF document processing
  - Vector embeddings (using Qdrant)
  - RAG implementation variants
  - LLM provider integrations

## Setup

### Environment Variables

Create a `.env` file based on `.env.example` with the following variables:

```bash
# LLM API Keys
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
NEXT_PUBLIC_API_URL=http://localhost:8000
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Modal Configuration
USE_MODAL=true
```

You only need to provide API keys for the LLM providers you plan to use.

- [Get a Groq API key](https://console.groq.com/)
- [Get a Google AI API key](https://ai.google.dev/)
- [Get an OpenRouter API key](https://openrouter.ai/)

### Backend Setup

```bash
cd backend
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontwnd Setup

```bash
npm install
npm run dev
```

Visit http://localhost:3000 to access the application.

### Usage
- Upload a PDF document
- Wait for processing to complete
- Enter your question about the document
- Select your preferred LLM provider and model
- Adjust temperature setting if desired
- Click "Submit" to see responses from all three RAG architectures
- Compare the results, including the document chunks each RAG method retrieved

### Deployment
The application is configured for deployment on Vercel (frontend) and Railway (backend):

- Frontend: Static export with next build (configured in next.config.js)
- Backend: Python FastAPI service with dependencies specified in requirements.txt

### Technologies
- Frontend: Next.js, React, Tailwind CSS, shadcn/ui
- Backend: FastAPI, LangChain, PyPDF, Sentence Transformers
- Vector Database: Qdrant
- LLM Providers: Groq, Google Gemini, OpenRouter