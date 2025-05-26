# RAG Architectures Comparison

A tool for comparing different Retrieval-Augmented Generation (RAG) architectures using various LLM providers.

## Deployed URL


## Features

- PDF document upload and processing
- Three RAG architectures: Basic, Self-Query, and Reranker
- Multiple LLM provider options: Groq, Google Gemini, and OpenRouter
- Configurable model parameters (temperature)
- Side-by-side comparison of RAG implementations


## Setup

### Environment Variables

Export API keys as follows:
```bash
export GROQ_API_KEY=api_key
export GOOGLE_API_KEY=api_key
export OPENROUTER_API_KEY=api_key
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
python [main.py](http://_vscodecontentref_/4)
```

### Frontwnd Setup

```bash
npm install
npm run dev
```