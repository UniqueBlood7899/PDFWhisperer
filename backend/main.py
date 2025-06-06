from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Literal
import os
import uuid
import uvicorn
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(dotenv_path='../.env')

# Log important startup information
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Python version: {sys.version}")
logger.info(f"GROQ_API_KEY set: {'Yes' if os.environ.get('GROQ_API_KEY') else 'No'}")
logger.info(f"GOOGLE_API_KEY set: {'Yes' if os.environ.get('GOOGLE_API_KEY') else 'No'}")
logger.info(f"OPENROUTER_API_KEY set: {'Yes' if os.environ.get('OPENROUTER_API_KEY') else 'No'}")

from rag_implementations import RAGFactory, LLMProvider

app = FastAPI(title="RAG Comparison API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pdf-whisperer-psi.vercel.app", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for uploaded PDFs and their embeddings
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
# Create absolute path to ensure consistent directory location
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)
# Create directory with permissive permissions
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"Upload directory: {UPLOAD_DIR}")
try:
    # Test write permissions
    test_file = os.path.join(UPLOAD_DIR, ".write_test")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    logger.info("Upload directory is writable")
except Exception as e:
    logger.error(f"Upload directory may not be writable: {str(e)}")

# Initialize RAG factory
rag_factory = RAGFactory(UPLOAD_DIR)

# In-memory storage for documents (in production, use a database)
documents = {}

class QueryRequest(BaseModel):
    document_id: str
    query: str
    rag_types: List[str]  # ["basic", "self_query", "reranker"]
    llm_provider: str = LLMProvider.GROQ  # Default to Groq
    llm_model: Optional[str] = None
    llm_temperature: float = 0.1

class ChunkInfo(BaseModel):
    text: str
    page: int
    score: Optional[float] = None

class RAGResponse(BaseModel):
    answer: str
    chunks: List[ChunkInfo]
    time: float
    llm_provider: str

class QueryResponse(BaseModel):
    basic: Optional[RAGResponse] = None
    self_query: Optional[RAGResponse] = None
    reranker: Optional[RAGResponse] = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate a unique ID for the document
    document_id = str(uuid.uuid4())
    
    # Save the file
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Store document info
    documents[document_id] = {
        "filename": file.filename,
        "path": file_path,
        "processed": False
    }
    
    # Process the PDF using RAG Factory
    try:
        success = rag_factory.process_pdf(file_path, document_id)
        if success:
            documents[document_id]["processed"] = True
        else:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
    except Exception as e:
        # Improved error handling
        error_message = f"Error processing PDF: {str(e)}"
        print(error_message)
        # Clean up the file if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)
        # Remove document from memory
        if document_id in documents:
            del documents[document_id]
        raise HTTPException(status_code=500, detail=error_message)
    
    return {"document_id": document_id, "filename": file.filename, "status": "processed"}

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Query a document using specified RAG architectures"""
    if request.document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not documents[request.document_id]["processed"]:
        raise HTTPException(status_code=400, detail="Document is still processing")
    
    response = QueryResponse()
    
    # Prepare LLM kwargs from request
    llm_kwargs = {}
    if request.llm_model:
        llm_kwargs["model"] = request.llm_model
    llm_kwargs["temperature"] = request.llm_temperature
    
    try:
        # Process with Basic RAG
        if "basic" in request.rag_types:
            basic_result = rag_factory.basic_rag(
                request.document_id, 
                request.query,
                llm_provider=request.llm_provider,
                llm_kwargs=llm_kwargs
            )
            response.basic = RAGResponse(
                answer=basic_result["answer"],
                chunks=[ChunkInfo(**chunk) for chunk in basic_result["chunks"]],
                time=basic_result["time"],
                llm_provider=basic_result["llm_provider"]
            )
        
        # Process with Self-Query RAG
        if "self_query" in request.rag_types:
            self_query_result = rag_factory.self_query_rag(
                request.document_id, 
                request.query,
                llm_provider=request.llm_provider,
                llm_kwargs=llm_kwargs
            )
            response.self_query = RAGResponse(
                answer=self_query_result["answer"],
                chunks=[ChunkInfo(**chunk) for chunk in self_query_result["chunks"]],
                time=self_query_result["time"],
                llm_provider=self_query_result["llm_provider"]
            )
        
        # Process with Reranker RAG
        if "reranker" in request.rag_types:
            reranker_result = rag_factory.reranker_rag(
                request.document_id, 
                request.query,
                llm_provider=request.llm_provider,
                llm_kwargs=llm_kwargs
            )
            response.reranker = RAGResponse(
                answer=reranker_result["answer"],
                chunks=[ChunkInfo(**chunk) for chunk in reranker_result["chunks"]],
                time=reranker_result["time"],
                llm_provider=reranker_result["llm_provider"]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    return response

# Add an endpoint to list available LLM providers and models
@app.get("/llm-providers")
async def get_llm_providers():
    """Get available LLM providers and their models"""
    providers = {
        LLMProvider.GROQ: {
            "name": "Groq",
            "models": [
                {"id": "llama3-8b-8192", "name": "Llama 3 8B"},
                {"id": "llama3-70b-8192", "name": "Llama 3 70B"}
            ]
        },
        LLMProvider.GEMINI: {
            "name": "Google Gemini",
            "models": [
                {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
                {"id": "gemini-1.5-flash-8b", "name": "Gemini 1.5 Flash 8B"},
            ]
        },
        LLMProvider.OPENROUTER: {
            "name": "OpenRouter",
            "models": [
                {"id": "mistralai/mistral-7b-instruct", "name": "Mistral 7B"},
                {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku"},
                {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus"},
                {"id": "meta-llama/llama-3-70b-instruct", "name": "Llama 3 70B (Meta)"}
            ]
        }
    }
    
    return providers

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "healthy",
        "api_version": "1.0",
        "environment": {
            "groq_api_key": bool(os.environ.get("GROQ_API_KEY")),
            "google_api_key": bool(os.environ.get("GOOGLE_API_KEY")),
            "openrouter_api_key": bool(os.environ.get("OPENROUTER_API_KEY")),
            "upload_dir": UPLOAD_DIR,
            "upload_dir_writable": True  # We already tested this above
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)