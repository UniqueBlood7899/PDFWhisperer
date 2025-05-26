from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import os
from typing import List, Optional
import uuid
from rag_implementations import RAGFactory

app = FastAPI(title="RAG Comparison API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for uploaded PDFs and their embeddings
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize RAG factory
rag_factory = RAGFactory(UPLOAD_DIR)

# In-memory storage for documents (in production, use a database)
documents = {}

class QueryRequest(BaseModel):
    document_id: str
    query: str
    rag_types: List[str] = ["basic", "self_query", "reranker"]

class ChunkInfo(BaseModel):
    text: str
    page: int
    score: Optional[float] = None

class RAGResponse(BaseModel):
    answer: str
    chunks: List[ChunkInfo]
    time: float

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
    
    try:
        # Process with Basic RAG
        if "basic" in request.rag_types:
            basic_result = rag_factory.basic_rag(request.document_id, request.query)
            response.basic = RAGResponse(
                answer=basic_result["answer"],
                chunks=[ChunkInfo(**chunk) for chunk in basic_result["chunks"]],
                time=basic_result["time"]
            )
        
        # Process with Self-Query RAG
        if "self_query" in request.rag_types:
            self_query_result = rag_factory.self_query_rag(request.document_id, request.query)
            response.self_query = RAGResponse(
                answer=self_query_result["answer"],
                chunks=[ChunkInfo(**chunk) for chunk in self_query_result["chunks"]],
                time=self_query_result["time"]
            )
        
        # Process with Reranker RAG
        if "reranker" in request.rag_types:
            reranker_result = rag_factory.reranker_rag(request.document_id, request.query)
            response.reranker = RAGResponse(
                answer=reranker_result["answer"],
                chunks=[ChunkInfo(**chunk) for chunk in reranker_result["chunks"]],
                time=reranker_result["time"]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)