import os
from typing import List, Dict, Any, Optional
import time
import traceback
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseChatModel
import qdrant_client

# More detailed error printing
def print_error(e: Exception, message: str = "Error"):
    print(f"{message}: {str(e)}")
    traceback.print_exc()

# Initialize embedding model with fallback
try:
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print("Embedding model initialized successfully")
except Exception as e:
    print_error(e, "Failed to load primary embedding model")
    print("Falling back to a simpler model...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Fallback embedding model initialized successfully")
    except Exception as e:
        print_error(e, "Failed to load fallback embedding model")
        raise

# Define LLM provider options
class LLMProvider:
    GROQ = "groq"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    
    @staticmethod
    def get_llm(provider: str, **kwargs) -> BaseChatModel:
        """Get LLM instance based on provider name"""
        if provider == LLMProvider.GROQ:
            api_key = os.environ.get("GROQ_API_KEY", "")
            model = kwargs.get("model", "llama3-8b-8192")
            temperature = kwargs.get("temperature", 0.1)
            
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
                
            try:
                return ChatGroq(
                    api_key=api_key,
                    model=model,
                    temperature=temperature
                )
            except Exception as e:
                print_error(e, f"Error initializing Groq with model {model}")
                # Fall back to a default model
                return ChatGroq(
                    api_key=api_key,
                    model="llama3-8b-8192",  # Default fallback
                    temperature=temperature
                )
            
        elif provider == LLMProvider.GEMINI:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            # Updated model name - gemini-1.5-pro is the newer model
            model = kwargs.get("model", "gemini-1.5-pro")
            temperature = kwargs.get("temperature", 0.1)
            
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
                
            try:
                # Initialize with more explicit parameters
                return ChatGoogleGenerativeAI(
                    google_api_key=api_key,
                    model=model,
                    temperature=temperature,
                    convert_system_message_to_human=True  # Handle system messages properly
                )
            except Exception as e:
                print_error(e, f"Error initializing Gemini with model {model}")
                print("Trying alternate model name format...")
                try:
                    # Try with alternate model name
                    return ChatGoogleGenerativeAI(
                        google_api_key=api_key,
                        model="gemini-pro",  # Try older model name
                        temperature=temperature,
                        convert_system_message_to_human=True
                    )
                except Exception as e2:
                    print_error(e2, "Error with alternate model as well")
                    # Fall back to Groq if Gemini fails
                    print("Falling back to Groq due to Gemini API issues")
                    groq_api_key = os.environ.get("GROQ_API_KEY", "")
                    if groq_api_key:
                        return ChatGroq(
                            api_key=groq_api_key,
                            model="llama3-8b-8192",
                            temperature=temperature
                        )
                    else:
                        raise ValueError("Gemini failed and no GROQ_API_KEY for fallback")
        
        elif provider == LLMProvider.OPENROUTER:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            model = kwargs.get("model", "mistralai/mistral-7b-instruct")
            temperature = kwargs.get("temperature", 0.1)
            
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
                
            try:
                # OpenRouter can be accessed through the ChatOpenAI class with base URL override
                return ChatOpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    model=model,
                    temperature=temperature
                )
            except Exception as e:
                print_error(e, f"Error initializing OpenRouter with model {model}")
                # Try a different model if the specified one fails
                try:
                    return ChatOpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                        model="mistralai/mistral-7b-instruct",  # Fallback to reliable model
                        temperature=temperature
                    )
                except Exception as e2:
                    print_error(e2, "Error with fallback model as well")
                    # Fall back to Groq if OpenRouter fails
                    print("Falling back to Groq due to OpenRouter API issues")
                    groq_api_key = os.environ.get("GROQ_API_KEY", "")
                    if groq_api_key:
                        return ChatGroq(
                            api_key=groq_api_key,
                            model="llama3-8b-8192",
                            temperature=temperature
                        )
                    else:
                        raise ValueError("OpenRouter failed and no GROQ_API_KEY for fallback")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

class RAGFactory:
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir
        print(f"Initializing RAGFactory with upload directory: {upload_dir}")
        try:
            self.qdrant_client = qdrant_client.QdrantClient(":memory:")  # In-memory for development
            print("Qdrant client initialized successfully")
        except Exception as e:
            print_error(e, "Failed to initialize Qdrant client")
            raise
        self.document_stores = {}
    
    def process_pdf(self, file_path: str, document_id: str) -> bool:
        """Process a PDF and store its embeddings"""
        print(f"Processing PDF file: {file_path} with document_id: {document_id}")
        
        if not os.path.exists(file_path):
            print(f"Error: File not found at path {file_path}")
            return False
            
        try:
            # Load PDF
            print("Loading PDF document...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            print(f"Successfully loaded {len(documents)} pages from PDF")
            
            # Split into chunks
            print("Splitting document into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} text chunks")
            
            # Check if collection exists and create it if not
            try:
                print(f"Checking if collection {document_id} exists...")
                self.qdrant_client.get_collection(collection_name=document_id)
                print(f"Collection {document_id} already exists")
            except ValueError:
                print(f"Collection {document_id} does not exist, creating it...")
                # Get vector size from embeddings
                sample_embedding = embeddings.embed_query("Sample text")
                vector_size = len(sample_embedding)
                
                # Create the collection with proper vector configuration
                self.qdrant_client.create_collection(
                    collection_name=document_id,
                    vectors_config={
                        "size": vector_size,
                        "distance": "Cosine"  # You can use "Cosine", "Euclid" or "Dot"
                    }
                )
                print(f"Collection {document_id} created successfully")
        
            # Create vector store
            print("Creating vector store...")
            vectorstore = Qdrant(
                client=self.qdrant_client,
                collection_name=document_id,
                embeddings=embeddings
            )
            
            # Add documents to vector store
            print("Adding documents to vector store...")
            vectorstore.add_documents(chunks)
            print("Documents successfully added to vector store")
            
            # Store reference to vector store
            self.document_stores[document_id] = {
                "vectorstore": vectorstore,
                "chunks": chunks
            }
            print(f"Document with ID {document_id} processed successfully")
            
            return True
        except Exception as e:
            print_error(e, f"Error processing PDF {file_path}")
            return False
    
    def basic_rag(self, document_id: str, query: str, llm_provider: str = "groq", llm_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Basic RAG implementation"""
        print(f"Basic RAG query for document {document_id}: {query} using {llm_provider}")
        start_time = time.time()
        
        # Get vector store
        if document_id not in self.document_stores:
            error_msg = f"Document ID {document_id} not found"
            print(error_msg)
            raise ValueError(error_msg)
        
        vectorstore = self.document_stores[document_id]["vectorstore"]
        
        try:
            # Initialize LLM
            llm_kwargs = llm_kwargs or {}
            llm = LLMProvider.get_llm(llm_provider, **llm_kwargs)
            
            # Create retriever
            print("Creating retriever...")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Execute query to get documents
            print("Retrieving relevant documents...")
            retrieved_docs = retriever.invoke(query)
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            # Format the context
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # Create RAG prompt template
            rag_prompt = f"""Answer the question based only on the following context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            # Execute query directly with the formatted prompt
            print("Generating answer...")
            answer = llm.invoke(rag_prompt)
            if hasattr(answer, 'content'):  # For ChatModels that return messages
                answer = answer.content
            print("Answer generated successfully")
            
            execution_time = time.time() - start_time
            
            # Prepare chunks info
            chunks = []
            for i, doc in enumerate(retrieved_docs):
                chunks.append({
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "score": 0.9 - (i * 0.1)  # Mock score
                })
            
            return {
                "answer": answer,
                "chunks": chunks,
                "time": execution_time,
                "llm_provider": llm_provider
            }
        except Exception as e:
            print_error(e, f"Error in basic RAG with {llm_provider}")
            raise
    
    def self_query_rag(self, document_id: str, query: str, llm_provider: str = "groq", llm_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Self-query RAG implementation"""
        print(f"Self-query RAG for document {document_id}: {query} using {llm_provider}")
        start_time = time.time()
        
        # Get vector store
        if document_id not in self.document_stores:
            error_msg = f"Document ID {document_id} not found"
            print(error_msg)
            raise ValueError(error_msg)
        
        vectorstore = self.document_stores[document_id]["vectorstore"]
        
        try:
            # Initialize LLM
            llm_kwargs = llm_kwargs or {}
            llm = LLMProvider.get_llm(llm_provider, **llm_kwargs)
            
            # Define metadata fields for self-querying
            metadata_field_info = [
                AttributeInfo(
                    name="page",
                    description="The page number in the document",
                    type="integer"
                )
            ]
            
            # Import the QdrantTranslator directly
            from langchain.retrievers.self_query.qdrant import QdrantTranslator
            
            # Create self-query retriever with explicit translator
            print("Creating self-query retriever...")
            self_query_retriever = SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=vectorstore,
                document_contents="PDF document content",
                metadata_field_info=metadata_field_info,
                structured_query_translator=QdrantTranslator(metadata_key="metadata"),
                search_kwargs={"k": 3}
            )
            
            # Execute query to get documents
            print("Retrieving relevant documents with self-query...")
            retrieved_docs = self_query_retriever.invoke(query)
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            # Format the context
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # Create RAG prompt template
            rag_prompt = f"""Answer the question based only on the following context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            # Execute query directly with the formatted prompt
            print("Generating answer...")
            answer = llm.invoke(rag_prompt)
            if hasattr(answer, 'content'):  # For ChatModels that return messages
                answer = answer.content
            print("Answer generated successfully")
            
            execution_time = time.time() - start_time
            
            # Prepare chunks info
            chunks = []
            for i, doc in enumerate(retrieved_docs):
                chunks.append({
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "score": 0.95 - (i * 0.05)  # Mock score
                })
            
            return {
                "answer": answer,
                "chunks": chunks,
                "time": execution_time,
                "llm_provider": llm_provider
            }
        except Exception as e:
            print_error(e, f"Error in self-query RAG with {llm_provider}")
            raise
    
    def reranker_rag(self, document_id: str, query: str, llm_provider: str = "groq", llm_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reranker RAG implementation"""
        print(f"Reranker RAG for document {document_id}: {query} using {llm_provider}")
        start_time = time.time()
        
        # Get vector store
        if document_id not in self.document_stores:
            error_msg = f"Document ID {document_id} not found"
            print(error_msg)
            raise ValueError(error_msg)
        
        vectorstore = self.document_stores[document_id]["vectorstore"]
        
        try:
            # Initialize LLM
            llm_kwargs = llm_kwargs or {}
            llm = LLMProvider.get_llm(llm_provider, **llm_kwargs)
            
            # Create retriever with larger k for reranking
            print("Creating retriever for reranking...")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            
            # Retrieve documents
            print("Retrieving documents for reranking...")
            retrieved_docs = retriever.invoke(query)
            print(f"Retrieved {len(retrieved_docs)} documents for reranking")
            
            # Mock reranking (in real implementation, use a reranker model)
            print("Performing mock reranking...")
            reranked_docs = retrieved_docs[:3]
            
            # Create formatted context from reranked documents
            context = "\n\n".join(doc.page_content for doc in reranked_docs)
            
            # Create RAG prompt template
            rag_prompt = f"""Answer the question based only on the following context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            # Execute query directly with the formatted prompt
            print("Generating answer...")
            answer = llm.invoke(rag_prompt)
            if hasattr(answer, 'content'):  # For ChatModels that return messages
                answer = answer.content
            print("Answer generated successfully")
            
            execution_time = time.time() - start_time
            
            # Prepare chunks info
            chunks = []
            for i, doc in enumerate(reranked_docs):
                chunks.append({
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "score": 0.98 - (i * 0.03)  # Mock score for reranked docs
                })
            
            return {
                "answer": answer,
                "chunks": chunks,
                "time": execution_time,
                "llm_provider": llm_provider
            }
        except Exception as e:
            print_error(e, f"Error in reranker RAG with {llm_provider}")
            raise