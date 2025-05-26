'use client'

import { useState } from 'react'
import { useToast } from '@/hooks/use-toast'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'

// API URL - change this to your FastAPI backend URL
const API_URL = 'http://localhost:8000';

// Types for API responses
type ChunkInfo = {
  text: string;
  page: number;
  score: number;
}

type RAGResponse = {
  answer: string;
  chunks: ChunkInfo[];
  time: number;
}

export default function Home() {
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [documentId, setDocumentId] = useState<string | null>(null)
  const [documentName, setDocumentName] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [isQuerying, setIsQuerying] = useState(false)
  const [responses, setResponses] = useState<{
    basic: RAGResponse | null;
    selfQuery: RAGResponse | null;
    reranker: RAGResponse | null;
  }>({
    basic: null,
    selfQuery: null,
    reranker: null
  })
  const { toast } = useToast()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPdfFile(e.target.files[0])
      setDocumentId(null) // Reset document ID when a new file is selected
    }
  }

  const handleUpload = async () => {
    if (!pdfFile) {
      toast({
        title: "Error",
        description: "Please select a PDF file first",
        variant: "destructive"
      })
      return
    }

    setIsUploading(true)
    
    try {
      // Create form data for file upload
      const formData = new FormData()
      formData.append('file', pdfFile)
      
      // Send file to backend
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to upload PDF')
      }
      
      const data = await response.json()
      
      // Store document ID for querying
      setDocumentId(data.document_id)
      setDocumentName(pdfFile.name)
      
      toast({
        title: "Success",
        description: "PDF uploaded and processed successfully",
      })
    } catch (error) {
      console.error('Upload error:', error)
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to upload PDF",
        variant: "destructive"
      })
    } finally {
      setIsUploading(false)
    }
  }

  const handleQuery = async () => {
    if (!prompt.trim()) {
      toast({
        title: "Error",
        description: "Please enter a prompt",
        variant: "destructive"
      })
      return
    }

    if (!documentId) {
      toast({
        title: "Error",
        description: "No PDF has been uploaded yet or processing failed",
        variant: "destructive"
      })
      return
    }

    setIsQuerying(true)
    
    try {
      // Prepare query request
      const queryRequest = {
        document_id: documentId,
        query: prompt,
        rag_types: ["basic", "self_query", "reranker"]
      }
      
      // Send query to backend
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(queryRequest)
      })
      
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to process query')
      }
      
      const data = await response.json()
      
      // Update responses state with API results
      setResponses({
        basic: data.basic,
        selfQuery: data.self_query,
        reranker: data.reranker
      })
    } catch (error) {
      console.error('Query error:', error)
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to process query",
        variant: "destructive"
      })
    } finally {
      setIsQuerying(false)
    }
  }

  return (
    <main className="container mx-auto py-10 space-y-8">
      <div className="flex flex-col items-center">
        <h1 className="text-4xl font-bold mb-4">RAG Architecture Comparison</h1>
        <p className="text-muted-foreground text-lg max-w-2xl text-center mb-8">
          Upload PDFs and compare outputs from different RAG architectures
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Card>
          <CardHeader>
            <CardTitle>Upload Document</CardTitle>
            <CardDescription>
              Upload a PDF document to be processed by the RAG architectures
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid w-full items-center gap-4">
              <div className="flex flex-col space-y-1.5">
                <Label htmlFor="pdf">PDF Document</Label>
                <Input id="pdf" type="file" accept=".pdf" onChange={handleFileChange} />
              </div>
              {documentId && documentName && (
                <div className="p-3 bg-green-50 text-green-700 rounded-md">
                  <p className="text-sm font-medium">Current document: {documentName}</p>
                </div>
              )}
            </div>
          </CardContent>
          <CardFooter>
            <Button onClick={handleUpload} disabled={isUploading}>
              {isUploading ? "Uploading..." : "Upload and Process"}
            </Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Ask a Question</CardTitle>
            <CardDescription>
              Enter your query to be processed by the RAG architectures
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid w-full items-center gap-4">
              <div className="flex flex-col space-y-1.5">
                <Label htmlFor="prompt">Your Question</Label>
                <Input 
                  id="prompt" 
                  placeholder="Enter your question here..." 
                  value={prompt} 
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button onClick={handleQuery} disabled={isQuerying || !documentId}>
              {isQuerying ? "Processing..." : "Submit Query"}
            </Button>
          </CardFooter>
        </Card>
      </div>

      <Tabs defaultValue="comparison" className="w-full">
        <TabsList className="grid w-full md:w-[400px] grid-cols-3">
          <TabsTrigger value="comparison">Comparison</TabsTrigger>
          <TabsTrigger value="details">Details</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>
        
        <TabsContent value="comparison" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Basic RAG</CardTitle>
                <CardDescription>Simple retrieval + generation</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[300px] w-full rounded-md border p-4">
                  {responses.basic ? responses.basic.answer : "No response yet"}
                </ScrollArea>
              </CardContent>
              <CardFooter>
                {responses.basic && <p>Response time: {responses.basic.time.toFixed(2)}s</p>}
              </CardFooter>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Self-Query RAG</CardTitle>
                <CardDescription>Query decomposition & refinement</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[300px] w-full rounded-md border p-4">
                  {responses.selfQuery ? responses.selfQuery.answer : "No response yet"}
                </ScrollArea>
              </CardContent>
              <CardFooter>
                {responses.selfQuery && <p>Response time: {responses.selfQuery.time.toFixed(2)}s</p>}
              </CardFooter>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Reranker RAG</CardTitle>
                <CardDescription>With semantic reranking</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[300px] w-full rounded-md border p-4">
                  {responses.reranker ? responses.reranker.answer : "No response yet"}
                </ScrollArea>
              </CardContent>
              <CardFooter>
                {responses.reranker && <p>Response time: {responses.reranker.time.toFixed(2)}s</p>}
              </CardFooter>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="details" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Retrieved Context</CardTitle>
              <CardDescription>Source chunks used for generating responses</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {responses.basic && (
                  <div>
                    <h3 className="font-medium mb-2">Basic RAG Context:</h3>
                    <div className="pl-4 border-l-2 border-primary">
                      {responses.basic.chunks.map((chunk, i) => (
                        <div key={i} className="text-sm mb-4 p-2 bg-gray-50 rounded">
                          <p className="mb-1">{chunk.text}</p>
                          <div className="text-xs text-gray-500 flex justify-between">
                            <span>Page: {chunk.page}</span>
                            <span>Score: {chunk.score.toFixed(2)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {responses.selfQuery && (
                  <div>
                    <h3 className="font-medium mb-2">Self-Query RAG Context:</h3>
                    <div className="pl-4 border-l-2 border-primary">
                      {responses.selfQuery.chunks.map((chunk, i) => (
                        <div key={i} className="text-sm mb-4 p-2 bg-gray-50 rounded">
                          <p className="mb-1">{chunk.text}</p>
                          <div className="text-xs text-gray-500 flex justify-between">
                            <span>Page: {chunk.page}</span>
                            <span>Score: {chunk.score.toFixed(2)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {responses.reranker && (
                  <div>
                    <h3 className="font-medium mb-2">Reranker RAG Context:</h3>
                    <div className="pl-4 border-l-2 border-primary">
                      {responses.reranker.chunks.map((chunk, i) => (
                        <div key={i} className="text-sm mb-4 p-2 bg-gray-50 rounded">
                          <p className="mb-1">{chunk.text}</p>
                          <div className="text-xs text-gray-500 flex justify-between">
                            <span>Page: {chunk.page}</span>
                            <span>Score: {chunk.score.toFixed(2)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {!responses.basic && !responses.selfQuery && !responses.reranker && (
                  <p>No responses available yet. Submit a query first.</p>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="metrics" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
              <CardDescription>Compare response times and other metrics</CardDescription>
            </CardHeader>
            <CardContent>
              {(responses.basic || responses.selfQuery || responses.reranker) ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="flex flex-col items-center p-4 border rounded-md">
                      <span className="text-lg font-medium">Basic RAG</span>
                      <span className="text-3xl font-bold mt-2">{responses.basic?.time.toFixed(2) || "-"}s</span>
                    </div>
                    <div className="flex flex-col items-center p-4 border rounded-md">
                      <span className="text-lg font-medium">Self-Query RAG</span>
                      <span className="text-3xl font-bold mt-2">{responses.selfQuery?.time.toFixed(2) || "-"}s</span>
                    </div>
                    <div className="flex flex-col items-center p-4 border rounded-md">
                      <span className="text-lg font-medium">Reranker RAG</span>
                      <span className="text-3xl font-bold mt-2">{responses.reranker?.time.toFixed(2) || "-"}s</span>
                    </div>
                  </div>
                  
                  <div className="mt-6">
                    <h3 className="font-medium mb-2">Chunks Retrieved:</h3>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="p-4 border rounded-md">
                        <span className="font-medium">Basic RAG:</span> {responses.basic?.chunks.length || 0} chunks
                      </div>
                      <div className="p-4 border rounded-md">
                        <span className="font-medium">Self-Query RAG:</span> {responses.selfQuery?.chunks.length || 0} chunks
                      </div>
                      <div className="p-4 border rounded-md">
                        <span className="font-medium">Reranker RAG:</span> {responses.reranker?.chunks.length || 0} chunks
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-6">
                    <h3 className="font-medium mb-2">Average Relevance Score:</h3>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="p-4 border rounded-md">
                        <span className="font-medium">Basic RAG:</span> {responses.basic 
                          ? (responses.basic.chunks.reduce((sum, chunk) => sum + chunk.score, 0) / responses.basic.chunks.length).toFixed(2) 
                          : "-"}
                      </div>
                      <div className="p-4 border rounded-md">
                        <span className="font-medium">Self-Query RAG:</span> {responses.selfQuery 
                          ? (responses.selfQuery.chunks.reduce((sum, chunk) => sum + chunk.score, 0) / responses.selfQuery.chunks.length).toFixed(2) 
                          : "-"}
                      </div>
                      <div className="p-4 border rounded-md">
                        <span className="font-medium">Reranker RAG:</span> {responses.reranker 
                          ? (responses.reranker.chunks.reduce((sum, chunk) => sum + chunk.score, 0) / responses.reranker.chunks.length).toFixed(2) 
                          : "-"}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <p>No metrics available yet. Submit a query first.</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </main>
  )
}
