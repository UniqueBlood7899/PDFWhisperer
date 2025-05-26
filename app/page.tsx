"use client";
import { useState, useEffect } from 'react'
import { useToast } from '@/hooks/use-toast'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'


// Use environment variables with a fallback
const getApiUrl = () => {
  const url = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  console.log('API URL:', url);
  return url;
};

const API_URL = getApiUrl();

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
  llm_provider: string;
}

// Types for LLM providers
type LLMModel = {
  id: string;
  name: string;
}

type LLMProvider = {
  name: string;
  models: LLMModel[];
}

type LLMProviders = {
  [key: string]: LLMProvider;
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
    self_query: RAGResponse | null;
    reranker: RAGResponse | null;
  }>({
    basic: null,
    self_query: null,
    reranker: null
  })
  const [llmProviders, setLlmProviders] = useState<LLMProviders>({})
  const [selectedProvider, setSelectedProvider] = useState<string>('groq')
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [temperature, setTemperature] = useState<number>(0.1)
  
  const { toast } = useToast()

  // Fetch available LLM providers when component mounts
  useEffect(() => {
    const fetchLLMProviders = async () => {
      try {
        console.log('Fetching providers from:', `${API_URL}/llm-providers`);
        const response = await fetch(`${API_URL}/llm-providers`);
        if (!response.ok) {
          throw new Error(`Failed to fetch providers: ${response.status}`);
        }
        const data = await response.json();
        setLlmProviders(data);
        // Set default model for initial provider
        if (data.groq && data.groq.models.length > 0) {
          setSelectedModel(data.groq.models[0].id);
        }
      } catch (error) {
        console.error('Error fetching LLM providers:', error);
      }
    };
    
    fetchLLMProviders();
  }, []);

  // Handle provider change
  const handleProviderChange = (provider: string) => {
    setSelectedProvider(provider);
    // Set first model of selected provider as default
    if (llmProviders[provider] && llmProviders[provider].models.length > 0) {
      setSelectedModel(llmProviders[provider].models[0].id);
    }
  };

  // File upload handler
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPdfFile(e.target.files[0])
    }
  }

  // Upload file to server
  const uploadFile = async () => {
    if (!pdfFile) return

    setIsUploading(true)
    const formData = new FormData()
    formData.append('file', pdfFile)

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }

      const data = await response.json()
      setDocumentId(data.document_id)
      setDocumentName(pdfFile.name)
      toast({
        title: 'Upload successful',
        description: 'Your PDF has been processed and is ready for queries.',
      })
    } catch (error) {
      console.error('Upload error:', error)
      toast({
        title: 'Upload error',
        description: error instanceof Error ? error.message : 'Failed to upload PDF',
        variant: 'destructive',
      })
    } finally {
      setIsUploading(false)
    }
  }

  // Submit query to server
  const submitQuery = async () => {
    if (!documentId || !prompt) return

    setIsQuerying(true)
    setResponses({
      basic: null,
      self_query: null,
      reranker: null
    })

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          document_id: documentId,
          query: prompt,
          rag_types: ['basic', 'self_query', 'reranker'],
          llm_provider: selectedProvider,
          llm_model: selectedModel,
          llm_temperature: temperature
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Query failed')
      }

      const data = await response.json()
      setResponses(data)
    } catch (error) {
      console.error('Query error:', error)
      toast({
        title: 'Query error',
        description: error instanceof Error ? error.message : 'Failed to process query',
        variant: 'destructive',
      })
    } finally {
      setIsQuerying(false)
    }
  }

  // Render functions
  const renderUploadSection = () => (
    <Card>
      <CardHeader>
        <CardTitle>Upload a PDF</CardTitle>
        <CardDescription>Upload a PDF document to query using different RAG architectures</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid w-full max-w-sm items-center gap-1.5">
          <Label htmlFor="pdf">PDF Document</Label>
          <Input id="pdf" type="file" accept=".pdf" onChange={handleFileChange} />
        </div>
      </CardContent>
      <CardFooter>
        <Button onClick={uploadFile} disabled={!pdfFile || isUploading}>
          {isUploading ? 'Uploading...' : 'Upload and Process'}
        </Button>
      </CardFooter>
    </Card>
  )

  const renderQuerySection = () => (
    <Card>
      <CardHeader>
        <CardTitle>Query Document: {documentName}</CardTitle>
        <CardDescription>Ask questions about the document using different RAG architectures</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <Label htmlFor="prompt">Your Question</Label>
            <Input 
              id="prompt" 
              value={prompt} 
              onChange={(e) => setPrompt(e.target.value)} 
              placeholder="What does this document say about..." 
            />
          </div>
          
          <div className="space-y-2">
            <Label>LLM Provider</Label>
            <Select value={selectedProvider} onValueChange={handleProviderChange}>
              <SelectTrigger>
                <SelectValue placeholder="Select LLM provider" />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(llmProviders).map(([key, provider]) => (
                  <SelectItem key={key} value={key}>{provider.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <Label>Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger>
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                {selectedProvider && llmProviders[selectedProvider]?.models.map(model => (
                  <SelectItem key={model.id} value={model.id}>{model.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Temperature: {temperature.toFixed(2)}</Label>
            </div>
            <Slider 
              value={[temperature]} 
              min={0} 
              max={1} 
              step={0.01} 
              onValueChange={(value) => setTemperature(value[0])} 
            />
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button onClick={submitQuery} disabled={!documentId || !prompt || isQuerying}>
          {isQuerying ? 'Processing...' : 'Submit Query'}
        </Button>
      </CardFooter>
    </Card>
  )

  const renderResults = () => (
    <Tabs defaultValue="basic">
      <TabsList className="grid w-full grid-cols-3">
        <TabsTrigger value="basic">Basic RAG</TabsTrigger>
        <TabsTrigger value="self_query">Self-Query RAG</TabsTrigger>
        <TabsTrigger value="reranker">Reranker RAG</TabsTrigger>
      </TabsList>
      
      <TabsContent value="basic">
        {renderRAGResult(responses.basic)}
      </TabsContent>
      
      <TabsContent value="self_query">
        {renderRAGResult(responses.self_query)}
      </TabsContent>
      
      <TabsContent value="reranker">
        {renderRAGResult(responses.reranker)}
      </TabsContent>
    </Tabs>
  )

  const renderRAGResult = (result: RAGResponse | null) => {
    if (!result) return <div className="py-4 text-center text-muted-foreground">No results yet</div>
    
    return (
      <Card>
        <CardHeader>
          <CardTitle>Answer</CardTitle>
          <CardDescription>
            Generated using {result.llm_provider} in {result.time.toFixed(2)} seconds
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-md bg-muted p-4">
            <p className="whitespace-pre-wrap">{result.answer}</p>
          </div>
          
          <div>
            <h4 className="mb-2 font-medium">Relevant Chunks:</h4>
            <ScrollArea className="h-[200px] rounded-md border p-4">
              {result.chunks.map((chunk, i) => (
                <div key={i} className="mb-4 rounded-md bg-muted p-2">
                  <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                    <span>Page {chunk.page}</span>
                    <span>Score: {chunk.score.toFixed(2)}</span>
                  </div>
                  <p className="text-sm">{chunk.text}</p>
                </div>
              ))}
            </ScrollArea>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <main className="container py-10">
      <h1 className="mb-8 text-3xl font-bold">RAG Architectures Comparison</h1>
      
      <div className="grid gap-8 md:grid-cols-2">
        <div className="space-y-8">
          {!documentId ? renderUploadSection() : renderQuerySection()}
        </div>
        
        <div>
          {(responses.basic || responses.self_query || responses.reranker) && renderResults()}
        </div>
      </div>
    </main>
  )
}
