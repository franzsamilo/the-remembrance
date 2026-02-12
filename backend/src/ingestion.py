import os
import sys
import asyncio
import warnings

# Suppress Google API warnings about Python 3.9
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.oauth2")

import neo4j
from typing import List, Optional, Union, Any, Sequence

# Append parent directory to sys.path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.tool import Tool

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase

from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GOOGLE_API_KEY, NEO4J_DATABASE

# --- Wrapper for Embeddings ---
class GeminiEmbedder(Embedder):
    def __init__(self, api_key):
        super().__init__()
        self.internal_embedder = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key
        )

    def embed_query(self, text: str) -> List[float]:
        return self.internal_embedder.embed_query(text)

# --- Wrapper for LLM ---
class GeminiLLM(LLMInterface):
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        super().__init__(model_name=model_name, model_params={})
        self.internal_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0
        )

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        try:
            # Construct a prompt, incorporating history if needed
            # For simplicity in this pipeline, we pass 'input' directly.
            # LangChain's invoke returns an AIMessage.
            result = self.internal_llm.invoke(input)
            return LLMResponse(content=str(result.content))
        except Exception as e:
             raise LLMGenerationError(e)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        try:
            result = await self.internal_llm.ainvoke(input)
            return LLMResponse(content=str(result.content))
        except Exception as e:
            raise LLMGenerationError(e)

    # Implement abstract methods to satisfy Interface, even if unused by SimpleKGPipeline
    def invoke_with_tools(self, input: str, tools: Sequence[Tool], message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None, system_instruction: Optional[str] = None):
         raise NotImplementedError("Tool calling not implemented for this wrapper.")

    async def ainvoke_with_tools(self, input: str, tools: Sequence[Tool], message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None, system_instruction: Optional[str] = None):
         raise NotImplementedError("Tool calling not implemented for this wrapper.")

async def process_documents():
    # 1. Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    # 2. Setup LLM & Embedder with Wrappers
    # We must wrap the LangChain LLM to match the neo4j-graphrag LLMInterface
    llm = GeminiLLM(api_key=GOOGLE_API_KEY, model_name="gemini-2.5-flash") # Use flash for speed/availability if pro acts up
    
    embedder = GeminiEmbedder(GOOGLE_API_KEY)

    # 3. Define Schema 
    schema_config = {
        "node_types": ["Method", "Researcher", "Dataset", "Concept", "Result", "Metric"],
        "relationship_types": ["USES", "CONTRADICTS", "EXTENDS", "PROPOSES", "EVALUATES", "ACHIEVES"],
    }

    # 4. Initialize Pipeline
    print("Initializing Knowledge Graph Pipeline with Gemini...")
    kg_pipeline = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=schema_config,
        neo4j_database=NEO4J_DATABASE,
        on_error="IGNORE",
        from_pdf=True
    )

    # 5. Load and Process Documents
    documents_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documents")
    print(f"Scanning documents in {documents_folder}...")

    if not os.path.exists(documents_folder):
         print(f"Directory not found: {documents_folder}")
         return

    pdf_files = [f for f in os.listdir(documents_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF documents found in backend/documents. Please place some PDFs there.")
        return

    print(f"Found {len(pdf_files)} documents. Starting processing...")

    for pdf_file in pdf_files:
        file_path = os.path.join(documents_folder, pdf_file)
        print(f"Processing: {pdf_file}...")
        try:
            # Run the pipeline for each file
            result = await kg_pipeline.run_async(file_path=file_path)
            print(f"Successfully processed {pdf_file}")
            # print(f"Result stats: {result}") 
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            import traceback
            traceback.print_exc()

    print("Ingestion complete!")
    driver.close()

if __name__ == "__main__":
    asyncio.run(process_documents())
