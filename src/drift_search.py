import os
import pandas as pd
import tiktoken
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_report_embeddings,
)
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.language_model.manager import ModelManager
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.structured_search.drift_search.drift_context import DRIFTSearchContextBuilder
from graphrag.query.structured_search.drift_search.search import DRIFTSearch

# Constants for drift search configuration
INPUT_DIR = "graphrag_workspace/output"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"

# Table names for GraphRAG data files
COMMUNITY_REPORT_TABLE = "community_reports"
COMMUNITY_TABLE = "communities"
ENTITY_TABLE = "entities"
RELATIONSHIP_TABLE = "relationships"
TEXT_UNIT_TABLE = "text_units"
COMMUNITY_LEVEL = 0  # Default community hierarchy level


class GraphRAGDriftSearch:
    """
    Drift Search (RIFT Search) implementation for GraphRAG.
    
    This class implements iterative refinement search over a knowledge graph,
    combining graph-based reasoning with vector similarity search to find
    relevant information through multiple refinement steps.
    
    The search process:
    1. Initial query understanding
    2. Entity and relationship mapping
    3. Iterative context refinement
    4. Response generation with supporting evidence
    """
    
    def __init__(self, 
                 input_dir: str = INPUT_DIR,
                 community_level: int = COMMUNITY_LEVEL,
                 api_key: Optional[str] = None,
                 chat_model: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        """
        Initialize Drift Search with GraphRAG data.
        
        Args:
            input_dir: Directory containing GraphRAG output files
            community_level: Level of community hierarchy to use
            api_key: OpenAI API key (defaults to environment variable)
            chat_model: LLM model name (defaults to environment variable)
            embedding_model: Embedding model name (defaults to environment variable)
        """
        self.input_dir = input_dir
        self.community_level = community_level
        
        # Get API credentials with fallbacks
        self.api_key = api_key or os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.chat_model = chat_model or os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-4-turbo-preview")
        self.embedding_model = embedding_model or os.environ.get("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set GRAPHRAG_API_KEY or OPENAI_API_KEY environment variable.")
        
        # Initialize components (will be set up in respective methods)
        self.search_engine = None
        self.chat_model_instance = None
        self.text_embedder = None
        self.token_encoder = None
        
        # Load data and setup components
        self._load_data()
        self._setup_models()
        self._setup_search_engine()
    
    def _load_data(self):
        """
        Load and prepare GraphRAG data files.
        
        This method:
        1. Loads parquet files containing entities, relationships, etc.
        2. Verifies file existence and data integrity
        3. Prepares data structures for search
        """
        print(f"Loading data from {self.input_dir}...")
        
        # Define file paths
        entity_file = f"{self.input_dir}/{ENTITY_TABLE}.parquet"
        community_file = f"{self.input_dir}/{COMMUNITY_TABLE}.parquet"
        relationship_file = f"{self.input_dir}/{RELATIONSHIP_TABLE}.parquet"
        text_unit_file = f"{self.input_dir}/{TEXT_UNIT_TABLE}.parquet"
        report_file = f"{self.input_dir}/{COMMUNITY_REPORT_TABLE}.parquet"
        
        # Verify file existence
        required_files = [entity_file, community_file, relationship_file, text_unit_file, report_file]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load data frames
        self.entity_df = pd.read_parquet(entity_file)
        self.community_df = pd.read_parquet(community_file)
        self.relationship_df = pd.read_parquet(relationship_file)
        self.text_unit_df = pd.read_parquet(text_unit_file)
        self.report_df = pd.read_parquet(report_file)
        
        print(f"Loaded data summary:")
        print(f"- Entities: {len(self.entity_df)}")
        print(f"- Relationships: {len(self.relationship_df)}")
        print(f"- Text units: {len(self.text_unit_df)}")
        
        # Prepare context data structures
        self.entities = read_indexer_entities(self.entity_df, self.community_df, self.community_level)
        self.relationships = read_indexer_relationships(self.relationship_df)
        self.text_units = read_indexer_text_units(self.text_unit_df)
        
        # Load reports with embeddings
        self.reports = read_indexer_reports(
            self.report_df,
            self.community_df,
            self.community_level,
            content_embedding_col="full_content_embeddings",
        )
        
        print("Data preparation complete!")
    
    def _setup_models(self):
        """
        Setup LLM and embedding models.
        
        Configures:
        1. Chat model for query understanding and response generation
        2. Embedding model for semantic search
        3. Token encoder for text processing
        """
        print("Setting up models...")
        
        # Configure chat model
        chat_config = LanguageModelConfig(
            api_key=self.api_key,
            type=ModelType.OpenAIChat,
            model=self.chat_model,
            max_retries=20,
            model_supports_json=False,  # Disable JSON mode for models that don't support it
        )
        
        # Initialize chat model
        self.chat_model_instance = ModelManager().get_or_create_chat_model(
            name="drift_search",
            model_type=ModelType.OpenAIChat,
            config=chat_config,
        )
        
        # Configure embedding model
        embedding_config = LanguageModelConfig(
            api_key=self.api_key,
            type=ModelType.OpenAIEmbedding,
            model=self.embedding_model,
            max_retries=20,
        )
        
        # Initialize embedding model
        self.text_embedder = ModelManager().get_or_create_embedding_model(
            name="drift_search_embedding",
            model_type=ModelType.OpenAIEmbedding,
            config=embedding_config,
        )
        
        # Setup token encoder
        self.token_encoder = tiktoken.encoding_for_model(self.chat_model)
        
        print(f"Models setup complete: {self.chat_model} (chat), {self.embedding_model} (embedding)")
    
    def _setup_search_engine(self):
        """
        Setup the Drift Search engine.
        
        Configures:
        1. Vector stores for embeddings
        2. Context builder with search parameters
        3. Search engine with all components
        """
        print("Setting up Drift Search engine...")
        
        # Set up embedding stores
        lancedb_uri = f"{self.input_dir}/lancedb"
        
        # Entity description embeddings
        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
        description_embedding_store.connect(db_uri=lancedb_uri)
        
        # Community report embeddings
        full_content_embedding_store = LanceDBVectorStore(collection_name="default-community-full_content")
        full_content_embedding_store.connect(db_uri=lancedb_uri)
        
        # Load report embeddings
        read_indexer_report_embeddings(self.reports, full_content_embedding_store)
        
        # Configure drift search parameters
        drift_params = DRIFTSearchConfig(
            temperature=0,
            max_tokens=12_000,
            primer_folds=1,
            drift_k_followups=3,
            n_depth=3,
            n=1,
        )
        
        # Setup context builder
        context_builder = DRIFTSearchContextBuilder(
            model=self.chat_model_instance,
            text_embedder=self.text_embedder,
            entities=self.entities,
            relationships=self.relationships,
            reports=self.reports,
            entity_text_embeddings=description_embedding_store,
            text_units=self.text_units,
            token_encoder=self.token_encoder,
            config=drift_params,
        )
        
        # Initialize search engine
        self.search_engine = DRIFTSearch(
            model=self.chat_model_instance,
            context_builder=context_builder,
            token_encoder=self.token_encoder,
        )
        
        print("Drift Search engine setup complete!")
    
    async def search(self, query: str) -> Dict[str, Any]:
        """
        Perform drift search over the knowledge graph.
        
        This method:
        1. Validates the search engine state
        2. Executes the search with error handling
        3. Returns results with metadata
        
        Args:
            query: User query for iterative refinement search
            
        Returns:
            Dictionary containing:
            - query: Original query
            - response: Search result
            - success: Whether search succeeded
            - metadata: Search execution details
            - error: Error message if failed
        """
        if not self.search_engine:
            raise RuntimeError("Search engine not initialized. Call _setup_search_engine() first.")
        
        print(f"Performing drift search for: '{query}'")
        
        try:
            result = await self.search_engine.search(query)
            
            return {
                'query': query,
                'response': result.response,
                'success': True,
                'metadata': {
                    'chat_model_used': self.chat_model,
                    'embedding_model_used': self.embedding_model,
                    'community_level': self.community_level,
                    'num_entities': len(self.entities),
                    'num_relationships': len(self.relationships),
                    'num_text_units': len(self.text_units),
                    'search_type': 'drift_search'
                }
            }
            
        except Exception as e:
            print(f"Error during drift search: {e}")
            print(f"Model being used: {self.chat_model}")
            print(f"Error type: {type(e).__name__}")
            return {
                'query': query,
                'response': f"Error: {str(e)}",
                'success': False,
                'error': str(e)
            }
    
    def search_sync(self, query: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for drift search.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing search results
        """
        return asyncio.run(self.search(query))


def main():
    """Example usage of Drift Search with test queries."""
    
    # Initialize Drift Search
    try:
        drift_search = GraphRAGDriftSearch()
        
        # Test queries for iterative refinement search
        test_queries = [
            "What is the relationship between water and pressure in fluid mechanics?",
            "How do vectors and scalar products work together in physics?",
            "Tell me about Blaise Pascal and his contributions to fluid mechanics",
            "What is the difference between gauge pressure and absolute pressure?",
            "How do materials like brass and wood behave differently in water?",
            "What is Pascal's principle and how does it apply to hydraulic systems?",
            "Which scientist made a super important contribution to fluid mechanics?",
            "What is the relationship between density and pressure in fluids?",
            "How do hydraulic brakes work?",
            "What is the significance of atmospheric pressure in various systems?"
        ]
        
        print("\n=== Testing Drift Search ===\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 50)
            
            result = drift_search.search_sync(query)
            
            if result['success']:
                print(f"Response: {result['response']}")
                print(f"Metadata: {result['metadata']}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            print("\n" + "="*60 + "\n")
    
    except Exception as e:
        print(f"Error initializing Drift Search: {e}")


if __name__ == "__main__":
    main()
