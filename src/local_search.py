# src/local_search.py
"""
Local Search implementation for GraphRAG.

This module provides entity and relationship-focused search capabilities,
optimized for finding specific facts and connections in the knowledge graph.
"""

import os
import pandas as pd
import tiktoken
import asyncio
from typing import Optional, Dict, Any
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.vector_stores.lancedb import LanceDBVectorStore


class GraphRAGLocalSearch:
    """
    Local Search implementation for GraphRAG.
    
    This class focuses on answering questions about specific entities,
    facts, or relationships by examining local context within the
    knowledge graph. It's optimized for precise, factual queries
    rather than broad thematic questions.
    
    Key capabilities:
    - Entity-centric search
    - Relationship exploration
    - Local context understanding
    - Fact verification
    """
    
    def __init__(self, 
                 input_dir: str = "graphrag_workspace/output",
                 community_level: int = 0,
                 api_key: Optional[str] = None,
                 chat_model: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        """
        Initialize Local Search with GraphRAG data.
        
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
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.chat_model = chat_model or os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-4")
        self.embedding_model = embedding_model or os.environ.get("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
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
        3. Prepares data structures for local search
        """
        print(f"Loading data from {self.input_dir}...")
        
        # Define file paths
        entity_file = f"{self.input_dir}/entities.parquet"
        community_file = f"{self.input_dir}/communities.parquet"
        relationship_file = f"{self.input_dir}/relationships.parquet"
        text_unit_file = f"{self.input_dir}/text_units.parquet"
        report_file = f"{self.input_dir}/community_reports.parquet"
        
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
        self.reports = read_indexer_reports(self.report_df, self.community_df, self.community_level)
        
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
        )
        
        # Initialize chat model
        self.chat_model_instance = ModelManager().get_or_create_chat_model(
            name="local_search",
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
            name="local_search_embedding",
            model_type=ModelType.OpenAIEmbedding,
            config=embedding_config,
        )
        
        # Setup token encoder
        self.token_encoder = tiktoken.encoding_for_model(self.chat_model)
        
        print(f"Models setup complete: {self.chat_model} (chat), {self.embedding_model} (embedding)")
    
    def _setup_search_engine(self):
        """
        Setup the Local Search engine.
        
        Configures:
        1. Vector store for entity embeddings
        2. Context builder with search parameters
        3. Search engine with model configuration
        """
        print("Setting up Local Search engine...")
        
        # Set up embedding store for entity descriptions
        lancedb_uri = f"{self.input_dir}/lancedb"
        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
        description_embedding_store.connect(db_uri=lancedb_uri)
        
        # Configure context builder parameters
        local_context_params = {
            # Content mixing ratios
            "text_unit_prop": 0.5,      # Weight for text unit context
            "community_prop": 0.1,       # Weight for community context
            
            # Conversation history settings
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            
            # Entity and relationship settings
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            
            # Other settings
            "return_candidate_context": False,
            "embedding_vectorstore_key": "id",
            "max_tokens": 12000,
        }
        
        # Configure model parameters
        model_params = {
            "max_tokens": 2000,
            "temperature": 0.0,  # Use deterministic responses
        }
        
        # Setup context builder
        context_builder = LocalSearchMixedContext(
            community_reports=self.reports,
            text_units=self.text_units,
            entities=self.entities,
            relationships=self.relationships,
            covariates=None,  # Only if you use covariates
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key="id",
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )
        
        # Initialize search engine
        self.search_engine = LocalSearch(
            model=self.chat_model_instance,
            context_builder=context_builder,
            token_encoder=self.token_encoder,
            model_params=model_params,
            context_builder_params=local_context_params,
            response_type="multiple paragraphs",
        )
        
        print("Local Search engine setup complete!")
    
    async def search(self, query: str) -> Dict[str, Any]:
        """
        Perform local search over the knowledge graph.
        
        This method:
        1. Validates the search engine state
        2. Executes the search with error handling
        3. Returns results with metadata
        
        Args:
            query: User query about specific entities or relationships
            
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
        
        print(f"Performing local search for: '{query}'")
        
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
                    'num_text_units': len(self.text_units)
                }
            }
            
        except Exception as e:
            print(f"Error during local search: {e}")
            return {
                'query': query,
                'response': f"Error: {str(e)}",
                'success': False,
                'error': str(e)
            }
    
    def search_sync(self, query: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for local search.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing search results
        """
        return asyncio.run(self.search(query))


def main():
    """Example usage of Local Search with test queries."""
    
    # Initialize Local Search
    try:
        local_search = GraphRAGLocalSearch()
        
        # Test queries based on extracted entities and relationships
        test_queries = [
            "What is the relationship between water and pressure?",
            "How do vectors and scalar products work together?",
            "Tell me about Blaise Pascal and his contributions to fluid mechanics",
            "What is the difference between gauge pressure and absolute pressure?",
            "How do materials like brass and wood behave differently in water?",
            "What is Pascal's principle and how does it apply to hydraulic systems?",
            "How do pressure gauges measure fluid pressure?",
            "What is the relationship between density and pressure in fluids?",
            "How do hydraulic brakes work?",
            "What is the significance of atmospheric pressure in various systems?"
        ]
        
        print("\n=== Testing Local Search ===\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 50)
            
            result = local_search.search_sync(query)
            
            if result['success']:
                print(f"Response: {result['response']}")
                print(f"Metadata: {result['metadata']}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            print("\n" + "="*60 + "\n")
    
    except Exception as e:
        print(f"Error initializing Local Search: {e}")


if __name__ == "__main__":
    main()