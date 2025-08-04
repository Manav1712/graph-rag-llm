"""
Global Search implementation for GraphRAG.

This module provides holistic search capabilities over the entire corpus,
focusing on broad themes, patterns, and high-level understanding through
community-level analysis and cross-document relationships.
"""

import os
import pandas as pd
import tiktoken
import asyncio
from typing import Optional, Dict, Any
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch


class GraphRAGGlobalSearch:
    """
    Global Search implementation for GraphRAG.
    
    This class focuses on answering questions that require understanding
    the entire corpus at a high level. It analyzes community structures,
    cross-document patterns, and thematic relationships to provide
    comprehensive answers.
    
    Key capabilities:
    - Thematic analysis across documents
    - Community-level insights
    - Cross-document relationship discovery
    - High-level pattern recognition
    """
    
    def __init__(self, 
                 input_dir: str = "graphrag_workspace/output",
                 community_level: int = 0,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize Global Search with GraphRAG data.
        
        Args:
            input_dir: Directory containing GraphRAG output files
            community_level: Level of community hierarchy to use
            api_key: OpenAI API key (defaults to environment variable)
            model_name: LLM model name (defaults to environment variable)
        """
        self.input_dir = input_dir
        self.community_level = community_level
        
        # Get API credentials with fallbacks
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name or os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-4")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize components (will be set up in respective methods)
        self.search_engine = None
        self.model = None
        self.token_encoder = None
        
        # Load data and setup components
        self._load_data()
        self._setup_model()
        self._setup_search_engine()
    
    def _load_data(self):
        """
        Load and prepare GraphRAG data files.
        
        This method:
        1. Loads parquet files containing communities, entities, and reports
        2. Verifies file existence and data integrity
        3. Prepares data structures for global search
        """
        print(f"Loading data from {self.input_dir}...")
        
        # Define file paths
        community_file = f"{self.input_dir}/communities.parquet"
        entity_file = f"{self.input_dir}/entities.parquet"
        report_file = f"{self.input_dir}/community_reports.parquet"
        
        # Verify file existence
        required_files = [community_file, entity_file, report_file]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load data frames
        self.community_df = pd.read_parquet(community_file)
        self.entity_df = pd.read_parquet(entity_file)
        self.report_df = pd.read_parquet(report_file)
        
        print(f"Loaded data summary:")
        print(f"- Communities: {len(self.community_df)}")
        print(f"- Entities: {len(self.entity_df)}")
        print(f"- Reports: {len(self.report_df)}")
        
        # Prepare context data structures
        self.communities = read_indexer_communities(self.community_df, self.report_df)
        self.reports = read_indexer_reports(self.report_df, self.community_df, self.community_level)
        self.entities = read_indexer_entities(self.entity_df, self.community_df, self.community_level)
        
        print("Data preparation complete!")
    
    def _setup_model(self):
        """
        Setup LLM model and token encoder.
        
        Configures:
        1. Chat model for query understanding and response generation
        2. Token encoder for text processing
        """
        print("Setting up LLM model...")
        
        # Configure LLM
        config = LanguageModelConfig(
            api_key=self.api_key,
            type=ModelType.OpenAIChat,
            model=self.model_name,
            max_retries=20,
        )
        
        # Initialize model
        self.model = ModelManager().get_or_create_chat_model(
            name="global_search",
            model_type=ModelType.OpenAIChat,
            config=config,
        )
        
        # Setup token encoder
        self.token_encoder = tiktoken.encoding_for_model(self.model_name)
        
        print(f"LLM model setup complete: {self.model_name}")
    
    def _setup_search_engine(self):
        """
        Setup the Global Search engine.
        
        Configures:
        1. Context builder with community focus
        2. Search parameters for global analysis
        3. Search engine with model configuration
        """
        print("Setting up Global Search engine...")
        
        # Configure context builder parameters
        context_builder_params = {
            # Community analysis settings
            "use_community_summary": False,  # Use full reports
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            
            # Community weight settings
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            
            # Other settings
            "max_tokens": 12_000,
            "context_name": "Reports",
        }
        
        # Configure LLM parameters
        map_llm_params = {
            "max_tokens": 1000,
            "temperature": 0.0,  # Use deterministic responses
            "response_format": {"type": "json_object"},
        }
        
        reduce_llm_params = {
            "max_tokens": 2000,
            "temperature": 0.0,  # Use deterministic responses
        }
        
        # Setup context builder
        context_builder = GlobalCommunityContext(
            communities=self.communities,
            reports=self.reports,
            entities=self.entities,
            token_encoder=self.token_encoder,
        )
        
        # Initialize search engine
        self.search_engine = GlobalSearch(
            model=self.model,
            context_builder=context_builder,
            token_encoder=self.token_encoder,
            max_data_tokens=12_000,
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=False,  # Prevent hallucination
            json_mode=True,
            context_builder_params=context_builder_params,
            concurrent_coroutines=32,
            response_type="multiple paragraphs",
        )
        
        print("Global Search engine setup complete!")
    
    async def search(self, query: str) -> Dict[str, Any]:
        """
        Perform global search over the knowledge graph.
        
        This method:
        1. Validates the search engine state
        2. Executes the search with error handling
        3. Returns results with metadata
        
        Args:
            query: User query requiring holistic understanding
            
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
        
        print(f"Performing global search for: '{query}'")
        
        try:
            result = await self.search_engine.search(query)
            
            return {
                'query': query,
                'response': result.response,
                'success': True,
                'metadata': {
                    'model_used': self.model_name,
                    'community_level': self.community_level,
                    'num_communities': len(self.communities),
                    'num_entities': len(self.entities),
                    'num_reports': len(self.reports)
                }
            }
            
        except Exception as e:
            print(f"Error during global search: {e}")
            return {
                'query': query,
                'response': f"Error: {str(e)}",
                'success': False,
                'error': str(e)
            }
    
    def search_sync(self, query: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for global search.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing search results
        """
        return asyncio.run(self.search(query))


def main():
    """Example usage of Global Search with test queries."""
    
    # Initialize Global Search
    try:
        global_search = GraphRAGGlobalSearch()
        
        # Test queries for thematic understanding
        test_queries = [
            "What are the main themes in physics?",
            "How do different physics concepts relate to each other?",
            "What are the fundamental principles that connect various physics topics?",
            "Summarize the key relationships between energy, matter, and forces."
        ]
        
        print("\n=== Testing Global Search ===\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 50)
            
            result = global_search.search_sync(query)
            
            if result['success']:
                print(f"Response: {result['response']}")
                print(f"Metadata: {result['metadata']}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            print("\n" + "="*60 + "\n")
    
    except Exception as e:
        print(f"Error initializing Global Search: {e}")


if __name__ == "__main__":
    main()