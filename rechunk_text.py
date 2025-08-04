import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import os

def rechunk_text(input_file_path, output_file_path, chunk_size=600, chunk_overlap=100):
    """
    Rechunk text file with specified token size and overlap.
    
    Args:
        input_file_path: Path to the input text file
        output_file_path: Path to save the chunked data (Parquet format)
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
    """
    
    # Read the input file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    
    # Create a DataFrame with chunk data
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            'chunk_id': i,
            'text': chunk,
            'chunk_size': len(chunk.split()),
            'source_file': os.path.basename(input_file_path)
        })
    
    # Convert to DataFrame and save as Parquet
    df = pd.DataFrame(chunk_data)
    df.to_parquet(output_file_path, index=False)
    
    print(f"Successfully chunked text into {len(chunks)} chunks")
    print(f"Chunk size: {chunk_size} tokens")
    print(f"Chunk overlap: {chunk_overlap} tokens")
    print(f"Output saved to: {output_file_path}")
    
    return df

if __name__ == "__main__":
    # Define input and output paths
    input_file = "graphrag_workspace/input/input_file.txt"
    output_file = "data/chunks/rechunked_text.parquet"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Rechunk the text
    chunked_df = rechunk_text(input_file, output_file)
    
    # Display some statistics
    print(f"\nChunking Statistics:")
    print(f"Total chunks: {len(chunked_df)}")
    print(f"Average chunk size: {chunked_df['chunk_size'].mean():.1f} words")
    print(f"Min chunk size: {chunked_df['chunk_size'].min()} words")
    print(f"Max chunk size: {chunked_df['chunk_size'].max()} words") 