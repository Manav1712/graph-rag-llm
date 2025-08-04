import pandas as pd
import json
import os

def extract_entities_to_json():
    """
    Read the entities parquet file and extract entity ID and description to JSON
    """
    # Path to the entities parquet file
    entities_file = "graphrag_workspace/output/entities.parquet"
    
    # Check if file exists
    if not os.path.exists(entities_file):
        print(f"Error: {entities_file} not found")
        return
    
    try:
        # Read the parquet file
        df = pd.read_parquet(entities_file)
        
        print(f"Successfully read entities parquet file")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Extract entity ID and description
        # Assuming the columns are named 'id' and 'description' or similar
        # Let's check what columns we have and find the right ones
        print(f"\nColumn names: {list(df.columns)}")
        
        # Look for ID and description columns
        id_col = None
        desc_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'id' in col_lower or 'entity' in col_lower:
                id_col = col
            elif 'desc' in col_lower or 'text' in col_lower or 'content' in col_lower:
                desc_col = col
        
        if id_col is None:
            # Try to find the first column that looks like an ID
            for col in df.columns:
                if df[col].dtype in ['int64', 'object'] and 'id' in col.lower():
                    id_col = col
                    break
        
        if desc_col is None:
            # Try to find a text column
            for col in df.columns:
                if df[col].dtype == 'object' and col != id_col:
                    desc_col = col
                    break
        
        print(f"Using ID column: {id_col}")
        print(f"Using description column: {desc_col}")
        
        if id_col is None or desc_col is None:
            print("Could not identify ID and description columns")
            return
        
        # Look for title/name column
        title_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'title' in col_lower or 'name' in col_lower:
                title_col = col
                break
        
        print(f"Using title column: {title_col}")
        
        # Extract the data
        entities_data = []
        for _, row in df.iterrows():
            entity_info = {
                "entity_id": str(row[id_col]),
                "entity_name": str(row[title_col]) if title_col and pd.notna(row[title_col]) else "Unknown",
                "description": str(row[desc_col]) if pd.notna(row[desc_col]) else ""
            }
            entities_data.append(entity_info)
        
        # Write to JSON file
        output_file = "entities_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully extracted {len(entities_data)} entities to {output_file}")
        print(f"Sample of extracted data:")
        for i, entity in enumerate(entities_data[:3]):
            print(f"  {i+1}. ID: {entity['entity_id']}")
            print(f"     Description: {entity['description'][:100]}...")
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    extract_entities_to_json() 