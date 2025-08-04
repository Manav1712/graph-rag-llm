import pandas as pd
import json
import os

def extract_community_reports_to_json():
    """Extract community reports from parquet file to JSON"""
    reports_file = "graphrag_workspace/output/community_reports.parquet"
    
    if not os.path.exists(reports_file):
        print(f"Error: {reports_file} not found")
        return
    
    try:
        # Read the parquet file
        df = pd.read_parquet(reports_file)
        
        print(f"Successfully loaded community reports parquet file")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Identify relevant columns
        relevant_columns = []
        for col in df.columns:
            col_lower = col.lower()
            # Look for common report-related columns
            if any(keyword in col_lower for keyword in ['report', 'name', 'title', 'content', 'text', 'description', 'summary', 'community', 'id']):
                relevant_columns.append(col)
        
        # If no specific columns found, use all columns
        if not relevant_columns:
            relevant_columns = list(df.columns)
        
        print(f"\nUsing columns: {relevant_columns}")
        
        # Extract the data
        reports_data = []
        for _, row in df.iterrows():
            report_info = {}
            for col in relevant_columns:
                value = row[col]
                # Handle different data types
                if pd.isna(value):
                    report_info[col] = None
                elif isinstance(value, (int, float)):
                    report_info[col] = value
                else:
                    report_info[col] = str(value)
            reports_data.append(report_info)
        
        # Save to JSON file
        output_file = "community_reports_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reports_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully extracted {len(reports_data)} community reports to {output_file}")
        print(f"Sample report structure:")
        if reports_data:
            print(json.dumps(reports_data[0], indent=2))
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_community_reports_to_json() 