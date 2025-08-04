import pandas as pd
import json

# Load the data
entities_df = pd.read_parquet('graphrag_workspace/output/entities.parquet')
relationships_df = pd.read_parquet('graphrag_workspace/output/relationships.parquet')

# Convert entities to nodes format
nodes = []
for _, row in entities_df.iterrows():
    nodes.append({
        'id': row['id'],
        'title': row['title'],
        'type': row['type'],
        'description': row.get('description', ''),  # Include description
        'frequency': row['frequency'],
        'degree': row['degree']
    })

# Convert relationships to links format
links = []
for _, row in relationships_df.iterrows():
    links.append({
        'source': row['source'],
        'target': row['target'],
        'description': row['description'],
        'weight': row['weight'],
        'combined_degree': row['combined_degree']
    })

# Save as JSON
graph_data = {
    'nodes': nodes,
    'links': links
}

with open('graph_data.json', 'w') as f:
    json.dump(graph_data, f, indent=2)

print(f"Loaded {len(nodes)} nodes and {len(links)} links")
print("Data saved to graph_data.json") 