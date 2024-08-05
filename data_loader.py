
import pandas as pd
import json

def load_jsonl(filepath):
    """Load JSONL file into a DataFrame."""
    with open(filepath, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    df = load_jsonl('/Users/barniabouikila/Desktop/Clustering_of_learnings/raw_data.jsonl')
    print(df.head())
