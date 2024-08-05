
from sklearn.cluster import KMeans

def perform_clustering(X, n_clusters):
    """Cluster data using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_

# Example usage
if __name__ == "__main__":
    import pandas as pd
    from preprocessing import preprocess_text
    df = pd.read_json('/Users/barniabouikila/Desktop/Clustering_of_learnings/raw_data.jsonl', lines=True)  # For testing purposes
    X = preprocess_text(df, 'text') 
    labels = perform_clustering(X, n_clusters=5)  # We can adjust the number of clusters we want
    print(labels)
