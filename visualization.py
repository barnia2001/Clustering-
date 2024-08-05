
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_clusters(X, labels, output_path):
    """Reduce dimensionality and plot clusters."""
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X.toarray())
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_X[:, 0], reduced_X[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Cluster Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(output_path)
    plt.show()

# Example usage
if __name__ == "__main__":
    import pandas as pd
    from preprocessing import preprocess_text
    from clustering import perform_clustering
    df = pd.read_json('/Users/barniabouikila/Desktop/Clustering_of_learnings/raw_data.jsonl', lines=True)  # For testing purposes
    X = preprocess_text(df, 'text') 
    labels = perform_clustering(X, n_clusters=5)  # We can adjust the number of clusters as we want 
    visualize_clusters(X, labels, 'output/plots/cluster_plot.png')
