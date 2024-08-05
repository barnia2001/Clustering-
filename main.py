
from data_loader import load_jsonl
from preprocessing import preprocess_text
from clustering import perform_clustering
from visualization import visualize_clusters

# Main workflow
jsonl_filepath = '/Users/barniabouikila/Desktop/Clustering_of_learnings/raw_data.jsonl'  # Path to your JSONL file
df = load_jsonl(jsonl_filepath)

text_column = 'text' 
X = preprocess_text(df, text_column)

n_clusters = 5  # We can adjust the number of clusters as needed 
labels = perform_clustering(X, n_clusters)

df['Cluster'] = labels
visualize_clusters(X, labels, 'output/plots/cluster_plot.png')
