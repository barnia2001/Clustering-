
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(df, text_column):
    """Convert text to TF-IDF features."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df[text_column])
    return X

# Example usage
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_json('/Users/barniabouikila/Desktop/Clustering_of_learnings/raw_data.jsonl', lines=True)  # For testing purposes
    from preprocessing import preprocess_text
    X = preprocess_text(df, 'text')  
    print(X.shape)
