
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(df, text_column):
    """Convert text to TF-IDF features."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df[text_column])
    return X

# Example usage
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('path/to/your/data.csv')  # Load your CSV for testing
    X = preprocess_text(df, 'text')  # Replace 'text' with your column name
    print(X.shape)
