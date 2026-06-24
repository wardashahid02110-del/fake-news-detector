import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Clean and preprocess text data"""
    if isinstance(text, float):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data(fake_path='Fake.csv', true_path='True.csv'):
    """Load Fake and True CSV files and prepare dataset"""
    print("Loading data...")
    
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Add labels: 0 = Fake, 1 = True
    fake_df['label'] = 0
    true_df['label'] = 1
    
    # Combine both datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    print(f"Total articles: {len(df)}")
    print(f"Fake news: {len(fake_df)}")
    print(f"Real news: {len(true_df)}")
    
    # Combine title and text
    df['content'] = df['title'] + ' ' + df['text']
    
    # Clean text
    print("Cleaning text...")
    df['content_clean'] = df['content'].apply(clean_text)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def split_data(df, test_size=0.2):
    """Split data into train and test sets"""
    X = df['content_clean']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_and_prepare_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print("Preprocessing complete!")
