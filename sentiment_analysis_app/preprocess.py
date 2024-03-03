from transformers import BertTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    # Convert sentiments to binary
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    reviews = df['review'].values
    labels = df['sentiment'].values

    # Split the data
    train_reviews, test_reviews, train_labels, test_labels = train_test_split(reviews, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_reviews.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_reviews.tolist(), truncation=True, padding=True)

    return train_encodings, train_labels, test_encodings, test_labels
