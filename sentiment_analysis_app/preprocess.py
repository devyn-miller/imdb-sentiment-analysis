from transformers import BertTokenizer
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_reviews(reviews, max_length=512):
    return tokenizer(reviews, padding=True, truncation=True, max_length=max_length, return_tensors="tf")

def load_and_preprocess_data(max_length=512):
    train_data, test_data = tfds.load(name="imdb_reviews", split=('train', 'test'), as_supervised=True)
    train_reviews, train_labels = [], []
    test_reviews, test_labels = [], []

    for review, label in tfds.as_numpy(train_data):
        train_reviews.append(review.decode('utf-8'))
        train_labels.append(label)
    
    for review, label in tfds.as_numpy(test_data):
        test_reviews.append(review.decode('utf-8'))
        test_labels.append(label)

    train_encodings = encode_reviews(train_reviews, max_length)
    test_encodings = encode_reviews(test_reviews, max_length)

    train_labels = to_categorical(train_labels, num_classes=2)
    test_labels = to_categorical(test_labels, num_classes=2)

    return train_encodings, train_labels, test_encodings, test_labels
