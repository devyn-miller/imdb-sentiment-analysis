from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    prediction = tf.nn.softmax(outputs.logits, axis=-1)
    labels = ['Negative', 'Positive']
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label
