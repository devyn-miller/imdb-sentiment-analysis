import gradio as gr
from predict import predict_sentiment

def predict_and_visualize(review):
    sentiment = predict_sentiment(review)
    return sentiment

iface = gr.Interface(fn=predict_and_visualize, inputs="text", outputs="text")
iface.launch(host="localhost", share=True)
