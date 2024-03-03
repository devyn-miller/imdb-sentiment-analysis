import gradio as gr
from predict import predict_sentiment

def predict_and_visualize(review):
    sentiment = predict_sentiment(review)
    return sentiment

iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Type your review here..."),
    outputs="text",
    title="Sentiment Analysis Tool",
    description="Enter a movie review to determine its sentiment. Positive or Negative.",
    examples=[["I loved this movie, it was fantastic!"], ["I hated this movie, it was terrible."]]
)

iface.launch(share=True)
