# [Example: Use a short article to demonstrate how an abstractive summarization model (like BERT-based) condenses the text]
from transformers import pipeline

# Load pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example short article
article = (
    "NASA's Perseverance rover has successfully landed on Mars, marking another milestone "
    "in space exploration. The rover will search for signs of ancient life and collect "
    "rock samples for a future return mission to Earth. Scientists hope the mission will "
    "provide insights into the planet's past and its potential for sustaining life."
)

# Generate summary
summary = summarizer(article, max_length=50, min_length=20, do_sample=False)

# Print summarized text
print("Original Article:")
print(article)
print("\nSummarized Text:")
print(summary[0]['summary_text'])
