from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sample input article
text = (
    "NASA's Perseverance rover has successfully landed on Mars, completing its journey from Earth. "
    "The rover will search for signs of ancient life and collect rock samples for a future return mission. "
    "Scientists believe this mission could provide insights into whether life ever existed on Mars."
)

# Generate summary

# Print result
