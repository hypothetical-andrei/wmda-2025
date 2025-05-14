from transformers import pipeline

# Load pipelines
bert_fill = pipeline("fill-mask", model="bert-base-uncased")
gpt_gen = pipeline("text-generation", model="gpt2")

# BERT example: Masked word prediction

# GPT example: Prompt continuation
