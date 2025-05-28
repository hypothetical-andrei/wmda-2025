from transformers import pipeline

# Load pipelines
bert_fill = pipeline("fill-mask", model="bert-base-uncased")
gpt_gen = pipeline("text-generation", model="gpt2")

# BERT example: Masked word prediction
print("BERT - Masked Word Prediction")
sentence = "The capital of France is [MASK]."
for pred in bert_fill(sentence):
    print(f"{pred['sequence']} (score: {pred['score']:.4f})")

# GPT example: Prompt continuation
print("\nGPT - Prompt Continuation")
prompt = "Once upon a time, in a small village,"
for out in gpt_gen(prompt, max_length=30, num_return_sequences=1):
    print(out['generated_text'])