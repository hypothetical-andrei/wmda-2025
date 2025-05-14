#  [Example: Compare BERT and GPT results on completing sentences with missing words or continuing a prompt]
from transformers import pipeline

# Load pre-trained models
bert_fill_mask = pipeline("fill-mask", model="bert-base-uncased")
gpt_text_gen = pipeline("text-generation", model="gpt2")

# BERT: Fill in the missing word
bert_sentence = "The capital of France is [MASK]."
bert_results = bert_fill_mask(bert_sentence)
print("BERT Sentence Completion:")
for result in bert_results:
    print(f"{result['sequence']} (score: {result['score']:.4f})")

# GPT: Continue a given prompt
gpt_prompt = "Once upon a time, in a faraway land,"
gpt_results = gpt_text_gen(gpt_prompt, max_length=30, num_return_sequences=1)
print("\nGPT Sentence Continuation:")
for result in gpt_results:
    print(result['generated_text'])
