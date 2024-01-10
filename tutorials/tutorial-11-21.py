from transformers import GPT2Model, GPT2Config
# This loads an architecture that is randomly initialized
config = GPT2Config()
model = GPT2Model(config)
# If we want a model to be loaded with pre-trained weights, we need to say:
model_pre = GPT2Model.from_pretrained("gpt2")

print(config)

# Let's talk about different types of tokenizers
# First type looks at giving an id to every word in a sentence or sequence.
Text = "My name is Arman."
token_pre = Text.split()
print(token_pre)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Step 1 - break the input text to tokens
token = tokenizer.tokenize(Text)
print(token)
# Step 2 - generate ids
ids = tokenizer.convert_tokens_to_ids(token)
print(ids)
# Step 3 - add the special token ids
final_ids = tokenizer.prepare_for_model(ids)
print(final_ids)
# Now let's compare this to just the tokenize function
compare_ids = tokenizer(Text,return_tensors="pt")
print(compare_ids['input_ids'])

# We can decode the tokens back to text
text_back = tokenizer.decode(final_ids['input_ids'])
print(text_back)