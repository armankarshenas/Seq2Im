from transformers import pipeline
import sys
import numpy as np
import pandas as pd
# This basic scripts show how we can use existing model to use transformers
# We have to specify the model name and parameters using pipeline
model = pipeline("sentiment-analysis")
ans = model(["Hello I am dead","Hello my name is Arman"])
print(ans[0]['label'])

# Zero shot classification
model = pipeline("zero-shot-classification")
ans = model(["AATCGCTCTC","I love animals"],candidate_labels=["DNA","personal","fruits"])
print(ans[0]['scores'])

# Text generation
model = pipeline("text-generation")
ans = model("You have been very annoying so I am going to ",max_length=50)
print(ans)

# To use pre trained models from Hugging face hub, we can use the GUI to look for the model and then use the name of the model in the pipeline method to get the model
model = pipeline("text-generation",model="distilgpt2")
ans = model("You have been very annoying so I am going to ",max_length=50)
print(ans)

# Let's break the pipeline function into sub components
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

tokenize = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
input = ["Hello I am very happy today","I am crying so hard"]
input_processed = tokenize(input,padding=True,truncation=True,return_tensors="pt")
print(input_processed['input_ids'][0])
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
output = model(**input_processed)
print(output.logits)
with torch.no_grad():
    logits = output.logits
    prob = torch.nn.functional.softmax(logits,dim=1)
    print(prob)
    print(model.config.id2label)

