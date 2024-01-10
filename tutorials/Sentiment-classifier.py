from transformers import pipeline
from transformers import AutoTokenizer
import sys
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
file_path = "/media/zebrafish/Data2/Arman/Seq2Im/train.csv"

encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']

for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        # If reading succeeds, break out of the loop
        break
    except UnicodeDecodeError:
        print(f"Failed to decode with {encoding} encoding. Trying another encoding.")

# Display the DataFrame
print(df)

texts = np.array(df['selected_text'])
labels = np.array(df['sentiment'])
labels_numeric = np.zeros_like(labels)
print(np.shape(texts))
print(np.shape(labels))
print(np.unique(labels))

for i in range(len(labels)):
    if labels[i] == 'negative':
        labels_numeric[i] = 0
    elif labels[i] == 'neutral':
        labels_numeric[i] = 1
    else:
        labels_numeric[i] = 2

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_len = 128
X = np.zeros([len(texts),max_len])
for i in range(len(texts)):
    feature = tokenizer(str(texts[i]),padding = "max_length",truncation = True,max_length=max_len)
    X[i,:] = feature['input_ids']
print(np.shape(X))

# Now I have a feature space and a vector of labels, let's see if we can train a model to predict sentiment
svm_model = SVC(kernel='linear',C=1.0)
svm_model.fit(X,labels_numeric)
