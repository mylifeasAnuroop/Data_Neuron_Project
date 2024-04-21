"""PART-A The dataset comprises pairs of text paragraphs randomly sampled from a larger pool of data.
Each pair represents a comparison, with varying degrees of semantic similarity. The goal is to develop 
an algorithm that quantifies this similarity, predicting a value between 0 and 1. This value indicates
the degree of likeness between the paired text paragraphs. The dataset, augmented with similarity scores, 
has been saved as "submission.csv."""

import pandas as pd
import numpy as np
import tensorflow_hub as hub
import re
import nltk
from nltk.corpus import stopwords

# Loading Universal Sentence Encoder model
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to preprocess text
def preprocess_text(text):
    # Convert contractions
    text = decontracted(text)
    # Convert to lowercase
    text = text.lower()
    # Remove special symbols
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Convert contractions
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Calculate similarity score
def calculate_similarity(text1, text2):
    embeddings = model([text1, text2])
    cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return cosine_similarity

# Reading the dataset
data = pd.read_csv("DataNeuron_Text_Similarity.csv")

# Preprocess columns
data['text1'] = data['text1'].apply(preprocess_text)
data['text2'] = data['text2'].apply(preprocess_text)

# Calculate similarity score for each pair of text paragraphs
similarity_scores = [calculate_similarity(text1, text2) for text1, text2 in zip(data['text1'], data['text2'])]

# Normalize similarity scores to range( 0 to 1 )
normalized_scores = (np.array(similarity_scores) + 1) / 2

# Add similarity scores to the DataFrame
data['Similarity_Score'] = normalized_scores

# Save submission to CSV with text1, text2, and similarity score columns
submission = data[['text1', 'text2', 'Similarity_Score']]
submission.to_csv('Submission.csv', index=False)
