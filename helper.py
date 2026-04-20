import numpy as np
import pandas as pd
import re
import string
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

#loading the stopwords, model and vocabulary

with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

with open('static/model/sentiment_analysis_model.pkl', 'rb') as file:
    model = pickle.load(file)

vocab = pd.read_csv('static/model/volcabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocess_text(text):
    data = pd.DataFrame( [text], columns=['tweet'])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data["tweet"] = data ["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data["tweet"] = data["tweet"].apply(remove_punctuation)
    data["tweet"] = data["tweet"].str.replace(r'\d+', '', regex=True)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(word) for word in x.split()))
    return data["tweet"].iloc[0]

def vectorizer(ds, tokens):
    vectorized_lst = []
    for sentence in ds:
        sentence_lst= np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1
        vectorized_lst.append(sentence_lst)
    vectorized_lst_new=np.asarray(vectorized_lst, dtype=np.float32)
    return vectorized_lst_new

def get_prediction(vectorized_txt):
    prediction = model.predict(vectorized_txt)
    if prediction [0]== 0:
        return 'positive'
    else:
        return 'negative'