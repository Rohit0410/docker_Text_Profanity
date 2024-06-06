import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import streamlit as st
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
import pandas as pd

from flask import jsonify,request,Flask

pipe = pipeline("text-classification", model="parsawar/profanity_model2.0")

app = Flask(__name__)

def preprocessing(document):
    profanity_words={}
    text1 = document.replace('\n', '').lower()
    text = re.sub(r'[^\x00-\x7F]+', '', text1)  # Remove non-ASCII characters
    tokens = word_tokenize(text)
    token1 = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]  # Remove punctuation
    tokens = [token for token in tokens if token]  # Remove empty tokens
    filtered_tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    preprocessed_text = ' '.join(filtered_tokens)
    print('kkkkkkkkkkkkk',preprocessed_text)
    # for i in filtered_tokens:
    #     profanity_words[i]=pipe(i)[0]['label']
    #     # print(pipe(i))
    # df = pd.DataFrame(profanity_words.items(), columns=['words','Label'])
    # df1 = df[df['Label']=='profanity']
    # if len(df1)>=1:
    #     profanity_word=list(df1['words'])
    #     for i in range(len(tokens)):  # Iterate over tokens, not text1
    #         for j in profanity_word:
    #             if tokens[i] == j:
    #                 length = len(j) - 2
    #                 replace_word = j[0] + '*' * length + j[-1]
    #                 tokens[i] = replace_word
    #         preprocessed_text = ' '.join(tokens)
    #         data = {'label':'Profanity','text':preprocessed_text}
    #         return data.
    profanity = pipe(preprocessed_text)[0]
    data = {'label':profanity,'text':preprocessed_text}
    # else:
    #     profanity_word=[]
    #     preprocessed_text = ' '.join(tokens)
    #     data = {'label':'Non-Profanity','text':preprocessed_text}
    return data

# def preprocessing(document):
#     # profanity_words={}
#     text1 = document.replace('\n', '').lower()
#     text = re.sub(r'[^\x00-\x7F]+', '', text1)  # Remove non-ASCII characters
#     tokens = word_tokenize(text)
#     # token1 = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]  # Remove punctuation
#     # tokens = [token for token in tokens if token]  # Remove empty tokens
#     filtered_tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
#     preprocessed_text = ' '.join(filtered_tokens)
#     print('kkkkkkkkkkkkk',preprocessed_text)
#     label=pipe(preprocessed_text)
    
#     print('ccccccccccccc',preprocessed_text)
#     return label

@app.route('/detect', methods=['POST'])
def detect_profanity():
    text = request.form.get('text')
    if text is None:
        return jsonify({'error': 'No text provided in the form data'}), 400
    
    preprocessed_text1 = preprocessing(text)
    return jsonify(preprocessed_text1)

if __name__ == '__main__':
    app.run(debug=True)