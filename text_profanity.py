# -*- coding: utf-8 -*-

import spacy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import emoji
import nltk
from nltk.corpus import stopwords
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from googletrans import Translator
from flask import Flask, request, jsonify

# Define the path for the CSV file and load necessary resources
vector_csv_path = r'D:\Rohit\TextProfanity\latest_15\text_profanity\vector_data.csv'

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

app = Flask(__name__)

# Custom transformer to remove standalone special characters
class StandaloneSpecialCharRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pattern = r'(?<!\w)[^\w\s](?!\w)'
        return [re.sub(pattern, '', text) for text in X]

# Custom transformer to convert text to lowercase
class LowercaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [text.lower() for text in X]

# Custom transformer to convert emojis to textual representation
class EmojiTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [emoji.demojize(text) for text in X]

# Custom transformer to remove stopwords
class StopwordsRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join([word for word in text.split() if word not in stop_words]) for text in X]

# Load the SpaCy model
nlp = spacy.load('en_core_web_lg')

# Load the joblib preprocessing pipeline
preprocessing_pipeline = joblib.load(r'D:\Rohit\TextProfanity\latest_15\text_profanity\text_preprocessing_pipeline.joblib')

# Function to load CSV with a fallback encoding
def load_csv_with_fallback(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')
    return df

# Load the predefined words dictionary
def load_words(file_path):
    with open(file_path, 'r') as file:
        return set(line.strip().lower() for line in file)

# Check for forbidden words
def contains_forbidden_word(input_text, forbidden_words):
    words = input_text.split()
    return any(word.lower() in forbidden_words for word in words)

def transliterate_and_translate(text):
    # Transliterate Hindi text to Roman script (ITRANS)
    transliterated_text = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    print(f"Transliteration: {transliterated_text}")

    # Translate the transliterated text to English
    translator = Translator()
    translation = translator.translate(transliterated_text, src='hi', dest='en')
    print(f"Translation: {translation.text}")

    return transliterated_text, translation.text

# Load the forbidden words dictionary
forbidden_words_path = r'D:\Rohit\TextProfanity\latest_15\text_profanity\abusive_words.txt'
forbidden_words = load_words(forbidden_words_path)

# Load the vector data
vector_df = load_csv_with_fallback(vector_csv_path)

def process_user_input(user_input, vector_df=vector_df, nlp=nlp, preprocessing_pipeline=preprocessing_pipeline, forbidden_words=forbidden_words):
    # Initialize variables for transliterated and translated text
    transliterated_text = None
    translated_text = None

    # Function to save new entries to the vector DataFrame
    def save_entries(entries):
        new_entries_df = pd.DataFrame(entries)
        vector_df_updated = pd.concat([vector_df, new_entries_df], ignore_index=True)
        vector_df_updated.to_csv(vector_csv_path, index=False)
        print(f"Vector DataFrame saved to: {vector_csv_path}")
        return vector_df_updated

    # Detect if the input is in Hindi and translate it
    if re.search('[\u0900-\u097F]', user_input):
        transliterated_text, translated_text = transliterate_and_translate(user_input)
        translated_for_processing = translated_text

        # Create and save the entry for the original Hindi input
        original_vector = nlp(user_input).vector
        entries = [{
            "cleaned_text": user_input,
            "label": 1,
            "vector": ','.join(map(str, original_vector))
        }]

        # Save entries and update the vector DataFrame
        vector_df = save_entries(entries)
    else:
        translated_for_processing = user_input

    # Check for forbidden words before processing
    if contains_forbidden_word(translated_for_processing, forbidden_words):
        print("Your input contains restricted words from the dictionary. Please remove them and try again.")

        # Save the entries for transliterated and translated text if they exist
        entries = []

        if transliterated_text:
            transliterated_vector = nlp(transliterated_text).vector
            entries.append({
                "cleaned_text": transliterated_text,
                "label": 1,
                "vector": ','.join(map(str, transliterated_vector))
            })

        if translated_text:
            translated_vector = nlp(translated_text).vector
            entries.append({
                "cleaned_text": translated_text,
                "label": 1,
                "vector": ','.join(map(str, translated_vector))
            })

        # Save entries and update the vector DataFrame
        if entries:
            vector_df = save_entries(entries)

        return None, None, None, None  # Ensure all four variables are returned as None

    # Preprocess user input using the joblib pipeline
    cleaned_user_input = preprocessing_pipeline.transform([translated_for_processing])[0]

    # Process user input
    user_doc = nlp(cleaned_user_input)
    user_vector = user_doc.vector

    # Extract vectors from the vector DataFrame
    vectors = vector_df['vector'].apply(lambda x: np.array([float(i) for i in x.split(',')]) if isinstance(x, str) else np.zeros(len(user_vector))).values
    vectors = np.stack(vectors)

    # Compute cosine similarity
    similarities = cosine_similarity([user_vector], vectors)

    # Get the most similar vector (highest similarity score)
    most_similar_index = similarities[0].argmax()
    most_similar_entry = vector_df.iloc[most_similar_index]
    similarity_score = similarities[0][most_similar_index]

    # Define threshold for similarity
    threshold = 0.8

    if similarity_score >= threshold:
        # Assign the corresponding label
        assigned_label = most_similar_entry['label']
        print(f"Assigned label based on similarity: {assigned_label}, Score: {similarity_score}")
        hf_label_score = None  # No Hugging Face score needed
        assigned_based_on_similarity = True
    else:
        # Load the Hugging Face model for text classification
        classifier = pipeline('text-classification', model="parsawar/profanity_model_3.1")

        # Classify the text
        result = classifier(cleaned_user_input)
        classified_label = result[0]['label']
        hf_label_score = result[0]['score']
        print(f"Assigned label based on Hugging Face model: {classified_label}, Score: {hf_label_score}")

        # Convert the classified label to an integer if necessary (depends on your specific labels)
        assigned_label = classified_label  # Modify this line based on your label format
        assigned_based_on_similarity = False

    # Save the user input and assigned label to the vector CSV
    new_vector_entry = {
        "cleaned_text": translated_for_processing,
        "label": assigned_label,
        "vector": ','.join(map(str, user_vector))  # Save the vector as a comma-separated string
    }

    # Add the new entry to the vector DataFrame
    new_vector_entry_df = pd.DataFrame([new_vector_entry])  # Convert the new entry to a DataFrame
    vector_df = pd.concat([vector_df, new_vector_entry_df], ignore_index=True)  # Concatenate the new entry DataFrame with the existing DataFrame

    # Save the updated vector DataFrame back to CSV
    vector_df.to_csv(vector_csv_path, index=False)
    print(f"Vector DataFrame saved to: {vector_csv_path}")

    return similarity_score, assigned_label, hf_label_score, assigned_based_on_similarity


@app.route('/detect', methods=['POST'])
def score():
    text = request.form.get('text')
    print(f"Received text: {text}")  # Debug print to check received text
    
    if text is None or text.strip() == '':
        return jsonify({'error': "Please input text."}), 400
    
    # Process user input and get similarity score and labels
    similarity_score, assigned_label_similarity, hf_label_score, assigned_based_on_similarity = process_user_input(text, vector_df, nlp, preprocessing_pipeline, forbidden_words)
    
    # Check if similarity_score is None
    if similarity_score is None:
        return jsonify({'error': "Your text contains restricted words. Please remove them and try again."}), 400
    
    # Prepare the response based on assignment method
    if assigned_based_on_similarity:
        response = {
            'label': assigned_label_similarity,
            'score': similarity_score,
            'from': 'vector similarity'
        }
    else:
        response = {
            'label': assigned_label_similarity,  # Assuming assigned_label_similarity is used for Hugging Face model
            'score': hf_label_score,
            'from': 'hugging face model'
        }

    return jsonify(response), 200

if __name__ == "__main__":
    app.run(debug=True)

# @app.route('/detect', methods=['POST'])
# def score():
#     text = request.form.get('text')
#     if text is None:
#         return jsonify({'error': "Please input text."})
    
#     # Process user input and get similarity score and labels
#     similarity_score, assigned_label_similarity, hf_label_score, assigned_based_on_similarity = process_user_input(text, vector_df, nlp, preprocessing_pipeline, forbidden_words)
    
#     # Check if similarity_score is None
#     if similarity_score is None:
#         return jsonify({'error': "your text contains restricted words please remove it and try again."})
    
#     # Prepare the response based on assignment method
#     if assigned_based_on_similarity:
#         response = {
#             'label': assigned_label_similarity,
#             'score': similarity_score,
#             'from': 'vector similarity'
#         }
#     else:
#         response = {
#             'label': assigned_label_similarity,  # Assuming assigned_label_similarity is used for Hugging Face model
#             'score': hf_label_score,
#             'from': 'hugging face model'
#         }
    
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)


