# no gold challenge cleansing

import tempfile
import os
import csv
import re
import pandas as pd
import sqlite3
import pickle
import numpy as np
from flask import Flask, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier # Neural-Network Model library
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from keras.models import load_model

import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

app = Flask(__name__)

from flask import request
from flasgger import Swagger, LazyJSONEncoder
from flasgger import swag_from

app.json_encoder = LazyJSONEncoder

swagger_template = {
    'info' : {
        'title': 'API Documentation for Data Processing and Modeling',
        'version': '1.0.0',
        'description': 'Dokumentasi API untuk Data Processing dan Modeling',
        },
    'host' : '127.0.0.1:5000'
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,             
                  config=swagger_config)

# Checking data if there's an empty data 
def get_first_non_empty_column(file_path):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            for value in row:
                if value.strip():  # Check if the value is not empty or whitespace
                    return row.index(value)

# Function to create a SQLite database and insert data
def create_and_insert_into_db(data, cleaned_data, prediction, x):
    db_path = 'data_clean1.db'

    # Using with statement for SQLite connection
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS combined_data (
                Text TEXT,
                Text_clean TEXT,
                Sentiment TEXT
            )
        ''')
        #insert data for text input - neural network
        if x==1:
            cursor.execute('INSERT INTO combined_data (Text, Text_clean, Sentiment) VALUES (?, ?, ?)', (data, cleaned_data, prediction))

        # Insert cleaned data into the combined_data table for file - neural network
        if x==2:
            for text_raw, text_clean, predict in zip(data, cleaned_data, prediction):
                cursor.execute('INSERT INTO combined_data (Text, Text_clean, Sentiment) VALUES (?, ?, ?)', (text_raw, text_clean, predict))

        # Commit the changes
        conn.commit()

# delete words/char
def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('rt',' ',text) # Remove every reTweet symbol
    text = re.sub('user',' ',text) # Remove every username
    text = text.strip() #menghapus spasi di awal dan akhir
    text = re.sub(r'\n', ' ', text, flags=re.IGNORECASE)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub(r'(.)\1\1+', r'\1', text) #menghapus karakter berulang
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) #menghapus karakter non-alpanumerik
    text = re.sub(r'[øùºðµ¹ª³]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'â', 'a', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip() #menghapus spasi berlebih dan mengganti dengan satu spasi
    text = re.sub(r'^\s+$', '', text) #menghapus seluruh kalimat yg hanya berisi spasi
    text = re.sub(r'https\S+|[^a-zA-Z0-9]', ' ', text) # remove emoticon 
    return text

# Reduce words that start with /x
def remove_words_starting_with_x(text1):
    # Define a pattern to match words starting with \x
    pattern = re.compile(r'\b\\x\w*\b')
    
    # Replace the matched words with an empty string
    text = pattern.sub('', text1)
    
    return text

# Stopwords removal func
def stopwordsremoval(text):

     # Tokenize text
    freq_tokens = word_tokenize(text.lower())  # Convert text to lowercase for consistent comparison

    # Get Indonesian and English stopwords
    list_stopwords_id = list(set(stopwords.words('indonesian')))  # Convert set to list
    list_stopwords_en = list(set(stopwords.words('english')))     # Convert set to list

    # Extend the list with additional stopwords
    additional_stopwords = ['ya', 'yg', 'ga', 'yuk', 'dah', 'nya']
    list_stopwords_id.extend(additional_stopwords)

    # Combine Indonesian and English stopwords
    list_stopwords = list_stopwords_id + list_stopwords_en

    # Specify words to retain
    words_to_retain = ['tidak', 'baik', 'kurang']  # Add more words as needed (kurang, baik)

    # Remove stopwords from the tokenized text, except words to retain
    tokens_without_stopword = [word for word in freq_tokens if word not in list_stopwords or word in words_to_retain]

    # Reconstruct the text without stopwords
    text_without_stopwords = ' '.join(tokens_without_stopword)

    # print(text_without_stopwords)
    return text_without_stopwords

# func. stemming words -- indonesian only 
def stemming_words(text):
    # Tokenize the single sentence into words
    list_tokens = word_tokenize(text)

    # Stem each word in the list of tokens
    stemmed_tokens = [stemmer.stem(token) for token in list_tokens]

    # Reconstruct the output after stemming
    output = ' '.join(stemmed_tokens)

    return output

# func. kamus alay :
def replace_kamusalay(text1):
    # REPLACE TEXT
    replace_words = {}
    with open('Data/Data klasifikasi/new_kamusalay.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            replace_words[row[0]] = row[1]

    # Replace alay text 
    processed_text = text1  # Define processed_text before the loop
    for word, replacement in replace_words.items():
        processed_text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, processed_text)

    return processed_text

# CLEANSING1 for model prediction 
def cleansing_model (data_input,i ):

    if i == 1:
        text1 = data_input.lower()
        text1 = remove_unnecessary_char(text1)
        text1 = remove_words_starting_with_x(text1)
        text1 = replace_kamusalay(text1)
        text1 = stopwordsremoval(text1)
        cleaned_texts = stemming_words(text1)

    if i ==2:
        cleaned_texts = []
        for text in data_input:
            text1 = text.lower()
            text1 = remove_unnecessary_char(text1)
            text1 = remove_words_starting_with_x(text1)
            text1 = replace_kamusalay(text1)
            text1 = stopwordsremoval(text1)
            text1 = stemming_words(text1)
            cleaned_texts.append(text1)

    return cleaned_texts

# Load the trained vectorizer and model NN 
vectorizer = pickle.load(open("feature.p", "rb"))
model_NN = pickle.load(open("model.p", "rb"))

# =========================================================================

# PREDICTION MODEL Neural-Network
def prediction_sentiment_NN(text_bersih, vectorizer, model, y):
    
    if y==1 :
        text_vectorized = vectorizer.transform([text_bersih])
        sentiment = model.predict(text_vectorized)[0]
    if y==2 :
        text_vectorized = vectorizer.transform(text_bersih)
        sentiment = model.predict(text_vectorized)
    return sentiment

# =========================================================================

# Load the trained vectorizer and model LSTM
sentiment = ['negative', 'neutral', 'positive']

model_LSTM = load_model('model.h5')  # -->> bisa ganti model

# PREDICTION OF MODEL LSTM 
def prediction_sentiment_LSTM(text_bersih, model, z):
    # Load tokenizer from tokenizer.pickle
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load X from x_pad_sequences.pickle
    with open('x_pad_sequences.pickle', 'rb') as handle:
        X = pickle.load(handle)

    predicted_sentiments = []
    if z == 1 :
        predicted = tokenizer.texts_to_sequences(text_bersih)
        guess = pad_sequences(predicted, maxlen=X.shape[1])

        prediction = model.predict(guess)
        polarity = np.argmax(prediction[0])
        predicted_sentiments = sentiment[polarity]

    elif z == 2 : 
        for text in text_bersih:
            predicted = tokenizer.texts_to_sequences([text])
            guess = pad_sequences(predicted, maxlen=X.shape[1])

            prediction = model.predict(guess)
            polarity = np.argmax(prediction[0])
            predicted_sentiments.append(sentiment[polarity])

    return predicted_sentiments
# =========================================================================

@app.route('/')
def default_route():
    return jsonify({'message': 'Welcome to the Platinum Challenge Kelompok 3-DSC 15!'})

@swag_from("docs/text_processing.yml", methods=['POST'])
@app.route('/text-processing_NN', methods=['POST'])
def text_processing_NN():

    text_p = request.form.get('text')
    processed_text = cleansing_model(text_p, 1)
    prediction = prediction_sentiment_NN(processed_text, vectorizer, model_NN, 1)

    #HOW TO DO THE DATABASE?
    create_and_insert_into_db(text_p, processed_text, prediction, 1)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': text_p, 
        'clean_data' : processed_text,
        'Sentiment' : prediction.tolist(),
    }

    response_data = jsonify(json_response)

    return response_data

@swag_from("docs/text_processing.yml", methods=['POST'])
@app.route('/text-processing_LSTM', methods=['POST'])
def text_processing_LSTM():

    text_p = request.form.get('text')
    processed_text = cleansing_model(text_p, 1)
    prediction = prediction_sentiment_LSTM(processed_text, model_LSTM, 1)

    #HOW TO DO THE DATABASE?
    create_and_insert_into_db(text_p, processed_text, prediction, 1)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': text_p, 
        'clean_data' : processed_text,
        'Sentiment' : prediction,
    }

    response_data = jsonify(json_response)

    return response_data

@swag_from("docs/text_processing_file.yml", methods=['POST'])
@app.route('/text-processing-file_NN', methods=['POST'])
def text_processing_file_NN():
    
    # Uploaded file
    uploaded_file = request.files.getlist('file')[0]

    # Save the file to a temporary location
    temp_file_path = tempfile.mktemp(suffix=".csv", dir=tempfile.gettempdir())
    uploaded_file.save(temp_file_path)

    # Determine the index of the first non-empty column
    first_column_index = get_first_non_empty_column(temp_file_path)

    # Define chunk size
    chunk_size = 13170  # Adjust as needed

    # Process the file in chunks
    cleaned_text = []
    raw_text = [] 
    predict_file = []
    for chunk in pd.read_csv(temp_file_path, chunksize=chunk_size, encoding='latin1'):  
        # Extract texts from the DataFrame
        column_name = chunk.columns[first_column_index]
        texts = chunk[column_name].tolist()
        # Store raw text
        raw_text.extend(texts)
        
        # Perform cleansing on texts
        cleaned_chunk = cleansing_model(texts, 2)
        cleaned_text.extend(cleaned_chunk)
        
        # Predict sentiment for each cleaned text chunk
        chunk_predictions = prediction_sentiment_NN(cleaned_chunk, vectorizer, model_NN, 2)
        predict_file.extend(chunk_predictions)
    
    # Store cleaned data into the SQLite database
    create_and_insert_into_db(raw_text, cleaned_text, predict_file, 2) 

    # Remove the temporary file
    os.remove(temp_file_path)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': raw_text, 
        'clean_data' : cleaned_text,
        'Sentiment' : predict_file,
    }

    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/text_processing_file.yml", methods=['POST'])
@app.route('/text-processing-file_LSTM', methods=['POST'])
def text_processing_file_LSTM():
    
    # Uploaded file
    uploaded_file = request.files.getlist('file')[0]

    # Save the file to a temporary location
    temp_file_path = tempfile.mktemp(suffix=".csv", dir=tempfile.gettempdir())
    uploaded_file.save(temp_file_path)

    # Determine the index of the first non-empty column
    first_column_index = get_first_non_empty_column(temp_file_path)

    # Define chunk size
    chunk_size = 13170  # Adjust as needed

    # Process the file in chunks
    cleaned_text = []
    raw_text = [] 
    predict_file = []
    for chunk in pd.read_csv(temp_file_path, chunksize=chunk_size, encoding='latin1'):  
        # Extract texts from the DataFrame
        column_name = chunk.columns[first_column_index]
        texts = chunk[column_name].tolist()
        # Store raw text
        raw_text.extend(texts)
        
        # Perform cleansing on texts
        cleaned_chunk = cleansing_model(texts, 2)
        cleaned_text.extend(cleaned_chunk)
        
        # Predict sentiment for each cleaned text chunk
        chunk_predictions = prediction_sentiment_LSTM(cleaned_chunk, model_LSTM, 2)
        predict_file.extend(chunk_predictions)
    
    # Store cleaned data into the SQLite database
    create_and_insert_into_db(raw_text, cleaned_text, predict_file, 2) 

    # Remove the temporary file
    os.remove(temp_file_path)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': raw_text, 
        'clean_data' : cleaned_text,
        'Sentiment' : predict_file,
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run(debug=True)