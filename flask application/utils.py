import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model


def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words


def bag_of_words(sentence):
    words = pickle.load(open('model/words.pkl', 'rb'))

    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    classes = pickle.load(open('model/classes.pkl', 'rb'))
    model = load_model('model/chatbot_model.keras')

    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.4

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list


def get_response(intents_list):
    """Retrieves the chatbot's response based on the predicted intent."""
    
    # Ensure the intents.json file path is correct
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    intents_json_path = os.path.join(BASE_DIR, "intents.json")

    # Check if intents.json exists
    if not os.path.exists(intents_json_path):
        return "⚠️ ERROR: 'intents.json' not found! Please ensure the file exists."

    # Load the intents.json file
    try:
        with open(intents_json_path, "r", encoding="utf-8") as file:
            intents_json = json.load(file)
    except json.JSONDecodeError:
        return "⚠️ ERROR: Failed to parse 'intents.json'. Check the JSON format!"

    # Check if intents_list is empty
    if not intents_list:
        return "I'm sorry, I don't understand that."

    tag = intents_list[0].get("intent")  # Safely get intent key
    if not tag:
        return "⚠️ ERROR: No intent found in the response."

    # Find the matching intent and return a random response
    for intent in intents_json.get("intents", []):  # Use .get() to prevent crashes
        if intent.get("tag") == tag:
            return random.choice(intent.get("responses", ["I'm not sure how to respond to that."]))

    return "I'm sorry, I don't understand that."