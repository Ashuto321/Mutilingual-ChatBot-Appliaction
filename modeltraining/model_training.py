import os
import random
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

# Ensure necessary downloads
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents.json with absolute path
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "intents.json")

# Ensure the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"🚨 ERROR: 'intents.json' not found in {base_path}. Please move it there!")

with open(file_path, encoding="utf-8") as file:
    intents = json.load(file)

# Initialize lists
words = []
classes = []
documents = []
ignore_symbols = ['?', '!', '.', ',']

# Process patterns and intents
for intent in intents['intents']:
    for pattern in intent['patterns']:  # Fixed key name
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and sort
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_symbols]
words = sorted(set(words))
classes = sorted(set(classes))

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save processed words and classes
pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0] if word not in ignore_symbols]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Convert training data to NumPy array
train_x = np.array([np.array(b) for b, _ in training])
train_y = np.array([np.array(o) for _, o in training])

# Build Neural Network
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile Model (Removed weight_decay)
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train Model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1, callbacks=[early_stopping])

# Save Model
model.save('model/chatbot_model.keras')
print("✅ Model Training Completed Successfully!")
