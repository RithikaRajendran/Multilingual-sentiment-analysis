from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = load_model("model.h5")

# Vocabulary size and sentence length
voc_size = 20000
sent_length = 10
# Define label as a global variable
label=['Mixed_feelings','Negative','Positive','unknown_state']

# Preprocessing function
def preprocess_text(text):
    import string
    # Remove punctuations
    text_nopunc = "".join([c for c in text if c not in string.punctuation])
    # One hot encoding
    text_onehot = [one_hot(text_nopunc, voc_size)]
    # Padding sequence
    text_padded = pad_sequences(text_onehot, padding='pre', maxlen=sent_length)
    return text_padded

# Prediction function
def predict_sentiment(text_padded):
    sentiment = model.predict(text_padded)
    output1 = np.argmax(sentiment)
    output = label[output1]
    return output

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_padded = preprocess_text(text)
    sentiment = predict_sentiment(text_padded)
    return render_template('result.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
