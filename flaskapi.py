import subprocess
subprocess.run(["pip","install","-r","requirements.txt"],check=True)
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load the models and tokenizers outside the route to avoid reloading on each request
arabic_model_name = 'CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment'
arabic_tokenizer = AutoTokenizer.from_pretrained(arabic_model_name)
arabic_model = AutoModelForSequenceClassification.from_pretrained(arabic_model_name)

english_model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
english_tokenizer = AutoTokenizer.from_pretrained(english_model_name)
english_model = AutoModelForSequenceClassification.from_pretrained(english_model_name)


# Define the function to analyze sentiment
def analyze_sentiment(text):
    max_length = 512

    if any(char in text for char in 'أبجدية'):  # Arabic text
        inputs = arabic_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = arabic_model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        sentiment = torch.argmax(predictions, dim=1).item()
        return 1 if sentiment == 0 else 0  # Positive: 1, Negative: 0
    else:  # English text
        inputs = english_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = english_model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        sentiment_index = torch.argmax(predictions, dim=1).item()
        return 1 if sentiment_index == 2 else 0  # Positive: 1, Negative/Neutral: 0


# Define the Flask route for sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    print(data)
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    sentiment = analyze_sentiment(text)
    result = 'Positive' if sentiment == 1 else 'Negative'

    return jsonify({'sentiment': result})


if __name__ == '__main__':
    app.run(debug=True)
