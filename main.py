import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS

# Define the IMDBDataset class
class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Define the model architecture
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output[:, -1, :])
        return output

# Load the IMDB dataset from Hugging Face
def load_imdb_data():
    imdb_dataset = load_dataset('imdb')
    return imdb_dataset['train'], imdb_dataset['test']

# Tokenize the text data
def tokenize_data(dataset, tokenizer):
    encodings = tokenizer(dataset['text'], truncation=True, padding=True)
    return encodings

# Set up the data loaders
def setup_data_loaders(train_encodings, test_encodings, train_labels, test_labels, batch_size):
    train_dataset = IMDBDataset(train_encodings, train_labels)
    test_dataset = IMDBDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

# Initialize the model
def initialize_model(vocab_size, embedding_dim, hidden_dim, output_dim):
    model = SentimentModel(vocab_size, embedding_dim, hidden_dim, output_dim)
    return model

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['labels'].float()
            optimizer.zero_grad()
            outputs = model(input_ids).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Evaluate the model
def evaluate_model(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            outputs = model(input_ids).squeeze(1)
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')

# Flask web application
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    text = request.json['text']
    encoded_text = tokenizer([text], truncation=True, padding=True)
    input_ids = torch.tensor(encoded_text['input_ids'])
    output = model(input_ids).squeeze(0)
    sentiment = torch.sigmoid(output).item()
    sentiment_label = 'Positive' if sentiment >= 0.5 else 'Negative'
    return jsonify({'sentiment': sentiment_label})

if __name__ == '__main__':
    # Load the IMDB dataset from Hugging Face
    train_data, test_data = load_imdb_data()

    # Tokenize the text data
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenize_data(train_data, tokenizer)
    test_encodings = tokenize_data(test_data, tokenizer)

    # Set up the data loaders
    batch_size = 64
    train_loader, test_loader = setup_data_loaders(train_encodings, test_encodings, train_data['label'], test_data['label'], batch_size)

    # Initialize the model
    vocab_size = tokenizer.vocab_size
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1
    model = initialize_model(vocab_size, embedding_dim, hidden_dim, output_dim)

    # Set up the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    num_epochs = 5
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model
    evaluate_model(model, test_loader)

    # Run the Flask app
    app.run(debug=True)