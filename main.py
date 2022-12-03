import streamlit as st
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import nltk
import emoji
nltk.download('punkt')


class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        # batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.view(128, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

word2idx = pickle.load(open('dict.pickle', 'rb'))

vocab_size = len(word2idx) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 512
n_layers = 2

model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model.load_state_dict(torch.load('./state_dict.pt',map_location=torch.device('cpu')))

def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

st.title('Polarity of Review')
review = st.text_input("Reviews:", value="")
sent = [review,"","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""]
for i, review in enumerate(sent):
  sent[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(review)]
sent = pad_input(sent, 200)
sent = TensorDataset(torch.from_numpy(sent))
live_loader = DataLoader(sent, shuffle=False, batch_size=128)


if st.button('Predict'):
    h = model.init_hidden(128)
    model.eval()
    h = tuple([each.data for each in h])
    for inputs in live_loader:
        inputs = inputs[0]
        output, h = model(inputs, h)
        pred = torch.round(output.squeeze())  # Rounds the output to 0/1
        break
    if int(pred[0].item()) == 0:
        st.write('Given Review is Negative ' + "U+2639")
    else:
        st.write('Given Review is Positive '+ emoji.emojize(":grinning_face_with_big_eyes:"))