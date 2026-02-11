import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.freq_threshold = freq_threshold
        self.word_freq = Counter()

    def build_vocabulary(self, captions):
        """Build vocabulary from list of captions"""
        for caption in captions:
            tokens = caption.split()
            self.word_freq.update(tokens)

        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, caption):
        """Convert caption string to list of indices"""
        tokens = caption.split()
        return [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]

    def decode(self, indices):
        """Convert indices to caption string"""
        return ' '.join([self.idx2word.get(idx, '<unk>') for idx in indices])

    def __len__(self):
        return len(self.word2idx)


class Encoder(nn.Module):
    """Projects 2048-dim image features to hidden_size"""

    def __init__(self, image_feature_dim=2048, hidden_size=768):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(image_feature_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)

    def forward(self, image_features):
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        x = self.fc(image_features)
        if x.size(0) == 1:
            self.bn.eval()
        x = self.bn(x)
        return x  # (batch_size, hidden_size)


class Decoder(nn.Module):
    """LSTM Decoder"""

    def __init__(self, vocab_size, embed_size=384, hidden_size=768, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        features: (batch_size, hidden_size)
        captions: (batch_size, seq_len)
        """
        embeddings = self.embed(captions)

        # Initialize hidden state from encoder
        h0 = features.unsqueeze(0)  # (1, batch, hidden)
        c0 = torch.zeros_like(h0)

        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.fc(lstm_out)

        return outputs

    def sample(self, features, start_idx, end_idx, max_len=20):
        """Greedy decoding"""
        device = features.device

        if features.dim() == 1:
            features = features.unsqueeze(0)

        # Initialize hidden state
        h = features.unsqueeze(0)
        c = torch.zeros_like(h)

        # Start token
        inputs = torch.tensor([[start_idx]], device=device)
        inputs = self.embed(inputs)

        predicted_ids = []

        for _ in range(max_len):
            lstm_out, (h, c) = self.lstm(inputs, (h, c))
            outputs = self.fc(lstm_out.squeeze(1))
            predicted = outputs.argmax(1)
            predicted_ids.append(predicted.item())

            if predicted.item() == end_idx:
                break

            inputs = self.embed(predicted.unsqueeze(0))

        return predicted_ids


class ImageCaptioningModel(nn.Module):

    def __init__(self, vocab_size, embed_size=384, hidden_size=768, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder(2048, hidden_size)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers)

    def forward(self, image_features, captions):
        features = self.encoder(image_features)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image_feature, vocab, max_length=20):
        self.eval()
        device = next(self.parameters()).device

        if image_feature.dim() == 1:
            image_feature = image_feature.unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(image_feature.to(device))

            predicted_ids = self.decoder.sample(
                features,
                start_idx=vocab.word2idx['<start>'],
                end_idx=vocab.word2idx['<end>'],
                max_len=max_length
            )

        words = []
        for idx in predicted_ids:
            word = vocab.idx2word.get(idx, "<unk>")
            if word not in ["<start>", "<end>", "<pad>", "<unk>"]:
                words.append(word)

        return " ".join(words)
