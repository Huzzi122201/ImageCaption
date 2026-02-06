
import torch
import torch.nn as nn
import torch.nn.functional as F

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, words):
        """Convert list of words to indices"""
        return [self.word2idx.get(word, self.word2idx.get('<unk>', 0)) for word in words]
    
    def decode(self, indices):
        """Convert list of indices to sentence"""
        words = [self.idx2word.get(idx, '<unk>') for idx in indices 
                 if idx not in [self.word2idx.get('<pad>', 0), 
                               self.word2idx.get('<start>', 1), 
                               self.word2idx.get('<end>', 2)]]
        return ' '.join(words)

class Encoder(nn.Module):
    def __init__(self, image_feature_dim=2048, hidden_size=512):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(image_feature_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, image_features):
        # Ensure batch dimension for BatchNorm1d
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        x = self.fc(image_features)
        # Handle single sample case for BatchNorm
        if x.size(0) == 1:
            self.bn.eval()  # Use running stats for single sample
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, captions, hidden_state, cell_state=None):
        embeddings = self.embedding(captions)
        embeddings = self.dropout(embeddings)
        
        h0 = hidden_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        if cell_state is None:
            c0 = torch.zeros_like(h0)
        else:
            c0 = cell_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        lstm_out, (hn, cn) = self.lstm(embeddings, (h0, c0))
        outputs = self.fc(lstm_out)
        
        return outputs, (hn, cn)

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder(image_feature_dim=2048, hidden_size=hidden_size)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers)
    
    def forward(self, image_features, captions):
        hidden_state = self.encoder(image_features)
        outputs, _ = self.decoder(captions, hidden_state)
        return outputs
    
    def generate_caption(self, image_feature, vocab, max_length=20, method='greedy', beam_size=3):
        self.eval()
        with torch.no_grad():
            if method == 'greedy':
                return self._greedy_search(image_feature, vocab, max_length)
            elif method == 'beam':
                return self._beam_search(image_feature, vocab, max_length, beam_size)
    
    def _greedy_search(self, image_feature, vocab, max_length):
        device = next(self.parameters()).device
        
        if image_feature.dim() == 1:
            image_feature = image_feature.unsqueeze(0)
        
        hidden = self.encoder(image_feature.to(device))
        input_word = torch.tensor([[vocab.word2idx['<start>']]], device=device)
        caption = []
        
        h = hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        c = torch.zeros_like(h)
        
        for _ in range(max_length):
            embeddings = self.decoder.embedding(input_word)
            lstm_out, (h, c) = self.decoder.lstm(embeddings, (h, c))
            outputs = self.decoder.fc(lstm_out.squeeze(1))
            predicted = outputs.argmax(1)
            predicted_idx = predicted.item()
            
            if predicted_idx == vocab.word2idx['<end>']:
                break
            
            caption.append(predicted_idx)
            input_word = predicted.unsqueeze(0)
        
        return vocab.decode(caption)
    
    def _beam_search(self, image_feature, vocab, max_length, beam_size):
        device = next(self.parameters()).device
        
        if image_feature.dim() == 1:
            image_feature = image_feature.unsqueeze(0)
        
        hidden = self.encoder(image_feature.to(device))
        h = hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        c = torch.zeros_like(h)
        
        beams = [([vocab.word2idx['<start>']], 0.0, h, c)]
        completed = []
        
        for _ in range(max_length):
            candidates = []
            
            for seq, score, h_state, c_state in beams:
                if seq[-1] == vocab.word2idx['<end>']:
                    completed.append((seq, score))
                    continue
                
                input_word = torch.tensor([[seq[-1]]], device=device)
                embeddings = self.decoder.embedding(input_word)
                lstm_out, (h_new, c_new) = self.decoder.lstm(embeddings, (h_state, c_state))
                outputs = self.decoder.fc(lstm_out.squeeze(1))
                log_probs = F.log_softmax(outputs, dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_size)
                
                for i in range(beam_size):
                    next_word = top_indices[0, i].item()
                    next_score = score + top_log_probs[0, i].item()
                    candidates.append((seq + [next_word], next_score, h_new, c_new))
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            if not beams:
                break
        
        completed.extend([(seq, score) for seq, score, _, _ in beams])
        
        if completed:
            best_seq = max(completed, key=lambda x: x[1])[0]
            caption_indices = [idx for idx in best_seq if idx not in [vocab.word2idx['<start>'], vocab.word2idx['<end>']]]
            return vocab.decode(caption_indices)
        
        return ""
