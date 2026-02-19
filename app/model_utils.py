import torch
import torch.nn as nn
import torch.nn.functional as F
import re


class MyanmarTextPreprocessor:
    """Preprocessor for Myanmar text - syllable-level tokenization"""
    def __init__(self):
        self.syllable_pattern = r"(([A-Za-z0-9]+)|[က-အ|ဥ|ဦ](င်္|[က-အ][ှ]*[့း]*[်]|္[က-အ]|[ါ-ှႏꩻ][ꩻ]*){0,}|.)"
    
    def tokenize_syllable(self, text: str):
        """Syllable-level tokenization"""
        text = re.sub(self.syllable_pattern, r"\1 ", text)
        tokens = text.strip().split()
        return tokens


class BiLSTMSeq2SeqWithAttention(nn.Module):
    """BiLSTM Seq2Seq model with attention for headline generation"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        dropout = 0.3 if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                      batch_first=True, bidirectional=True, dropout=dropout)
        
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.attention = nn.Linear(hidden_dim * 3, 1)
        self.decoder = nn.LSTM(embedding_dim + hidden_dim * 2, hidden_dim, 
                       num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        
        embedded = self.embedding(src)
        enc_out, (h, c) = self.encoder(embedded)
        
        h = h.view(self.num_layers, 2, batch_size, self.hidden_dim)
        c = c.view(self.num_layers, 2, batch_size, self.hidden_dim)
        h = torch.cat([h[:, 0], h[:, 1]], dim=2)
        c = torch.cat([c[:, 0], c[:, 1]], dim=2)
        h = torch.tanh(self.bridge_h(h))
        c = torch.tanh(self.bridge_c(c))
        
        outputs = torch.zeros(batch_size, tgt_len, self.vocab_size).to(src.device)
        inp = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            emb = self.embedding(inp)
            h_rep = h[-1].unsqueeze(1).repeat(1, enc_out.size(1), 1)
            attn = torch.cat([h_rep, enc_out], dim=2)
            w = F.softmax(self.attention(attn).squeeze(2), dim=1)
            ctx = torch.bmm(w.unsqueeze(1), enc_out)
            rnn_in = torch.cat([emb, ctx], dim=2)
            out, (h, c) = self.decoder(rnn_in, (h, c))
            pred = self.fc(out)
            outputs[:, t] = pred.squeeze(1)
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            inp = tgt[:, t].unsqueeze(1) if teacher_force else pred.argmax(2)
        
        return outputs


def encode_sentence(tokens, max_len, word2idx):
    """Encode tokens to indices"""
    indices = [word2idx.get(t, word2idx["<unk>"]) for t in tokens]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [word2idx["<pad>"]] * (max_len - len(indices))
    return indices


def generate_headline(model, processor, text, word2idx, idx2word, max_text_len=256, max_head_len=20, device='cpu'):
    """Generate headline from input text"""
    model.eval()
    with torch.no_grad():
        # Tokenize input
        tokens = processor.tokenize_syllable(text)
        
        # Encode
        src = torch.tensor(
            encode_sentence(tokens, max_text_len, word2idx), 
            dtype=torch.long
        ).unsqueeze(0).to(device)
        
        # Encode
        embedded = model.embedding(src)
        enc_out, (h, c) = model.encoder(embedded)
        
        # Bridge
        h = h.view(model.num_layers, 2, 1, model.hidden_dim)
        c = c.view(model.num_layers, 2, 1, model.hidden_dim)
        h = torch.cat([h[:, 0], h[:, 1]], dim=2)
        c = torch.cat([c[:, 0], c[:, 1]], dim=2)
        h = torch.tanh(model.bridge_h(h))
        c = torch.tanh(model.bridge_c(c))
        
        # Decode
        result = []
        inp = torch.tensor([[word2idx["<sos>"]]], device=device)
        
        for _ in range(max_head_len):
            emb = model.embedding(inp)
            h_rep = h[-1].unsqueeze(1).repeat(1, enc_out.size(1), 1)
            attn = torch.cat([h_rep, enc_out], dim=2)
            w = F.softmax(model.attention(attn).squeeze(2), dim=1)
            ctx = torch.bmm(w.unsqueeze(1), enc_out)
            rnn_in = torch.cat([emb, ctx], dim=2)
            out, (h, c) = model.decoder(rnn_in, (h, c))
            pred = model.fc(out.squeeze(1)).argmax(1).item()
            
            if pred == word2idx["<eos>"]:
                break
            if pred not in [word2idx[k] for k in ["<unk>", "<pad>", "<sos>"]]:
                result.append(idx2word[pred])
            inp = torch.tensor([[pred]], device=device)
        
        return ''.join(result)