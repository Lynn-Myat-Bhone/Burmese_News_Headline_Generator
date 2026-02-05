import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Hyperparameters
# ----------------------------
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 32
MAX_TEXT_LEN = 200
MAX_HEAD_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2idx["<pad>"])
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = True  # fine-tune embeddings

        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, trg_input):
        # src: (batch, MAX_TEXT_LEN)
        # trg_input: (batch, MAX_HEAD_LEN-1)

        # Encoder
        embedded_src = self.embedding(src)  # (batch, seq_len, embed_dim)
        _, (hidden, cell) = self.encoder(embedded_src)

        # Decoder
        embedded_trg = self.embedding(trg_input)  # (batch, seq_len, embed_dim)
        outputs, _ = self.decoder(embedded_trg, (hidden, cell))  # initialize decoder with encoder hidden state

        # Final output
        logits = self.fc(outputs)  # (batch, seq_len, vocab_size)
        return logits


