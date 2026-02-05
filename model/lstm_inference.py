# model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import re

# ----------------------------
# Myanmar Text Preprocessor
# ----------------------------
class MyanmarTextPreprocessor:
    def __init__(self, dict_path: str, stop_path: str):
        self.dictionary = self.load_dictionary(dict_path)
        self.stopwords = self.load_stopwords(stop_path)
        self.pattern = re.compile(
            r"(([A-Za-z0-9]+)|[က-အဥဦ](င်္|[က-အ][ှ]*[့း]*[်]?|္[က-အ]|[ါ-ှ]*)*|.)"
        )

    def load_dictionary(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}

    def load_stopwords(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}

    def merge_with_dictionary(self, syllables):
        merged_tokens = []
        i = 0
        n = len(syllables)
        while i < n:
            matched = False
            for j in range(n, i, -1):
                combined = ''.join(syllables[i:j])
                if combined in self.dictionary:
                    merged_tokens.append(combined)
                    i = j
                    matched = True
                    break
            if not matched:
                merged_tokens.append(syllables[i])
                i += 1
        return merged_tokens

    def preprocessing(self, text: str):
        # Split Burmese syllables and merge dictionary words
        text = self.pattern.sub(r"\1 ", text)
        syllables = text.strip().split()
        merged_tokens = self.merge_with_dictionary(syllables)
        # ⚠️ Keep stopwords for generation
        return ' '.join(merged_tokens)

    def tokenize(self, text: str):
        return text.split()


# ----------------------------
# Sequence encoding
# ----------------------------
def encode_sentence(sentence, word2idx, max_len, add_sos_eos=False, tokenizer=None):
    if tokenizer is None:
        tokenizer = lambda x: x.split()
    tokens = tokenizer(sentence)
    ids = []

    if add_sos_eos:
        ids.append(word2idx["<sos>"])

    for tok in tokens[:max_len]:
        ids.append(word2idx.get(tok, word2idx["<unk>"]))

    if add_sos_eos:
        ids.append(word2idx["<eos>"])

    # Pad / truncate
    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


# ----------------------------
# Dataset class
# ----------------------------
class HeadlineDataset(Dataset):
    def __init__(self, texts, headlines, word2idx, max_text_len=200, max_head_len=20, tokenizer=None):
        self.texts = texts
        self.headlines = headlines
        self.word2idx = word2idx
        self.max_text_len = max_text_len
        self.max_head_len = max_head_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        src = torch.tensor(
            encode_sentence(self.texts[idx], self.word2idx, self.max_text_len, add_sos_eos=False, tokenizer=self.tokenizer),
            dtype=torch.long
        )
        trg = torch.tensor(
            encode_sentence(self.headlines[idx], self.word2idx, self.max_head_len, add_sos_eos=True, tokenizer=self.tokenizer),
            dtype=torch.long
        )
        decoder_input = trg[:-1]
        decoder_target = trg[1:]
        return src, decoder_input, decoder_target


# ----------------------------
# Seq2Seq LSTM Model
# ----------------------------
class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, embedding_matrix=None, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = True  # fine-tune embeddings

        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, trg_input):
        # Encoder
        embedded_src = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded_src)

        # Decoder
        embedded_trg = self.embedding(trg_input)
        outputs, _ = self.decoder(embedded_trg, (hidden, cell))

        # Predict
        logits = self.fc(outputs)
        return logits


# ----------------------------
# Headline generation
# ----------------------------
def generate_headline(model, text, word2idx, idx2word, processor, max_len=20, temperature=1.0, device='cpu'):
    model.eval()
    with torch.no_grad():
        # Preprocess input
        text_proc = processor.preprocessing(text)
        src_ids = torch.tensor(encode_sentence(text_proc, word2idx, max_len=200, add_sos_eos=False, tokenizer=processor.tokenize)).unsqueeze(0).to(device)

        # Encoder
        embedded_src = model.embedding(src_ids)
        _, (hidden, cell) = model.encoder(embedded_src)

        # Decoder
        input_token = torch.tensor([[word2idx["<sos>"]]], dtype=torch.long).to(device)
        generated_ids = []

        for _ in range(max_len):
            embedded = model.embedding(input_token)
            output, (hidden, cell) = model.decoder(embedded, (hidden, cell))
            logits = model.fc(output[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1).item()

            if next_id == word2idx["<eos>"]:
                break

            generated_ids.append(next_id)
            input_token = torch.tensor([[next_id]], dtype=torch.long).to(device)

        return " ".join([idx2word[i] for i in generated_ids])
