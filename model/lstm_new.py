import torch
import pickle
from model.lstm_inference import MyanmarTextPreprocessor, Seq2SeqLSTM, generate_headline

# 1️⃣ Load vocab
with open("vocab.pkl", "rb") as f:
    vocab_data = pickle.load(f)
word2idx = vocab_data["word2idx"]
idx2word = vocab_data["idx2word"]

# 2️⃣ Load processor
with open("processor.pkl", "rb") as f:
    proc_info = pickle.load(f)
processor = MyanmarTextPreprocessor(proc_info["dict_path"], proc_info["stopwords_path"])

# 3️⃣ Load embedding matrix
embedding_matrix = torch.load("embedding_matrix.pt")

# 4️⃣ Initialize model
vocab_size = len(word2idx)
model = Seq2SeqLSTM(vocab_size, embedding_dim=300, hidden_dim=256, embedding_matrix=embedding_matrix)
model.load_state_dict(torch.load("seq2seq_model.pth"))
model.eval()

# 5️⃣ Generate headline
article = "မိုးသည်းထန်စွာ ရွာသွန်းနေသော မြို့တော်တွင် လျှပ်စစ်မီးဖြတ်ခြင်းဖြစ်ပေါ်"
headline = generate_headline(model, article, word2idx, idx2word, processor)
print("Generated Headline:", headline)
