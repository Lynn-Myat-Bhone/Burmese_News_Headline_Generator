import torch
import json
import numpy as np
from gensim.models import KeyedVectors
from lstmModel import Encoder, Decoder, Seq2Seq
import os
from pathlib import Path

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent
UTILITIES_DIR = SCRIPT_DIR.parent / "utilities"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
# Load vocab
# ------------------
with open(UTILITIES_DIR / "headline_vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

word2idx = vocab_data["word2idx"]
idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
vocab_size = len(word2idx)


embedding_dim = 300
hidden_dim = 256
MAX_TEXT_LEN = 200
MAX_HEAD_LEN = 20

# ------------------
# Load fastText again
# ------------------
print("Loading fastText...")
ft = KeyedVectors.load_word2vec_format(str(UTILITIES_DIR / "cc.my.300.vec"))

embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
for word, idx in word2idx.items():
    if word in ft:
        embedding_matrix[idx] = ft[word]

embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

# ------------------
# Rebuild model
# ------------------
enc = Encoder(vocab_size, embedding_dim, hidden_dim, embedding_matrix)
dec = Decoder(vocab_size, embedding_dim, hidden_dim, embedding_matrix)
model = Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load(str(UTILITIES_DIR / "headline_lstm_fasttext.pt"), map_location=device))
model.eval()
print("✅ Model loaded!")

# ---------------------------
# Tokenization & Encoding
# ---------------------------
def tokenize(text):
    return text.split()

def encode_sentence(sentence, max_len):
    tokens = tokenize(sentence)
    ids = [word2idx.get(tok, word2idx["<unk>"]) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids).unsqueeze(0).to(device)

# ---------------------------
# Headline Generation
# ---------------------------
def generate_headline(text):
    src = encode_sentence(text, MAX_TEXT_LEN)

    with torch.no_grad():
        hidden, cell = model.encoder(src)

    input_token = torch.tensor([word2idx["<sos>"]]).to(device)
    result = []
    used_tokens = set()

    for _ in range(MAX_HEAD_LEN):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell)

        pred_token = output.argmax(1).item()

        if pred_token == word2idx["<eos>"]:
            break
        if pred_token in used_tokens:
            break  # prevent repetition loop

        used_tokens.add(pred_token)
        result.append(idx2word[pred_token])
        input_token = torch.tensor([pred_token]).to(device)

    return " ".join(result)

# ---------------------------
# Run Example
# ---------------------------
if __name__ == "__main__":
    news = """
‌မလေးရှားနိုင်ငံ၊ ဂျိုဟိုးပြည်နယ်မှာ တရားမဝင် အလုပ်လုပ်ကိုင်နေတဲ့ မြန်မာနိုင်ငံသား ၈ ဦး အပါအဝင် နိုင်ငံခြားသား ၁၆ ဦးကို ဇန်နဝါရီ ၁၀ ရက်နေ့မှာ ဖမ်းဆီးခဲ့တယ်လို့ မလေးရှား လူဝင်မှုကြီးကြပ်ရေးဌာနက ထုတ်ပြန်ပါတယ်။ ဂျိုးဟိုးပြည်နယ် ဘာတူပါဟပ်ဒေသရှိ ဈေးဆိုင်တွေမှာ နိုင်ငံခြားသားတွေ လုပ်ကိုင်နေတဲ့ သတင်းရလို့ မလေးလူဝင်မှုကြီးကြပ်ရေး အရာရှိတွေနဲ့ ဘာတူပါဟပ် လူဝင်မှုကြီး‌ကြပ်‌ရေးဌာနခွဲ အရာရှိတွေ ပူးပေါင်းပြီး ဇန်နဝါရီ ၁၀ ရက်က စစ်ဆင်ရေးလုပ်ခဲ့ရာ အထောက်အထား စာရွက်စာတမ်း မပါရှိတဲ့ မြန်မာနိုင်ငံသား အမျိုးသား ၄ ဦးနဲ့ အမျိုးသမီး ၄ ဦး၊ ပါကစ္စတန်နိုင်ငံသား ၄ ဦး၊ အင်ဒိုနီးရှားနိုင်ငံသား ၂ ဦးနဲ့ ဗီယက်နမ်နိုင်ငံသား ၂ ဦးတို့ကို ဖမ်းဆီးခဲ့ပါတယ်။ မလေးရောက် မြန်မာနိုင်ငံသားတဦးက “မလေးမှာ ဆက်တိုက်ကို ဖမ်းနေတာ၊ ကျနော်တို့တောင် မနည်းရှောင်တိမ်းပြီး နေနေရတယ်။ ဒီလ လတဝက်တောင် မကျိုးသေးဘူး၊ မြန်မာ ၃၀၀ နီးပါးလောက် ဖမ်းခံထားရပြီ။ အခုလက်ရှိကတော့ ကျနော်တို့ အဖမ်းခံရပြီး မြန်မာပြည်ပြန်ပို့တာတွေ ဖြစ်မှာစိုးလို့ UN ကတ်လျှောက်နေပါတယ်” လို့ ပြောပါတယ်။ ဖမ်းဆီးရမိခဲ့တဲ့ မြန်မာနိုင်ငံသား ၈ ဦး အပါအဝင် နိုင်ငံခြားသား ၁၆ ဦးတို့ကို အထောက်‌အထားမရှိဘဲ တရားမဝင် နေထိုင်မှုအတွက် ၁၉၅၃/၆၃ လူဝင်မှုကြီးကြပ်ရေး ဥပဒေနဲ့ အရေးယူသွားမယ်လို့ မလေးလူဝင်မှုကြီးကြပ်ရေးဌာနက ထုတ်ပြန်ခဲ့ပါတယ်။ မလေးရှားနိုင်ငံ၊ ကီလန်တန်ပြည်နယ်မှာ ဇန်နဝါရီ ၆ ရက်ကလည်း တရားမဝင် နေထိုင်နေတဲ့ မြန်မာနိုင်ငံသား ၁၂၂ ယောက်ကို မလေးလူဝင်မှုကြီးကြပ်ရေးအဖွဲ့က ဖမ်းဆီးခဲ့ပါတယ်။ 
"""
    headline = generate_headline(news)
    print("\n📰 Generated Headline:")
    print(headline)
    print(word2idx["<unk>"])
    print(idx2word[1])
    print(vocab_size)
    print(word2idx.get("မြန်မာ", "not found"))