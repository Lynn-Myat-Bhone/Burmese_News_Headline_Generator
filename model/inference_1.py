import torch
import json
import numpy as np
from gensim.models import KeyedVectors
from lstmModel import Encoder, Decoder, Seq2Seq
from pathlib import Path
import torch.nn.functional as F

# ------------------
# Paths & Device
# ------------------
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

pad_idx = word2idx["<pad>"]
unk_idx = word2idx["<unk>"]
sos_idx = word2idx["<sos>"]
eos_idx = word2idx["<eos>"]

vocab_size = len(word2idx)

# ------------------
# Model params
# ------------------
embedding_dim = 300
hidden_dim = 256
MAX_TEXT_LEN = 200
MAX_HEAD_LEN = 20
TEMPERATURE = 0.8

# ------------------
# Load fastText
# ------------------
print("Loading fastText...")
ft = KeyedVectors.load_word2vec_format(
    str(UTILITIES_DIR / "cc.my.300.vec")
)

embedding_matrix = np.random.normal(
    scale=0.6, size=(vocab_size, embedding_dim)
)

for word, idx in word2idx.items():
    if word in ft:
        embedding_matrix[idx] = ft[word]

embedding_matrix = torch.tensor(
    embedding_matrix, dtype=torch.float
)

# ------------------
# Build model
# ------------------
encoder = Encoder(vocab_size, embedding_dim, hidden_dim, embedding_matrix)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim, embedding_matrix)

model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(
    torch.load(
        UTILITIES_DIR / "headline_lstm_fasttext.pt",
        map_location=device
    )
)

model.eval()
print("✅ Model loaded")

# ------------------
# Tokenization (same as training)
# ------------------
def tokenize(text):
    return text.split()

def encode_sentence(sentence, max_len):
    tokens = tokenize(sentence)
    ids = [word2idx.get(tok, unk_idx) for tok in tokens[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return torch.tensor(ids).unsqueeze(0).to(device)

# ------------------
# Headline Generation
# ------------------
def generate_headline(text):
    src = encode_sentence(text, MAX_TEXT_LEN)

    with torch.no_grad():
        hidden, cell = model.encoder(src)

    # 🔥 Decoder expects (batch,), NOT (batch, 1)
    input_token = torch.tensor([sos_idx], device=device)
    result = []

    for _ in range(MAX_HEAD_LEN):
        with torch.no_grad():
            output, hidden, cell = model.decoder(
                input_token, hidden, cell
            )

        # output: (batch, vocab_size)
        probs = torch.softmax(output / TEMPERATURE, dim=1)
        pred_token = torch.multinomial(probs, 1).item()

        if pred_token == eos_idx:
            break
        if pred_token in (pad_idx, unk_idx):
            continue

        result.append(idx2word[pred_token])

        # 🔥 keep shape (batch,)
        input_token = torch.tensor([pred_token], device=device)

    return "".join(result)   # Burmese-friendly

# ------------------
# Run example
# ------------------
if __name__ == "__main__":
    news = """
    ထိုင်းအရှေ့တောင်ပိုင်း၊ ချွန်းပူးရီးခရိုင်၊ ဖနာ့ဆ်နိခုမ်းမြို့နယ်ထဲမှာ ဇန်နဝါရီ ၈ ရက် မနက်က မြန်မာအလုပ်သမ တယောက် စက္ကူစက် အကြိတ်ခံရပြီး သေဆုံးသွားတယ်လို့ Matichon ထိုင်းသတင်းမှာ ဖော်ပြပါတယ်။ ဖနာ့ဆ်နိခုမ်းမြို့ ခုတ်ဖော့ရပ်ကွက်  ဆာမ်ရန်းဖာနစ်ကုမ္ပဏီထဲမှာ လုပ်ငန်းခွင်မတော်တဆဖြစ်ပြီး သေဆုံးသူရှိကြောင်း အကြောင်းကြားချက်ရလို့ လူနာတင်ကားနဲ့ သွားရောက်စစ်ဆေး ကြည့်ရှုခဲ့တယ်လို့ ဖနာ့ဆ်နိခုမ်းနယ်မြေရဲစခန်း ရဲမှူး ဆာမတ်ဘွန်းနစ်က သတင်းထောက်တွေကို ပြောပါတယ်။ အခင်းဖြစ်တဲ့နေရာမှာ စက္ကူကြိတ်စက်ထဲ အမျိုးသမီးတယောက်ရဲ့ ခန္ဓာကိုယ်တစ်ခုလုံး အကြိတ်ခံထားရပြီး ကြေမွနေလို့ တစစီ ဆွဲထုတ်ကာ အဝတ်နဲ့ ထုတ်ခဲ့ရကြောင်း၊ သေဆုံးသူမှာ အသက် ၃၀ နှစ်ရွယ်ရှိ မနွေးလှိုင် (ထိုင်းအသံထွက် အနီးစပ်ဆုံးအမည်) ဆိုသူ မြန်မာအလုပ်သမလို့ ဆိုပါတယ်။ လုပ်ဖော်ကိုင်ဖက် အလုပ်သမားတယောက်ရဲ့ ပြောပြချက်အရ အခင်းဖြစ်တဲ့အချိန် လုပ်ငန်းခွင်မှာ အလုပ်သမား ၃ ယောက် ရှိကြောင်း၊ စက်ထဲ စက္ကူခံနေရင် မိန်းခလုတ်ကို ပိတ်ပြီး ပြန်ဆွဲထုတ်ရတယ်၊ သေဆုံးသူမှာ စက်ထဲ စက္ကူထည့်ရသူ ဖြစ်တယ်။ အလုပ်သမား ၂ ယောက် အောက်ထပ်မှာရှိနေစဉ် အပေါ်ထပ်မှာ စက်လည်သံ မကြားရတော့  စက်ချို့ယွင်းတယ် ထင်ပြီး လာကြည့်ရာမှာ မနွေးလှိုင်ရဲ့ ခန္ဓာကိုယ်ဟာ စက်ထဲ ကြိတ်မိနေပြီလို့ ပြောပါတယ်။ အလုပ်သမဟာ စက်ထဲကို ဘယ်လိုကျသွားလဲဆိုတာ မသိရပေမဲ့ စက်လည်နေစဉ် ရာသီဥတုပူပြင်းပြီး မိုက်ခနဲဖြစ်ကာ စက်ပေါ် မှောက်ကျသွားတာ ဖြစ်နိုင်တယ်လို့ အလုပ်သမား ခေါင်းဆောင်က သုံးသပ်ပါတယ်။ အဆိုပါအလုပ်သမရဲ့အလောင်းကို ဖနာ့ဆ်နိခုမ်းဆေးရုံကို ပို့ဆောင်ထားပြီး သက်ဆိုင်ရာဆွေမျိုးတွေကို ဆက်သွယ်ထားကြောင်း၊  ဖြစ်စဉ်အသေးစိတ်ကိုလည်း ဆက်လက်စုံစမ်းစစ်ဆေးသွားမယ်လို့ ရဲစခန်းက ပြောသွားပါတယ်။ ကိုးကား - matichon.co.th 

    """

    headline = generate_headline(news)

    print("\n📰 Generated Headline:")
    print(headline)
