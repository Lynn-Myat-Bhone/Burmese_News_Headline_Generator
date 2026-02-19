"""
Gradio Web Interface for Burmese Headline Generator
Combined: mT5, mBart-50, and BiLSTM models
"""

import gradio as gr
import torch
import pickle
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import model architecture and utilities
from model_utils import (
    MyanmarTextPreprocessor,
    BiLSTMSeq2SeqWithAttention,
    generate_headline
)

# ──────────────────────────────────────────────
# mT5 Configuration
# ──────────────────────────────────────────────
MT5_MODEL_PATH = "model_v2"

# ──────────────────────────────────────────────
# mBart Configuration
# ──────────────────────────────────────────────
MBART_MODEL_PATH = "model_v3"

# ──────────────────────────────────────────────
# LSTM Configuration
# ──────────────────────────────────────────────
MAX_TEXT_LEN = 256
MAX_HEAD_LEN = 20
LSTM_MODEL_PATH = "seq2seq_headline_model/seq2seq_model_improved.pth"
VOCAB_PATH = "seq2seq_headline_model/vocab_improved.pkl"

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# Load mT5
# ──────────────────────────────────────────────
mt5_tokenizer = AutoTokenizer.from_pretrained(MT5_MODEL_PATH, local_files_only=True)
mt5_model = AutoModelForSeq2SeqLM.from_pretrained(MT5_MODEL_PATH, local_files_only=True)
mt5_model.to(device)
mt5_model.eval()
print(f"mT5 model loaded on {device}")

# ──────────────────────────────────────────────
# Load mBart
# ──────────────────────────────────────────────
mbart_tokenizer = AutoTokenizer.from_pretrained(MBART_MODEL_PATH, local_files_only=True)
mbart_model = AutoModelForSeq2SeqLM.from_pretrained(MBART_MODEL_PATH, local_files_only=True)
mbart_model.to(device)
mbart_model.eval()
print(f"mBart model loaded on {device}")

# ──────────────────────────────────────────────
# Load LSTM
# ──────────────────────────────────────────────
processor = MyanmarTextPreprocessor()
lstm_model = None
word2idx = None
idx2word = None


def _infer_num_layers(state_dict, prefix):
    layer_indices = []
    for key in state_dict.keys():
        if key.startswith(prefix) and key.endswith(".weight_ih_l0"):
            layer_indices.append(0)
        elif key.startswith(prefix) and ".weight_ih_l" in key and "_reverse" not in key:
            try:
                layer_idx = int(key.split(".weight_ih_l")[1].split(".")[0])
                layer_indices.append(layer_idx)
            except ValueError:
                continue
    return (max(layer_indices) + 1) if layer_indices else 1


def load_model_and_vocab(model_path, vocab_path):
    """Load trained model and vocabulary"""
    # Load vocabulary
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    # Handle different vocabulary file formats
    if isinstance(vocab_data, dict):
        if "word2idx" in vocab_data and "idx2word" in vocab_data:
            word2idx = vocab_data["word2idx"]
            idx2word = vocab_data["idx2word"]
            vocab_size = vocab_data.get("vocab_size", len(word2idx))
        else:
            raise ValueError("Vocabulary file missing 'word2idx' or 'idx2word' keys")
    else:
        raise ValueError("Vocabulary file should contain a dictionary")

    print(f"✔ Loaded vocabulary: {vocab_size} words")
    print(f"✔ Special tokens: {[k for k in word2idx.keys() if k.startswith('<')]}")

    # Load weights first to infer architecture
    state_dict = torch.load(model_path, map_location=DEVICE)
    if "encoder.weight_ih_l0" not in state_dict:
        raise ValueError("Checkpoint missing encoder.weight_ih_l0; cannot infer model dimensions")

    embedding_dim = state_dict["encoder.weight_ih_l0"].shape[1]
    hidden_dim = state_dict["encoder.weight_ih_l0"].shape[0] // 4
    num_layers = _infer_num_layers(state_dict, "encoder")

    # Create model
    pad_idx = word2idx["<pad>"]
    model = BiLSTMSeq2SeqWithAttention(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_idx=pad_idx
    ).to(DEVICE)

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    print("✔ Model loaded successfully!")

    return model, word2idx, idx2word


print("\n" + "="*60)
print("LOADING LSTM MODEL...")
print("="*60)
try:
    if not Path(LSTM_MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found: {LSTM_MODEL_PATH}")
    if not Path(VOCAB_PATH).exists():
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")

    lstm_model, word2idx, idx2word = load_model_and_vocab(LSTM_MODEL_PATH, VOCAB_PATH)
    print("="*60)
    print("✔ LSTM MODEL READY!")
    print("="*60 + "\n")
except Exception as e:
    print(f"✘ Failed to load LSTM model: {e}")
    print("Please check that the model files exist and paths are correct.")
    exit(1)

# ──────────────────────────────────────────────
# Inference functions
# ──────────────────────────────────────────────

def generate_headline_mt5(article_text):
    """Generate a single headline with default parameters"""
    if not article_text.strip():
        return "Please enter article text."

    input_text = "summarize: " + article_text

    inputs = mt5_tokenizer(
        input_text,
        max_length=64,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = mt5_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            length_penalty=1.0,
            temperature=0.7
        )

    headline = mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return headline


def generate_headline_mbart(article_text):
    """Generate a single headline with default parameters"""
    if not article_text.strip():
        return "Please enter article text."

    input_text = "summarize: " + article_text

    inputs = mbart_tokenizer(
        input_text,
        max_length=64,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = mbart_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            length_penalty=1.0,
            temperature=0.7
        )

    headline = mbart_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return headline


def predict_lstm(text):
    """Gradio prediction function for LSTM"""
    if not text.strip():
        return "⚠️ Please enter some text!"

    try:
        headline = generate_headline(
            model=lstm_model,
            processor=processor,
            text=text,
            word2idx=word2idx,
            idx2word=idx2word,
            max_text_len=MAX_TEXT_LEN,
            max_head_len=MAX_HEAD_LEN,
            device=DEVICE
        )

        if not headline:
            return "⚠️ Could not generate headline (empty output)"

        return headline

    except Exception as e:
        return f"✘ Error: {str(e)}"


# ──────────────────────────────────────────────
# Gradio Interface
# ──────────────────────────────────────────────

with gr.Blocks(title="Burmese Headline Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🇲🇲 Burmese Headline Generator
        Generate Burmese news headlines from article text using three different models.
        """
    )

    with gr.Tabs():

        # ── Tab 1: mT5 ──────────────────────────────
        with gr.Tab("mT5 Model"):
            gr.Markdown("Generate Burmese news headlines from article text using your fine-tuned mT5 model.")
            with gr.Row():
                with gr.Column(scale=2):
                    mt5_article_input = gr.Textbox(
                        label="Article Text (Burmese)",
                        placeholder="Enter Burmese article text here...",
                        lines=12
                    )
                    mt5_generate_btn = gr.Button("Generate Headline", variant="primary", size="lg")

                    gr.Markdown(
                        """
                        **Model Info & Parameters**  
                        - Model: `model_v2` (fine-tuned google/mt5-small)  
                        - Task: Burmese headline generation  
                        - Training data: ~17,000 Burmese news articles  
                        - Trainable parameters: 556M (full model)
                        """
                    )

                with gr.Column(scale=1):
                    mt5_output = gr.Textbox(
                        label="Generated Headline",
                        lines=12
                    )

            mt5_generate_btn.click(fn=generate_headline_mt5, inputs=mt5_article_input, outputs=mt5_output)

        # ── Tab 2: mBart ────────────────────────────
        with gr.Tab("mBart-50 Model"):
            gr.Markdown("Generate Burmese news headlines from article text using your fine-tuned mBart-50 model.")
            with gr.Row():
                with gr.Column(scale=2):
                    mbart_article_input = gr.Textbox(
                        label="Article Text (Burmese)",
                        placeholder="Enter Burmese article text here...",
                        lines=12
                    )
                    mbart_generate_btn = gr.Button("Generate Headline", variant="primary", size="lg")

                    gr.Markdown(
                        """
                        **Model Info & Parameters**  
                        - Model: `model_v3` (fine-tuned facebook/mbart-large-50)  
                        - Task: Burmese headline generation  
                        - Training data: ~17,000 Burmese news articles  
                        - Trainable parameters: 556M (full model)
                        """
                    )

                with gr.Column(scale=1):
                    mbart_output = gr.Textbox(
                        label="Generated Headline",
                        lines=12
                    )

            mbart_generate_btn.click(fn=generate_headline_mbart, inputs=mbart_article_input, outputs=mbart_output)

        # ── Tab 3: LSTM ─────────────────────────────
        with gr.Tab("BiLSTM Model"):
            gr.Markdown("""
            ### LSTM-based Neural Headline Generation for Myanmar News

            This model uses a BiLSTM Seq2Seq architecture with attention mechanism
            to generate Myanmar language headlines from article text.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Input Article")
                    lstm_text_input = gr.Textbox(
                        label="Article Text (Myanmar)",
                        placeholder="Enter Myanmar article text here...",
                        lines=10,
                        max_lines=15
                    )

                    with gr.Row():
                        lstm_clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
                        lstm_submit_btn = gr.Button("✨ Generate Headline", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### 📰 Generated Headline")
                    lstm_output = gr.Textbox(
                        label="Generated Headline",
                        lines=6,
                        interactive=False,
                    )

            gr.Markdown("---")

            with gr.Accordion("📊 Model Information", open=False):
                gr.Markdown(f"""
                ### Architecture Details
                - **Model Type:** BiLSTM Seq2Seq with Attention Mechanism
                - **Tokenization:** Syllable-level (Myanmar script)
                - **Embedding Dimension:** {lstm_model.embedding_dim}
                - **Hidden Units:** {lstm_model.hidden_dim}
                - **LSTM Layers:** {lstm_model.num_layers} (Bidirectional Encoder + Unidirectional Decoder)
                - **Max Input Length:** 256 syllables
                - **Max Output Length:** 20 syllables
                - **Device:** {DEVICE}
                - **Model File:** {LSTM_MODEL_PATH}
                - **Vocab File:** {VOCAB_PATH}
                """)

            lstm_submit_btn.click(
                fn=predict_lstm,
                inputs=[lstm_text_input],
                outputs=lstm_output
            )

            lstm_clear_btn.click(
                fn=lambda: ("", ""),
                outputs=[lstm_text_input, lstm_output]
            )

    gr.Markdown("""
    ---
    <center>
    <small>Built with Gradio | Myanmar NLP Project</small>
    </center>
    """)

if __name__ == "__main__":
    demo.launch(share=False)