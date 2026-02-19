import gradio as gr
import torch
import pickle
from pathlib import Path

# Import model architecture and utilities
from model_utils import (
    MyanmarTextPreprocessor,
    BiLSTMSeq2SeqWithAttention,
    generate_headline
)

# Configuration
MAX_TEXT_LEN = 256
MAX_HEAD_LEN = 20

# Model file paths (change these to your file locations)
MODEL_PATH = "seq2seq_headline_model/seq2seq_model_improved.pth"  # Update this path
VOCAB_PATH = "seq2seq_headline_model/vocab_improved.pkl"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Initialize processor
processor = MyanmarTextPreprocessor()

# Global variables for model
model = None
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
        # Check if it's in the expected format
        if "word2idx" in vocab_data and "idx2word" in vocab_data:
            word2idx = vocab_data["word2idx"]
            idx2word = vocab_data["idx2word"]
            # vocab_size might be stored or we calculate it
            vocab_size = vocab_data.get("vocab_size", len(word2idx))
        else:
            raise ValueError("Vocabulary file missing 'word2idx' or 'idx2word' keys")
    else:
        raise ValueError("Vocabulary file should contain a dictionary")
    
    print(f"✓ Loaded vocabulary: {vocab_size} words")
    print(f"✓ Special tokens: {[k for k in word2idx.keys() if k.startswith('<')]}")
    
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
    
    print("✓ Model loaded successfully!")
    
    return model, word2idx, idx2word


def predict(text):
    """Gradio prediction function"""
    if not text.strip():
        return "⚠️ Please enter some text!"
    
    try:
        # Generate headline
        headline = generate_headline(
            model=model,
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
        return f"❌ Error: {str(e)}"


# Load model on startup
print("\n" + "="*60)
print("LOADING MODEL...")
print("="*60)
try:
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not Path(VOCAB_PATH).exists():
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")
    
    model, word2idx, idx2word = load_model_and_vocab(MODEL_PATH, VOCAB_PATH)
    print("="*60)
    print("✓ MODEL READY!")
    print("="*60 + "\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("Please check that the model files exist and paths are correct.")
    exit(1)


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Myanmar Headline Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🇲🇲 Myanmar Headline Generator
        ### LSTM-based Neural Headline Generation for Myanmar News
        
        This model uses a BiLSTM Seq2Seq architecture with attention mechanism 
        to generate Myanmar language headlines from article text.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Article")
                text_input = gr.Textbox(
                    label="Article Text (Myanmar)",
                    placeholder="Enter Myanmar article text here...",
                    lines=10,
                    max_lines=15
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
                    submit_btn = gr.Button("✨ Generate Headline", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📰 Generated Headline")
                output = gr.Textbox(
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
            - **Embedding Dimension:** {model.embedding_dim}
            - **Hidden Units:** {model.hidden_dim}
            - **LSTM Layers:** {model.num_layers} (Bidirectional Encoder + Unidirectional Decoder)
            - **Max Input Length:** 256 syllables
            - **Max Output Length:** 20 syllables
            - **Device:** {DEVICE}
            - **Model File:** {MODEL_PATH}
            - **Vocab File:** {VOCAB_PATH}
            
            """)
    
        
        # Event handlers
        submit_btn.click(
            fn=predict,
            inputs=[text_input],
            outputs=output
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[text_input, output]
        )
        
        
        gr.Markdown("""
        ---
        <center>
        <small>Built with using Gradio | Myanmar NLP Project</small>
        </center>
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
    )