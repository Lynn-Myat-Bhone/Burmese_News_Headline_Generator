"""
Gradio Web Interface for Burmese Headline Generator
2-column layout with info panel
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration
MODEL_PATH = "model_v2"

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)
model.to(device)
model.eval()
print(f"Model loaded on {device}")


def generate_headline(article_text):
    """Generate a single headline with default parameters"""
    if not article_text.strip():
        return "Please enter article text."

    input_text = "summarize: " + article_text

    # Tokenize and move to device
    inputs = tokenizer(
        input_text,
        max_length=64,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate headline
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            length_penalty=1.0,
            temperature=0.7
        )

    headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return headline


# Gradio interface
with gr.Blocks(title="Burmese Headline Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🇲🇲 Burmese Headline Generator
        Generate Burmese news headlines from article text using your fine-tuned mT5 model.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            article_input = gr.Textbox(
                label="Article Text (Burmese)",
                placeholder="Enter Burmese article text here...",
                lines=12
            )
            generate_btn = gr.Button("Generate Headline", variant="primary", size="lg")
            
            # Info panel at bottom left
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
            output = gr.Textbox(
                label="Generated Headline",
                lines=12
            )

    # Connect button
    generate_btn.click(fn=generate_headline, inputs=article_input, outputs=output)

if __name__ == "__main__":
    demo.launch(share=False)
