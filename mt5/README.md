# Burmese Headline Generation - Fine-tuning Pipeline

Fine-tune mT5 model for generating Burmese headlines from article text.

## 📋 Overview

This project provides a complete pipeline for:
- Fine-tuning multilingual T5 (mT5) models on Burmese text
- Generating headlines from news articles
- Evaluating model performance with ROUGE metrics
- Batch processing and interactive inference

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Your CSV file should have at least two columns:
- One with article text
- One with corresponding headlines

Example CSV structure:
```csv
article,headline
"သတင်းဆောင်းပါး၏ စာသားများ...",ခေါင်းစဉ်
"More article text...",Another headline
```

### 3. Train the Model

Edit the configuration in `train_burmese_headlines.py`:

```python
# Update these variables
CSV_PATH = "your_data.csv"  # Path to your CSV file
TEXT_COLUMN = "article"     # Column with article text
HEADLINE_COLUMN = "headline"  # Column with headlines
MODEL_NAME = "google/mt5-small"  # or "google/mt5-base" for better quality
```

Then run:
```bash
python train_burmese_headlines.py
```

### 4. Generate Headlines

```bash
# Interactive mode
python inference.py --interactive

# Or use in your code
python inference.py
```

## 📊 Model Options

### mT5 Model Variants

| Model | Parameters | Quality | Speed | GPU Memory |
|-------|-----------|---------|-------|------------|
| `mt5-small` | 300M | Good | Fast | ~4GB |
| `mt5-base` | 580M | Better | Medium | ~8GB |
| `mt5-large` | 1.2B | Best | Slow | ~16GB |

**Recommendation**: Start with `mt5-small` for 15K samples. Upgrade to `mt5-base` if quality isn't sufficient.

## ⚙️ Training Configuration

### Key Hyperparameters

```python
trainer_obj.train(
    train_dataset,
    val_dataset,
    num_epochs=3,              # Number of training epochs
    batch_size=8,              # Batch size (reduce if OOM)
    learning_rate=5e-5,        # Learning rate
    save_steps=1000,           # Save checkpoint every N steps
    eval_steps=1000,           # Evaluate every N steps
    gradient_accumulation_steps=1  # Increase if batch_size too small
)
```

### GPU Memory Optimization

If you run out of GPU memory:

1. **Reduce batch size**:
   ```python
   batch_size=4  # or even 2
   ```

2. **Use gradient accumulation**:
   ```python
   batch_size=4
   gradient_accumulation_steps=2  # Effective batch size = 4 * 2 = 8
   ```

3. **Use a smaller model**:
   ```python
   MODEL_NAME = "google/mt5-small"
   ```

## 📈 Evaluation Metrics

The model is evaluated using ROUGE scores:

- **ROUGE-1**: Unigram overlap (individual word matching)
- **ROUGE-2**: Bigram overlap (two consecutive words)
- **ROUGE-L**: Longest common subsequence

Higher scores = better quality (0-1 scale)

Typical scores for headline generation:
- Good: ROUGE-L > 0.3
- Very Good: ROUGE-L > 0.4
- Excellent: ROUGE-L > 0.5

## 💻 Usage Examples

### Training

```python
from train_burmese_headlines import BurmeseHeadlineTrainer

# Initialize
trainer = BurmeseHeadlineTrainer(
    model_name="google/mt5-small",
    max_input_length=512,
    max_target_length=128
)

# Load and prepare data
df, text_col, headline_col = trainer.load_data("data.csv")
train_df, val_df, test_df = trainer.split_data(df)

# Preprocess
train_dataset = trainer.preprocess_data(train_df, text_col, headline_col)
val_dataset = trainer.preprocess_data(val_df, text_col, headline_col)

# Train
trainer.train(train_dataset, val_dataset, num_epochs=3)
```

### Inference

```python
from inference import HeadlineGenerator

# Load model
generator = HeadlineGenerator("./burmese_headline_model")

# Generate single headline
article = "Your Burmese article text here..."
headline = generator.generate_headline(article)

# Generate multiple variants
headlines = generator.generate_headline(
    article,
    num_return_sequences=3,
    num_beams=5
)

# Batch processing from CSV
generator.generate_from_csv(
    input_csv="articles.csv",
    output_csv="with_headlines.csv",
    text_column="article"
)
```

## 🔧 Advanced Configuration

### Custom Data Preprocessing

```python
# If your data needs cleaning
def clean_text(text):
    # Remove extra whitespace
    text = " ".join(text.split())
    # Add your custom cleaning here
    return text

df['article'] = df['article'].apply(clean_text)
df['headline'] = df['headline'].apply(clean_text)
```

### Generation Parameters

```python
headline = generator.generate_headline(
    article,
    num_beams=4,        # Beam search width (1-10)
    max_length=128,     # Max headline length
    min_length=10,      # Min headline length
    temperature=1.0,    # Sampling temperature (0.5-1.5)
    num_return_sequences=1  # Number of variants
)
```

## 📁 Output Structure

After training, you'll have:

```
burmese_headline_model/
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights
├── tokenizer_config.json    # Tokenizer config
├── spiece.model            # SentencePiece model
├── test_metrics.json       # Evaluation results
└── checkpoint-*/           # Training checkpoints
```

## 🎯 Best Practices

### Data Quality
- **Clean your data**: Remove duplicates, empty entries, and malformed text
- **Balance length**: Very short or very long articles may hurt performance
- **Diverse examples**: Include various news categories and writing styles

### Training Tips
1. **Start small**: Train on 1000 samples first to verify everything works
2. **Monitor metrics**: Watch validation loss - if it stops improving, training is done
3. **Early stopping**: The model automatically saves the best checkpoint
4. **Experiment**: Try different learning rates (1e-5 to 1e-4) and epochs (2-5)

### Inference Tips
1. **Beam search**: Higher beams (4-8) give better quality but are slower
2. **Multiple variants**: Generate 3-5 variants and pick the best
3. **Length control**: Adjust `max_length` based on your headline style

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```python
# Solution: Reduce batch size
batch_size=2
gradient_accumulation_steps=4
```

**Slow training**
```python
# Use smaller model or reduce data
MODEL_NAME = "google/mt5-small"
# Or sample your data
df = df.sample(n=10000)
```

**Poor quality headlines**
- Try `mt5-base` instead of `mt5-small`
- Increase training epochs to 5
- Check data quality - ensure headlines are actually good summaries

**Unicode/encoding errors**
```python
# When reading CSV
df = pd.read_csv("data.csv", encoding='utf-8')

# When saving
df.to_csv("output.csv", encoding='utf-8', index=False)
```

## 📚 Additional Resources

- [mT5 Paper](https://arxiv.org/abs/2010.11934)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [ROUGE Metric](https://huggingface.co/spaces/evaluate-metric/rouge)

## 💡 Tips for 15K Dataset

With 15,000 samples:
- **Train/Val/Test split**: 12,000 / 1,500 / 1,500 (80/10/10)
- **Training time**: 
  - mt5-small: ~2-4 hours on GPU
  - mt5-base: ~6-10 hours on GPU
- **Expected quality**: Should achieve good ROUGE scores (>0.35 ROUGE-L)
- **Recommendations**:
  - Start with mt5-small to establish baseline
  - If quality is insufficient, upgrade to mt5-base
  - 3 epochs is usually sufficient

## 🤝 Next Steps

1. **Fine-tune on your data**
2. **Evaluate on test set**
3. **Test interactively** to validate quality
4. **Deploy** using inference script
5. **Iterate** - improve based on errors you observe

Good luck with your Burmese headline generation project! 🎉
