# Burmese News Headline Generator

Transforming Burmese news articles into concise, meaningful headlines using deep learning models including BiLSTM, mT5, and mBART-50.

---

## Overview

This project builds and compares multiple NLP architectures for **Burmese news headline generation**, a sequence-to-sequence text generation task.

It focuses on:
- Low-resource language NLP (Burmese)
- Model comparison (LSTM vs Transformers)
- Full pipeline: training → evaluation → inference → demo

---

## Live Demo

### Gradio Interface

![Demo UI](assets/demo.png)

---

## Problem Statement

Generating accurate and concise headlines from Burmese news articles is challenging due to:

- Low-resource language limitations
- Complex sentence structures
- Lack of large-scale labeled datasets
- Need for semantic compression of long text

---

## Dataset

- **Source**: Kaggle (Burmese News Dataset - DVB, 2019–2024)
- **Total Articles**: 15,899
- **Train / Validation Split**: 80 / 20
- **Test Set**: 3,000 samples

### Preprocessing

- Removed duplicate articles  
- Removed HTML tags  
- Normalized Unicode text  
- Filtered very short articles  

---

## Models

### mT5-small
- Multilingual transformer model  
- Balanced performance and efficiency  
- Best trade-off model  

---

### mBART-50
- Large multilingual transformer  
- Highest semantic quality  
- Best overall performance  

---

### BiLSTM (Baseline)
- Sequence-to-sequence LSTM model  
- Fast inference  
- Weak semantic understanding (baseline)

---

## Results

| Model     | ROUGE-L | BLEU-4 | BERTScore F1 | Quality |
|----------|--------|--------|--------------|---------|
| BiLSTM   | 0.76   | 21.95  | 71.96        | Low     |
| mT5      | 5.38   | 64.04  | 85.81        | High    |
| mBART-50 | 3.48   | 49.85  | 81.24        | Highest |

---

##  Key Insights

- Transformer models significantly outperform BiLSTM baseline  
- mBART-50 achieves the highest semantic similarity  
- mT5 provides the best balance between performance and efficiency  
- ROUGE scores are lower due to abstractive generation nature  

---
## Project Structure
app/        → Gradio demo UI  
mt5/        → mT5 training, inference, evaluation  
train/      → LSTM experiments  
mmgpt/      → experimental models  
assets/     → screenshots for README  

---

## How to Run

```bash
git clone https://github.com/your-username/Burmese_News_Headline_Generator.git
cd Burmese_News_Headline_Generator

pip install -r requirements.txt

python app/gradio_app_main.py
```
---

## Tech Stack
- Python
- PyTorch / HuggingFace Transformers
- mT5, mBART-50
- FastText (for LSTM embeddings)
- Gradio (UI)

---
## Key Contributions
- Built and compared 3 NLP architectures
- Designed full training and evaluation pipeline
- Conducted benchmarking for Burmese NLP task
- Developed interactive Gradio demo

---
## Future Improvements
- Improve Burmese tokenization techniques
- Add reinforcement learning for better headline quality
- Deploy as web API service
- Add human evaluation study

---
## Authors
- Lynn Myat Bhone
- Kyaw Thuta Oo
- Pyae Phyo Maung
- Htet Aung Hlyan
