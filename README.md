# Burmese News Headline Generator

A collection of models, training scripts, and apps for generating Burmese (Myanmar) news headlines from article text. This repository includes experiments with LSTM-based sequence models, transformer-based approaches (mBART / mT5), utilities for training / evaluation, and Gradio demo apps for quick inference.

## Features
- Pre-built Gradio demo interfaces for quick inference (`app/`).
- Training and inference scripts for mT5 and other models (`mt5/`).
- LSTM experiments and notebooks in `train/` and `train/*.ipynb`.
- Utilities for data validation and evaluation.

## Repo structure

- `app/` — Gradio apps and model utilities:
	- `gradio_app.py` — main Gradio demo (generic)
	- `gradio_app_mbart.py` — demo for mBART-based model
	- `gradio_app_lstm.py` — demo for LSTM-based headline generator
	- `model_utils.py` — helper functions used by apps
- `mt5/` — mT5 training, inference and evaluation scripts
	- `train_burmese_headlines.py` — training script
	- `inference.py` — inference helper for mt5 models
	- `evaluate_model.py` — evaluation utilities
- `mmgpt/` — experiments and notebooks for MyanmarGPT-style models
- `train/` — training notebooks and diagnostic scripts (LSTM experiments, fixes)
- `dict-words.txt`, `stopwords.txt` — auxiliary data used by preprocessing
- `requirements.txt` — top-level Python dependencies for demos and utilities

## Requirements
- Python 3.8 or later
- Recommended: a virtual environment (venv / conda)
- Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# If you plan to train or run mt5 scripts, see mt5/requirements.txt
```

## Quickstart — Run Gradio demo

Run the simple demo 

```bash
# generic demo
python app/gradio_app_main.py
```

Open the displayed local URL to try the interface.

## Inference and Training

- Inference for mT5: see `mt5/inference.py` for helper functions and usage patterns.
- Train or fine-tune mT5 with `mt5/train_burmese_headlines.py`. Large models require GPUs and appropriate environment setup.
- LSTM experiments and small-scale training are available in the `train/` notebooks.

## Evaluation & Validation

- Use `mt5/evaluate_model.py` to compute evaluation metrics for generated headlines.
- Use `mt5/validate_data.py` (or `train/validate_data.py` where provided) to validate dataset formatting and tokenization.

## Data

This repository does not include large model checkpoints or proprietary datasets. Small helper files included:
- `dict-words.txt` — word list used in preprocessing
- `stopwords.txt` — Burmese stopwords used for cleaning

If you have trained checkpoints, place them in a clear path and update the app or inference script config to point to the model directory.

## Notebooks

There are several exploratory notebooks across `mt5/`, `mmgpt/`, and `train/` demonstrating training runs, data diagnostics, and inference examples. These are useful for reproducing experiments and understanding preprocessing choices.


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

