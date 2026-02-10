import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
output_dir = "burmese-headline-model-final/"
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
print("Model loaded successfully!")
