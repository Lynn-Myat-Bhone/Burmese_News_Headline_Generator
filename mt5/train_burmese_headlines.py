"""
Burmese Headline Generation - Fine-tuning Script
Fine-tunes mT5 model for generating headlines from Burmese article text
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer
)
import evaluate
import torch
from pathlib import Path
import json

class BurmeseHeadlineTrainer:
    def __init__(
        self,
        model_name="google/mt5-base",  # Options: mt5-small, mt5-base, mt5-large
        max_input_length=512,
        max_target_length=128,
        output_dir="./burmese_headline_model"
    ):
        """
        Initialize the trainer
        
        Args:
            model_name: Pretrained model to use (mT5 supports Burmese well)
            max_input_length: Max tokens for article text
            max_target_length: Max tokens for headline
            output_dir: Where to save the fine-tuned model
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.output_dir = output_dir
        
        print(f"Loading model and tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        
        # Load evaluation metrics
        self.rouge = evaluate.load("rouge")
        
    def load_data(self, csv_path, text_column="article", headline_column="headline"):
        """
        Load data from CSV file
        
        Args:
            csv_path: Path to CSV file
            text_column: Name of column containing article text
            headline_column: Name of column containing headlines
        """
        print(f"\nLoading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Basic validation
        assert text_column in df.columns, f"Column '{text_column}' not found in CSV"
        assert headline_column in df.columns, f"Column '{headline_column}' not found in CSV"
        
        # Remove rows with missing values
        df = df.dropna(subset=[text_column, headline_column])
        
        # Convert to strings and strip whitespace
        df[text_column] = df[text_column].astype(str).str.strip()
        df[headline_column] = df[headline_column].astype(str).str.strip()
        
        # Remove empty entries
        df = df[(df[text_column] != "") & (df[headline_column] != "")]
        
        print(f"Loaded {len(df)} examples")
        print(f"\nSample data:")
        print(f"Article: {df[text_column].iloc[0][:100]}...")
        print(f"Headline: {df[headline_column].iloc[0]}")
        
        return df, text_column, headline_column
    
    def split_data(self, df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        """
        Split data into train/validation/test sets
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        print(f"\nData split:")
        print(f"  Train: {len(train_df)} examples")
        print(f"  Validation: {len(val_df)} examples")
        print(f"  Test: {len(test_df)} examples")
        
        return train_df, val_df, test_df
    
    def preprocess_data(self, df, text_column, headline_column):
        """
        Tokenize the data
        """
        def preprocess_function(examples):
            # Add task prefix for mT5
            inputs = ["summarize: " + text for text in examples[text_column]]
            targets = examples[headline_column]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_input_length,
                truncation=True,
                padding=False  # Padding will be done by data collator
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                targets,
                max_length=self.max_target_length,
                truncation=True,
                padding=False
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Convert DataFrame to Dataset
        dataset = Dataset.from_pandas(df[[text_column, headline_column]])
        
        # Tokenize
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """
        Compute ROUGE scores for evaluation
        """
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=False  # Burmese doesn't use stemming
        )
        
        # Extract key metrics
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
        }
    
    def train(
        self,
        train_dataset,
        val_dataset,
        num_epochs=3,
        batch_size=8,
        learning_rate=5e-5,
        warmup_steps=500,
        save_steps=1000,
        eval_steps=1000,
        gradient_accumulation_steps=1
    ):
        """
        Fine-tune the model
        """
        print(f"\nStarting training...")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_epochs,
            predict_with_generate=True,
            generation_max_length=self.max_target_length,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            warmup_steps=warmup_steps,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        print(f"\nSaving final model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer
    
    def evaluate_model(self, test_dataset, trainer=None):
        """
        Evaluate the model on test set
        """
        if trainer is None:
            # Load the saved model
            print(f"\nLoading model from {self.output_dir}")
            self.model = MT5ForConditionalGeneration.from_pretrained(self.output_dir)
            self.tokenizer = MT5Tokenizer.from_pretrained(self.output_dir)
            
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.output_dir,
                predict_with_generate=True,
                generation_max_length=self.max_target_length,
                per_device_eval_batch_size=8,
            )
            
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                padding=True
            )
            
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )
        
        print("\nEvaluating on test set...")
        metrics = trainer.evaluate(test_dataset)
        
        print("\nTest Set Results:")
        for key, value in metrics.items():
            if key.startswith("eval_"):
                print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def generate_headline(self, article_text, num_beams=4, max_length=None):
        """
        Generate a headline for a given article
        """
        if max_length is None:
            max_length = self.max_target_length
        
        # Prepare input
        input_text = "summarize: " + article_text
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        # Decode
        headline = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return headline


def main():
    """
    Main training pipeline
    """
    # Configuration
    CSV_PATH = "../dataset/headlines_corpus.csv"  # Your CSV file path
    TEXT_COLUMN = "article"  # Column name with article text
    HEADLINE_COLUMN = "headline"  # Column name with headlines
    
    MODEL_NAME = "google/mt5-base"  # or "google/mt5-base" for better quality
    OUTPUT_DIR = "./burmese_headline_model"
    
    # Initialize trainer
    trainer_obj = BurmeseHeadlineTrainer(
        model_name=MODEL_NAME,
        max_input_length=512,
        max_target_length=128,
        output_dir=OUTPUT_DIR
    )
    
    # Load data
    df, text_col, headline_col = trainer_obj.load_data(
        CSV_PATH,
        text_column=TEXT_COLUMN,
        headline_column=HEADLINE_COLUMN
    )
    
    # Split data
    train_df, val_df, test_df = trainer_obj.split_data(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # Preprocess
    train_dataset = trainer_obj.preprocess_data(train_df, text_col, headline_col)
    val_dataset = trainer_obj.preprocess_data(val_df, text_col, headline_col)
    test_dataset = trainer_obj.preprocess_data(test_df, text_col, headline_col)
    
    # Train
    trainer = trainer_obj.train(
        train_dataset,
        val_dataset,
        num_epochs=10,
        batch_size=8,  # Adjust based on your GPU memory
        learning_rate=5e-5,
        save_steps=1000,
        eval_steps=1000
    )
    
    # Evaluate
    metrics = trainer_obj.evaluate_model(test_dataset, trainer)
    
    # Save metrics
    with open(f"{OUTPUT_DIR}/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Test generation
    print("\n" + "="*50)
    print("Testing headline generation:")
    print("="*50)
    
    sample_article = test_df[text_col].iloc[0]
    actual_headline = test_df[headline_col].iloc[0]
    generated_headline = trainer_obj.generate_headline(sample_article)
    
    print(f"\nArticle: {sample_article[:200]}...")
    print(f"\nActual headline: {actual_headline}")
    print(f"Generated headline: {generated_headline}")


if __name__ == "__main__":
    main()
