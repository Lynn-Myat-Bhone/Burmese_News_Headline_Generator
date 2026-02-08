"""
Burmese Headline Generation - Inference Script
Use the fine-tuned model to generate headlines
"""

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
import pandas as pd
from pathlib import Path

class HeadlineGenerator:
    def __init__(self, model_path="./burmese_headline_model"):
        """
        Load the fine-tuned model
        
        Args:
            model_path: Path to the saved model
        """
        print(f"Loading model from {model_path}")
        self.tokenizer = MT5Tokenizer.from_pretrained(model_path)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def generate_headline(
        self,
        article_text,
        num_beams=4,
        max_length=128,
        min_length=10,
        temperature=1.0,
        num_return_sequences=1
    ):
        """
        Generate headline(s) for an article
        
        Args:
            article_text: The article text
            num_beams: Beam search width (higher = better quality but slower)
            max_length: Maximum headline length
            min_length: Minimum headline length
            temperature: Sampling temperature (lower = more focused)
            num_return_sequences: Number of different headlines to generate
        
        Returns:
            Single headline string or list of headlines
        """
        # Prepare input
        input_text = "summarize: " + article_text
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False if num_beams > 1 else True
            )
        
        # Decode
        headlines = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return headlines[0] if num_return_sequences == 1 else headlines
    
    def batch_generate(self, articles, batch_size=8, **kwargs):
        """
        Generate headlines for multiple articles
        
        Args:
            articles: List of article texts
            batch_size: Number of articles to process at once
            **kwargs: Arguments for generate_headline
        
        Returns:
            List of generated headlines
        """
        headlines = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            batch_headlines = [
                self.generate_headline(article, **kwargs)
                for article in batch
            ]
            headlines.extend(batch_headlines)
            
            print(f"Processed {min(i+batch_size, len(articles))}/{len(articles)} articles")
        
        return headlines
    
    def generate_from_csv(
        self,
        input_csv,
        output_csv,
        text_column="article",
        **kwargs
    ):
        """
        Generate headlines for articles in a CSV file
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to output CSV file
            text_column: Name of column containing article text
            **kwargs: Arguments for generate_headline
        """
        print(f"Loading articles from {input_csv}")
        df = pd.read_csv(input_csv)
        
        articles = df[text_column].astype(str).tolist()
        
        print(f"Generating headlines for {len(articles)} articles...")
        headlines = self.batch_generate(articles, **kwargs)
        
        df['generated_headline'] = headlines
        
        print(f"Saving results to {output_csv}")
        df.to_csv(output_csv, index=False, encoding='utf-8')
        
        return df


def interactive_mode():
    """
    Interactive mode for testing the model
    """
    generator = HeadlineGenerator()
    
    print("\n" + "="*60)
    print("Burmese Headline Generator - Interactive Mode")
    print("="*60)
    print("Enter article text (or 'quit' to exit)")
    print("Options: Add '--multiple N' to generate N different headlines")
    print("         Add '--beams N' to use N beams (default: 4)")
    print("="*60 + "\n")
    
    while True:
        print("\nArticle text:")
        text = input("> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        # Parse options
        num_sequences = 1
        num_beams = 4
        
        if '--multiple' in text:
            parts = text.split('--multiple')
            text = parts[0].strip()
            try:
                num_sequences = int(parts[1].split()[0])
            except:
                num_sequences = 3
        
        if '--beams' in text:
            parts = text.split('--beams')
            text = parts[0].strip()
            try:
                num_beams = int(parts[1].split()[0])
            except:
                num_beams = 4
        
        # Generate
        headlines = generator.generate_headline(
            text,
            num_beams=num_beams,
            num_return_sequences=num_sequences
        )
        
        # Display
        print("\nGenerated headline(s):")
        if isinstance(headlines, list):
            for i, headline in enumerate(headlines, 1):
                print(f"  {i}. {headline}")
        else:
            print(f"  {headlines}")


def main():
    """
    Example usage
    """
    # Initialize generator
    generator = HeadlineGenerator(model_path="./burmese_headline_model")
    
    # Example 1: Generate single headline
    article = """
    သင့်ရဲ့ ဘာသာစကားအတွက် article text ကို ဒီမှာ ထည့်ပါ။
    This is where you would put your Burmese article text.
    """
    
    headline = generator.generate_headline(article)
    print(f"Article: {article[:100]}...")
    print(f"Headline: {headline}")
    
    # Example 2: Generate multiple variants
    print("\n" + "="*60)
    print("Generating multiple headline variants:")
    headlines = generator.generate_headline(
        article,
        num_return_sequences=3,
        num_beams=5
    )
    for i, h in enumerate(headlines, 1):
        print(f"{i}. {h}")
    
    # Example 3: Batch processing from CSV
    # generator.generate_from_csv(
    #     input_csv="test_articles.csv",
    #     output_csv="articles_with_headlines.csv",
    #     text_column="article",
    #     batch_size=8
    # )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
