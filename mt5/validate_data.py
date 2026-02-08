"""
Data Validation Script
Check your CSV data quality before training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

class DataValidator:
    def __init__(self, csv_path, text_column="article", headline_column="headline"):
        """
        Load and validate CSV data
        """
        print(f"Loading data from: {csv_path}")
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        self.text_col = text_column
        self.headline_col = headline_column
        
        print(f"Initial data size: {len(self.df)} rows")
        
    def check_columns(self):
        """
        Verify required columns exist
        """
        print("\n" + "="*60)
        print("COLUMN CHECK")
        print("="*60)
        
        available_columns = self.df.columns.tolist()
        print(f"Available columns: {available_columns}")
        
        if self.text_col not in available_columns:
            print(f"❌ ERROR: Column '{self.text_col}' not found!")
            print(f"   Please update TEXT_COLUMN to one of: {available_columns}")
            return False
        
        if self.headline_col not in available_columns:
            print(f"❌ ERROR: Column '{self.headline_col}' not found!")
            print(f"   Please update HEADLINE_COLUMN to one of: {available_columns}")
            return False
        
        print(f"✓ Text column '{self.text_col}' found")
        print(f"✓ Headline column '{self.headline_col}' found")
        return True
    
    def check_missing_values(self):
        """
        Check for missing values
        """
        print("\n" + "="*60)
        print("MISSING VALUES CHECK")
        print("="*60)
        
        text_missing = self.df[self.text_col].isna().sum()
        headline_missing = self.df[self.headline_col].isna().sum()
        
        print(f"Missing articles: {text_missing} ({text_missing/len(self.df)*100:.1f}%)")
        print(f"Missing headlines: {headline_missing} ({headline_missing/len(self.df)*100:.1f}%)")
        
        if text_missing > 0 or headline_missing > 0:
            print(f"⚠️  Will remove {max(text_missing, headline_missing)} rows during training")
        else:
            print("✓ No missing values")
    
    def check_empty_strings(self):
        """
        Check for empty strings after stripping
        """
        print("\n" + "="*60)
        print("EMPTY STRING CHECK")
        print("="*60)
        
        self.df[self.text_col] = self.df[self.text_col].astype(str).str.strip()
        self.df[self.headline_col] = self.df[self.headline_col].astype(str).str.strip()
        
        empty_articles = (self.df[self.text_col] == "").sum()
        empty_headlines = (self.df[self.headline_col] == "").sum()
        
        print(f"Empty articles: {empty_articles}")
        print(f"Empty headlines: {empty_headlines}")
        
        if empty_articles > 0 or empty_headlines > 0:
            print(f"⚠️  Will remove {max(empty_articles, empty_headlines)} empty entries")
        else:
            print("✓ No empty strings")
    
    def check_duplicates(self):
        """
        Check for duplicate entries
        """
        print("\n" + "="*60)
        print("DUPLICATE CHECK")
        print("="*60)
        
        article_dups = self.df[self.text_col].duplicated().sum()
        headline_dups = self.df[self.headline_col].duplicated().sum()
        both_dups = self.df.duplicated(subset=[self.text_col, self.headline_col]).sum()
        
        print(f"Duplicate articles: {article_dups}")
        print(f"Duplicate headlines: {headline_dups}")
        print(f"Duplicate pairs: {both_dups}")
        
        if both_dups > 0:
            print(f"⚠️  Consider removing {both_dups} duplicate entries")
        else:
            print("✓ No duplicate pairs")
    
    def analyze_lengths(self):
        """
        Analyze text lengths
        """
        print("\n" + "="*60)
        print("LENGTH ANALYSIS")
        print("="*60)
        
        # Character lengths
        article_lengths = self.df[self.text_col].str.len()
        headline_lengths = self.df[self.headline_col].str.len()
        
        # Word counts (approximate)
        article_words = self.df[self.text_col].str.split().str.len()
        headline_words = self.df[self.headline_col].str.split().str.len()
        
        print("\nArticle lengths (characters):")
        print(f"  Min: {article_lengths.min()}")
        print(f"  Max: {article_lengths.max()}")
        print(f"  Mean: {article_lengths.mean():.1f}")
        print(f"  Median: {article_lengths.median():.1f}")
        
        print("\nHeadline lengths (characters):")
        print(f"  Min: {headline_lengths.min()}")
        print(f"  Max: {headline_lengths.max()}")
        print(f"  Mean: {headline_lengths.mean():.1f}")
        print(f"  Median: {headline_lengths.median():.1f}")
        
        print("\nArticle lengths (words):")
        print(f"  Min: {article_words.min()}")
        print(f"  Max: {article_words.max()}")
        print(f"  Mean: {article_words.mean():.1f}")
        print(f"  Median: {article_words.median():.1f}")
        
        print("\nHeadline lengths (words):")
        print(f"  Min: {headline_words.min()}")
        print(f"  Max: {headline_words.max()}")
        print(f"  Mean: {headline_words.mean():.1f}")
        print(f"  Median: {headline_words.median():.1f}")
        
        # Warnings
        if article_lengths.max() > 2000:
            print(f"\n⚠️  Some articles are very long (>{article_lengths.max()} chars)")
            print("   Consider max_input_length=1024 for long articles")
        
        if headline_lengths.max() > 300:
            print(f"\n⚠️  Some headlines are very long (>{headline_lengths.max()} chars)")
            print("   This is unusual - verify your data")
        
        if headline_words.mean() < 3:
            print(f"\n⚠️  Headlines are very short (avg {headline_words.mean():.1f} words)")
            print("   Verify this is expected")
        
        return article_lengths, headline_lengths
    
    def check_language(self):
        """
        Basic check for Burmese text presence
        """
        print("\n" + "="*60)
        print("LANGUAGE CHECK")
        print("="*60)
        
        # Burmese unicode range: U+1000 to U+109F
        burmese_pattern = re.compile(r'[\u1000-\u109F]')
        
        articles_with_burmese = self.df[self.text_col].str.contains(
            burmese_pattern, regex=True, na=False
        ).sum()
        
        headlines_with_burmese = self.df[self.headline_col].str.contains(
            burmese_pattern, regex=True, na=False
        ).sum()
        
        total = len(self.df)
        
        print(f"Articles with Burmese script: {articles_with_burmese}/{total} ({articles_with_burmese/total*100:.1f}%)")
        print(f"Headlines with Burmese script: {headlines_with_burmese}/{total} ({headlines_with_burmese/total*100:.1f}%)")
        
        if articles_with_burmese < total * 0.5:
            print("\n⚠️  Warning: Less than 50% of articles contain Burmese script")
            print("   Make sure you're using the correct columns")
        else:
            print("✓ Burmese text detected")
    
    def show_samples(self, n=3):
        """
        Display sample entries
        """
        print("\n" + "="*60)
        print(f"SAMPLE DATA (showing {n} examples)")
        print("="*60)
        
        for i in range(min(n, len(self.df))):
            print(f"\n--- Example {i+1} ---")
            article = self.df[self.text_col].iloc[i]
            headline = self.df[self.headline_col].iloc[i]
            
            print(f"Article ({len(article)} chars):")
            print(f"  {article[:200]}{'...' if len(article) > 200 else ''}")
            print(f"\nHeadline ({len(headline)} chars):")
            print(f"  {headline}")
    
    def generate_report(self):
        """
        Generate a complete validation report
        """
        print("\n" + "="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        print(f"Dataset: {len(self.df)} samples\n")
        
        # Run all checks
        if not self.check_columns():
            print("\n❌ Fix column names before proceeding!")
            return False
        
        self.check_missing_values()
        self.check_empty_strings()
        self.check_duplicates()
        self.analyze_lengths()
        self.check_language()
        self.show_samples()
        
        # Final recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Clean data count
        clean_df = self.df[
            (self.df[self.text_col].notna()) & 
            (self.df[self.headline_col].notna()) &
            (self.df[self.text_col] != "") &
            (self.df[self.headline_col] != "")
        ]
        
        original_size = len(self.df)
        clean_size = len(clean_df)
        
        print(f"\nClean data after removing missing/empty: {clean_size}/{original_size}")
        
        if clean_size < 1000:
            print("⚠️  Dataset is quite small (<1000 samples)")
            print("   Consider collecting more data for better results")
        elif clean_size < 5000:
            print("✓ Dataset size is adequate (1K-5K samples)")
            print("  Should work well with mt5-small")
        elif clean_size < 15000:
            print("✓ Good dataset size (5K-15K samples)")
            print("  Can try mt5-base for better quality")
        else:
            print("✓ Excellent dataset size (>15K samples)")
            print("  Recommended: Start with mt5-small, then try mt5-base")
        
        # Save clean data
        if clean_size < original_size:
            clean_output = "cleaned_data.csv"
            clean_df.to_csv(clean_output, index=False, encoding='utf-8')
            print(f"\n✓ Saved cleaned data to: {clean_output}")
            print(f"  Use this file for training to avoid issues")
        
        print("\n" + "="*60)
        print("✓ Validation complete!")
        print("="*60)
        
        return True


def main():
    """
    Run validation on your dataset
    """
    # Configuration - UPDATE THESE
    CSV_PATH = "burmese_headlines.csv"  # Your CSV file
    TEXT_COLUMN = "article"              # Column with article text
    HEADLINE_COLUMN = "headline"         # Column with headlines
    
    # Run validation
    validator = DataValidator(CSV_PATH, TEXT_COLUMN, HEADLINE_COLUMN)
    validator.generate_report()


if __name__ == "__main__":
    main()
