"""
Diagnostic Script: Find Why Model Isn't Learning

Run this in your Colab to identify the problem
"""

import torch
import numpy as np
import pandas as pd
from collections import Counter

print("="*60)
print("DIAGNOSTIC REPORT")
print("="*60)

# ============================================
# 1. DATA QUALITY CHECK
# ============================================
print("\n[1] DATA QUALITY CHECK")
print("-"*60)

# Load your data
df = pd.read_csv("D:\\NLP LMB\\Headline_Generator\\headline_corpus.csv")
texts = df["text"].astype(str).tolist()
headlines = df["headline"].astype(str).tolist()

print(f"✓ Total articles: {len(texts)}")
print(f"✓ Total headlines: {len(headlines)}")

# Check for problems
empty_texts = sum(1 for t in texts if len(t.strip()) < 10)
empty_headlines = sum(1 for h in headlines if len(h.strip()) < 3)

print(f"\n⚠️  Very short articles (<10 chars): {empty_texts}")
print(f"⚠️  Very short headlines (<3 chars): {empty_headlines}")

# Sample length analysis
text_lengths = [len(t.split()) for t in texts[:1000]]
headline_lengths = [len(h.split()) for h in headlines[:1000]]

print(f"\nArticle word count:")
print(f"  Mean: {np.mean(text_lengths):.1f} words")
print(f"  Median: {np.median(text_lengths):.1f} words")
print(f"  Max: {max(text_lengths)} words")

print(f"\nHeadline word count:")
print(f"  Mean: {np.mean(headline_lengths):.1f} words")
print(f"  Median: {np.median(headline_lengths):.1f} words")
print(f"  Max: {max(headline_lengths)} words")

# ============================================
# 2. VOCABULARY OVERLAP CHECK
# ============================================
print("\n[2] VOCABULARY OVERLAP CHECK")
print("-"*60)

# Get vocabularies
text_vocab = set()
headline_vocab = set()

for text in texts[:1000]:
    text_vocab.update(text.split())
    
for headline in headlines[:1000]:
    headline_vocab.update(headline.split())

overlap = text_vocab & headline_vocab
print(f"Text vocabulary: {len(text_vocab)} unique words")
print(f"Headline vocabulary: {len(headline_vocab)} unique words")
print(f"Overlap: {len(overlap)} words ({len(overlap)/len(headline_vocab)*100:.1f}% of headline vocab)")

if len(overlap) / len(headline_vocab) < 0.5:
    print("⚠️  WARNING: Low overlap! Headlines use very different words than articles.")
    print("   This makes learning difficult.")

# ============================================
# 3. MODEL SANITY CHECK
# ============================================
print("\n[3] MODEL CONFIGURATION CHECK")
print("-"*60)

# Check if model exists
try:
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Check if embeddings are frozen
    emb_frozen = not model.embedding.weight.requires_grad
    print(f"Embeddings frozen: {emb_frozen}")
    
except NameError:
    print("❌ Model not found! Run training cells first.")

# ============================================
# 4. TRAINING HISTORY CHECK
# ============================================
print("\n[4] TRAINING HISTORY CHECK")
print("-"*60)

try:
    import matplotlib.pyplot as plt
    
    if len(train_losses) > 0 and len(val_losses) > 0:
        print(f"Epochs trained: {len(train_losses)}")
        print(f"\nLoss progression:")
        print(f"  Epoch 1:  Train={train_losses[0]:.3f}, Val={val_losses[0]:.3f}")
        print(f"  Epoch 10: Train={train_losses[9]:.3f}, Val={val_losses[9]:.3f}" if len(train_losses) >= 10 else "")
        print(f"  Epoch 25: Train={train_losses[24]:.3f}, Val={val_losses[24]:.3f}" if len(train_losses) >= 25 else "")
        print(f"  Epoch 50: Train={train_losses[49]:.3f}, Val={val_losses[49]:.3f}" if len(train_losses) >= 50 else "")
        
        # Check if learning is happening
        if train_losses[0] - train_losses[-1] < 0.5:
            print("\n❌ PROBLEM: Training loss barely decreased!")
            print("   Model is NOT learning properly.")
        
        # Check overfitting
        gap = val_losses[-1] - train_losses[-1]
        if gap > 2.0:
            print(f"\n⚠️  WARNING: Large train/val gap ({gap:.2f})")
            print("   Model is overfitting!")
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()
        
except NameError:
    print("❌ Training history not found! Train model first.")

# ============================================
# 5. PREDICTION QUALITY CHECK
# ============================================
print("\n[5] PREDICTION QUALITY CHECK")
print("-"*60)

try:
    # Test on a few examples
    model.eval()
    
    test_indices = [0, 10, 20, 30, 40]
    
    for idx in test_indices:
        article = texts[idx][:200]  # First 200 chars
        actual_headline = headlines[idx]
        
        # Generate headline
        try:
            generated = generate_headline(model, article, beam_width=1, max_len=25)
            
            print(f"\nExample {idx}:")
            print(f"Article: {article}...")
            print(f"Actual:  {actual_headline}")
            print(f"Generated: {generated}")
            
            # Check if generating meaningful output
            if len(generated) < 3 or generated.count('<') > 0:
                print("⚠️  Generated headline is garbage!")
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            
except NameError:
    print("❌ Cannot test predictions. Model or function not found.")

# ============================================
# 6. GRADIENT CHECK
# ============================================
print("\n[6] GRADIENT FLOW CHECK")
print("-"*60)

try:
    model.train()
    
    # Get a batch
    src, dec_input, dec_target = next(iter(train_loader))
    src = src.to(DEVICE)
    dec_input = dec_input.to(DEVICE)
    dec_target = dec_target.to(DEVICE)
    
    # Forward pass
    optimizer.zero_grad()
    output = model(src, dec_input, teacher_forcing_ratio=0.5)
    loss = criterion(output.reshape(-1, vocab_size), dec_target.reshape(-1))
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    print("Gradient norms (should be > 0):")
    for name, norm in list(grad_norms.items())[:10]:
        status = "✓" if norm > 1e-6 else "❌"
        print(f"  {status} {name}: {norm:.6f}")
    
    # Check for vanishing gradients
    if all(norm < 1e-5 for norm in grad_norms.values()):
        print("\n❌ VANISHING GRADIENTS! All gradients are near zero.")
        print("   Model cannot learn!")
    
except Exception as e:
    print(f"❌ Gradient check failed: {e}")

# ============================================
# 7. LEARNING RATE CHECK
# ============================================
print("\n[7] LEARNING RATE CHECK")
print("-"*60)

try:
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")
    
    if current_lr < 1e-6:
        print("❌ Learning rate is TOO SMALL!")
        print("   Model will learn extremely slowly.")
    elif current_lr > 0.01:
        print("⚠️  Learning rate is quite large.")
        print("   May cause instability.")
    else:
        print("✓ Learning rate looks reasonable.")
        
except NameError:
    print("❌ Optimizer not found.")

# ============================================
# 8. RECOMMENDATIONS
# ============================================
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

recommendations = []

# Check dataset size
if len(texts) < 5000:
    recommendations.append("📊 COLLECT MORE DATA: You have < 5000 articles. Aim for 10,000+")

# Check loss value
if len(val_losses) > 0 and val_losses[-1] > 5.0:
    recommendations.append("🔧 REDUCE VOCABULARY: 150k words is too large. Try max_vocab=30,000")
    recommendations.append("📈 INCREASE LEARNING RATE: Try lr=0.003 instead of 0.001")
    recommendations.append("🎯 ADD LABEL SMOOTHING: Helps with large vocabulary")

# Check if loss is stuck
if len(train_losses) > 10:
    recent_change = abs(train_losses[-1] - train_losses[-10])
    if recent_change < 0.1:
        recommendations.append("💥 RESET AND RETRAIN: Loss is stuck, model hit local minimum")
        recommendations.append("🔀 TRY DIFFERENT INITIALIZATION: Use Xavier/Kaiming init")

# General recommendations
recommendations.append("✂️  SIMPLIFY MODEL: Try 1 layer instead of 2, hidden_dim=256")
recommendations.append("📝 CHECK PREPROCESSING: Verify text and headline tokenization")
recommendations.append("🎲 USE CURRICULUM LEARNING: Start with short articles first")

for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec}")

print("\n" + "="*60)
print("Run the fixes below based on recommendations above")
print("="*60)
