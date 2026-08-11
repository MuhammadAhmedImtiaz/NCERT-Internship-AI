# =====================================================================
# PKCERT AI & SOFTWARE DEVELOPMENT INTERNSHIP
# TASK 26: INTRODUCTION TO TRANSFORMERS & ATTENTION MECHANISM
# =====================================================================

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Reproducibility Config
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Execution Device: {device}")

# =====================================================================
# PART C: PRETRAINED TRANSFORMER INFERENCE & ATTENTION EXTRACTION
# =====================================================================
print("""
===================================================================
PART C: TRANSFORMER ANALYSIS & ATTENTION VISUALIZATION (DISTILBERT)
===================================================================
""")

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Load Tokenizer & Model with output_attentions=True
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, output_attentions=True).to(device)
model.eval()

# 1. Sample Sequence Attention Extraction
sample_text = "The Transformer model completely revolutionized natural language processing."
inputs = tokenizer(sample_text, return_tensors="pt").to(device)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = model(**inputs)
    # Attentions shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    attentions = outputs.attentions

print(f"Input Tokens: {tokens}")
print(f"Extracted Attention Layers Count: {len(attentions)}")
print(f"Single Layer Attention Matrix Shape: {attentions[0].shape}")

# Visualize Attention Map from Layer 1, Head 1
layer_idx = 0
head_idx = 0
attn_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()

plt.figure(figsize=(8, 6))
sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="Blues", annot=True, fmt=".2f", cbar=True)
plt.title(f"Self-Attention Map (DistilBERT Layer {layer_idx+1}, Head {head_idx+1})", fontsize=11, fontweight='bold')
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.tight_layout()
plt.show()

# 2. Evaluation Benchmark on Sample Held-Out Evaluation Dataset
dataset = [
    ("This film was absolute perfection and brilliantly acted.", 1),
    ("Terrible story, boring characters, and completely predictable.", 0),
    ("A masterpiece of modern cinema with amazing visual effects.", 1),
    ("Waste of money, I hated every single minute of it.", 0),
    ("Deeply moving, emotional, and wonderful direction.", 1),
    ("Dull, sluggish, and completely lacks any genuine tone.", 0),
    ("An exhilarating experience with brilliant performances.", 1),
    ("Poorly written script with terrible dialogue.", 0)
]

texts, labels = zip(*dataset)

start_time = time.time()
eval_inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(device)

with torch.no_grad():
    eval_outputs = model(**eval_inputs)
    logits = eval_outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()

inference_time = time.time() - start_time

acc  = accuracy_score(labels, preds)
prec = precision_score(labels, preds)
rec  = recall_score(labels, preds)
f1   = f1_score(labels, preds)

print(f"\n--- Transformer Evaluation Metrics ---")
print(f"Test Accuracy:  {acc*100:.2f}%")
print(f"Precision:      {prec:.4f}")
print(f"Recall:         {rec:.4f}")
print(f"F1-Score:       {f1:.4f}")
print(f"Inference Time: {inference_time:.4f} seconds")

# Benchmark Matrix vs Task 25 LSTM
comparison_df = pd.DataFrame([
    {"Architecture": "Task 25 Bi-LSTM Baseline", "Accuracy": "85.00%", "Macro F1": 0.8450, "Sequential Bottleneck": "Yes (O(N) recurrence)"},
    {"Architecture": "Task 26 Pretrained Transformer (DistilBERT)", "Accuracy": f"{acc*100:.2f}%", "Macro F1": round(f1, 4), "Sequential Bottleneck": "No (Parallel Self-Attention)"}
])

print("\n--- Comparative Architecture Matrix ---")
print(comparison_df.to_string(index=False))

# Persist Summary
comparison_df.to_csv("transformer_vs_lstm_summary.csv", index=False)
print("\nSaved comparison table to 'transformer_vs_lstm_summary.csv'!")